import torch
from datetime import timedelta
import functools
from collections.abc import Mapping
from pathlib import Path
import abc
from dataclasses import MISSING, dataclass, fields
import sys
from collections.abc import Sequence
import inspect
from typing import Any, Callable
try:
    import annotationlib
except ImportError:
    annotationlib = None


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def no_grad(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            result = func(*args, **kwargs)
            return result

    return wrapper

def to_device(obj, device=device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, Mapping):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    return obj

def infer_batch_size(batch) -> int:
    if torch.is_tensor(batch):
        return batch.shape[0]
    if isinstance(batch, dict):
        for value in batch.values():
            if torch.is_tensor(value):
                return value.shape[0]
    if isinstance(batch, (tuple, list)):
        for value in batch:
            if torch.is_tensor(value):
                return value.shape[0]
    return 1

def data_iterator(dataloader: torch.utils.data.dataloader.DataLoader, device=device):
    for batch_sample in dataloader:
        yield to_device(batch_sample, device)

def validate_optimizer_device(model, optimizer):
    parameters = list(model.parameters())
    if not parameters:
        return
    param_device = parameters[0].device
    model_parameter_ids = {id(parameter) for parameter in parameters}

    if any(parameter.device != param_device for parameter in parameters):
        raise ValueError("Model parameters are on multiple devices.")

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in model_parameter_ids:
                raise ValueError("Optimizer contains a parameter not belonging to the model.")
            if parameter.device != param_device:
                raise ValueError(f"Model is on {param_device}, but an optimizer parameter is on {parameter.device}.")
            for state in optimizer.state.get(parameter, {}).values():
                if torch.is_tensor(state) and state.device != param_device:
                    raise ValueError(f"Model is on {param_device}, but optimizer state is on {state.device}.")

def create_loaders(training, *other_datasets, batch_size=32, number_of_workers=2):
    result = []
    pin_memory = device.type == "cuda"
    for index, dataset in enumerate([training, *other_datasets]):
        result.append(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True if index == 0 else False,
                # Drop last if training otherwise keep last
                drop_last=True if index == 0 else False,
                pin_memory=pin_memory,
                num_workers=number_of_workers,
                persistent_workers=number_of_workers > 0,
            )
        )
    return result

def format_seconds_into_time(seconds: int | float) -> str:
    return str(timedelta(seconds=seconds)).rstrip("0").rstrip(".")

def get_valid_argument(argument):
    name, separator, value = argument.partition("=")
    if not name:
        raise ValueError("Argument name cannot be empty")
    if not separator:
        raise ValueError(f"Argument {argument!r} must use name=value")
    return name, separator, value

def parse_keyword_arguments(args: Sequence[str] | None = None,) -> dict[str, str]:
    """
    Parse arbitrary command-line arguments in the form:
        --name=value
    Hyphens in names are converted to underscores:
        --learning-rate=0.001
        becomes
        {"learning_rate": "0.001"}
    """
    args = sys.argv[1:] if args is None else args
    result: dict[str, str] = {}
    for argument in args:
        if not argument.startswith("--"):
            raise ValueError(f"Invalid argument {argument!r}; arguments must begin with '--'")
        # Slice [2:] is to remove the dashes -- before the argument leaving only name=value
        name, separator, value = get_valid_argument(argument[2:])
        var_name = name.replace("-", "_")
        if var_name in result:
            raise ValueError(f"Argument --{name} was provided twice")
        result[var_name] = value
    return result

def parse_config_file(config_file: Path | str) -> dict[str, str]:
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f'Looked for config file but no such file exists: {config_file}')
    result = {}
    with open(config_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, separator, value = get_valid_argument(line)
            var_name = name.replace("-", "_")
            if var_name in result:
                raise ValueError(f"Argument --{name} was provided twice")
            result[var_name] = value
    return result

class OnlyKwargsAndDataclass(abc.ABCMeta):
    def __new__(mcls, name, bases, namespace, **kwargs):
        annotate, inferred_annotations = mcls.prepare_annotations(bases, namespace)
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if annotate is not None:
            mcls.install_annotation_wrapper(cls, annotate, inferred_annotations)
        # Automatically make the base and every subclass a dataclass.
        cls = mcls.convert_to_dataclass(cls)
        # mcls.check_for_non_default_fields(cls)
        return cls

    def convert_to_dataclass(cls):
        try:
            return dataclass(cls)
        except TypeError as error:
            raise TypeError( f"Could not create {cls.__qualname__} as a dataclass: {error}") from error

    def check_for_non_default_fields(cls):
        missing_defaults = [dataclass_field.name for dataclass_field in fields(cls)
                            if dataclass_field.default is MISSING and dataclass_field.default_factory is MISSING]
        if missing_defaults:
            raise TypeError(f"{cls.__qualname__} must give every dataclass field a default or default_factory. "
                            f"Missing defaults: {', '.join(missing_defaults)}")

    @classmethod
    def prepare_annotations(mcls, bases, namespace):
        from data_type_inference import infer_field_type

        own_annotations, annotate = mcls.get_namespace_annotations(namespace)
        known_annotations = {**mcls.get_inherited_annotations(bases), **own_annotations}
        inferred_annotations = {}

        for attribute_name, value in tuple(namespace.items()):
            if attribute_name.startswith("_") or attribute_name in known_annotations:
                continue
            if (inferred_type := infer_field_type(attribute_name, value)) is None:
                continue

            own_annotations[attribute_name] = inferred_type
            known_annotations[attribute_name] = inferred_type
            inferred_annotations[attribute_name] = inferred_type

        if annotate is None: namespace["__annotations__"] = own_annotations
        return annotate, inferred_annotations

    @staticmethod
    def get_namespace_annotations(namespace) -> tuple[dict[str, Any], Any | None]:
        if "__annotations__" in namespace: return dict(namespace["__annotations__"]), None
        if annotationlib and (annotate := annotationlib.get_annotate_from_class_namespace(namespace)):
            annotations = annotationlib.call_annotate_function(annotate, annotationlib.Format.FORWARDREF)
            return dict(annotations), annotate
        return {}, None

    @staticmethod
    def get_inherited_annotations(bases) -> dict[str, Any]:
        annotations = {}
        for base in bases:
            for ancestor in reversed(base.__mro__[:-1]):
                current = (
                    annotationlib.get_annotations(ancestor, format=annotationlib.Format.FORWARDREF)
                    if annotationlib else inspect.get_annotations(ancestor, eval_str=False)
                )
                annotations.update(current)

        return annotations

    @staticmethod
    def install_annotation_wrapper(cls, annotate, inferred_annotations):
        def wrapped_annotate(format):
            annotations = annotationlib.call_annotate_function(annotate, format, owner=cls)
            additions = annotationlib.annotations_to_string(inferred_annotations) if format == annotationlib.Format.STRING else inferred_annotations
            return {**annotations, **additions}

        cls.__annotate__ = wrapped_annotate

class CommandLineArguments(abc.ABC, metaclass=OnlyKwargsAndDataclass):
    @property
    def cls(self):
        return type(self)

    @classmethod
    def from_mapping(cls, mapping, *args, **kwargs):
        from data_type_inference import dataclass_from_mapping
        return dataclass_from_mapping(cls, mapping, *args, **kwargs)

    @classmethod
    def from_command_line_arguments(cls, *args, command_line_arguments=None, **kwargs):
        mapping = parse_keyword_arguments(command_line_arguments)
        return cls.from_mapping(mapping, *args, **kwargs)

    @classmethod
    def from_command_line(cls, *args, command_line_arguments=None, **kwargs):
        """
        Use argument `--config-file=PATH` to load the arguments from a config file.
        Otherwise, the arguments will be loaded from the command line as is.
        """
        mapping = parse_keyword_arguments(command_line_arguments)
        if mapping.get('config_file') is not None:
            return cls.from_config_file(mapping['config_file'])
        return cls.from_mapping(mapping, *args, **kwargs)

    @classmethod
    def from_config_file(cls, config_file: str | Path, *args, **kwargs):
        mapping = parse_config_file(config_file)
        return cls.from_mapping(mapping, *args, **kwargs)

class derived:
    __is_derived_field__ = True

    def __init__(self, factory: Callable[[Any], Any]):
        self.factory = factory

    def __set_name__(self, owner: type, name: str) -> None:
        self.storage_name = f"__derived_{name}"

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        # Needed for dataclass instantiation
        if instance is None:
            return self
        # Use the override if it exists
        if self.storage_name in instance.__dict__:
            return instance.__dict__[self.storage_name]
        # Use the function
        return self.factory(instance)

    def __set__(self, instance: Any, value: Any) -> None:
        # Set an override if it is not just the same value
        if value is not self:
            instance.__dict__[self.storage_name] = value

    def __repr__(self) -> str:
        return '<derived>'


