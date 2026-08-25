from __future__ import annotations
import inspect
import sys
from collections.abc import (Collection, Iterable, Iterator, Mapping, MutableMapping, MutableSequence, MutableSet, Sequence, Set as AbstractSet,)
from dataclasses import InitVar, is_dataclass, MISSING, Field
from types import UnionType
from typing import (Annotated, ForwardRef, Literal, TypeVar, Union, get_args, get_origin, Any, get_type_hints)

T = TypeVar("T")
UNIONS = {Union, UnionType}
SEQUENCES = {Iterable, Iterator, Collection, Sequence, MutableSequence}
MAPPINGS = {Mapping, MutableMapping}
SETS = {AbstractSet, MutableSet}
STRINGS = (str, bytes, bytearray)


class ConversionError(TypeError):
    pass


def _caller_locals() -> dict[str, Any]:
    """Return the local variables of the function calling the public helper."""
    frame = inspect.currentframe()
    try:
        return dict(frame.f_back.f_back.f_locals)
    finally:
        del frame


def _resolve_hints(cls: type, localns: Mapping[str, Any]) -> dict[str, Any]:
    localns = {**localns, cls.__name__: cls}
    try:
        return get_type_hints(
            cls,
            globalns=vars(sys.modules[cls.__module__]),
            localns=localns,
            include_extras=True,
        )
    except (NameError, TypeError) as error:
        raise ConversionError(
            f"Could not resolve annotations for {cls.__qualname__}. "
            "Pass localns explicitly if a referenced local type is not available in the caller's scope.") from error


def matches_type(value: Any, expected_type: Any) -> bool:
    """Return whether a value already fully satisfies a type annotation."""
    if expected_type is Any:
        return True
    origin = get_origin(expected_type)
    arguments = get_args(expected_type)
    if origin is Annotated:
        return matches_type(value, arguments[0])
    if origin in UNIONS:
        return any(matches_type(value, option) for option in arguments)
    if origin is Literal:
        return any(type(value) is type(option) and value == option for option in arguments)
    if expected_type is type(None):
        return value is None
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if not arguments:
            return True
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return all(matches_type(item, arguments[0]) for item in value)
        return len(value) == len(arguments) and all(matches_type(item, item_type) for item, item_type in zip(value, arguments))
    if origin in MAPPINGS | {dict}:
        key_type, item_type = arguments or (Any, Any)
        return isinstance(value, origin) and all(matches_type(key, key_type) and matches_type(item, item_type) for key, item in value.items())
    if origin in SETS | {set, frozenset}:
        item_type = arguments[0] if arguments else Any
        return isinstance(value, origin) and all(matches_type(item, item_type) for item in value)
    if origin in SEQUENCES | {list}:
        if not isinstance(value, origin) or isinstance(value, STRINGS):
            return False
        item_type = arguments[0] if arguments else Any
        # Do not consume one-shot iterators just to validate them.
        if isinstance(value, Iterator):
            return item_type is Any
        return all(matches_type(item, item_type) for item in value)
    return isinstance(value, expected_type) if isinstance(expected_type, type) else False


def convert_value(value: Any, expected_type: Any, *, localns: Mapping[str, Any], path: str = "value") -> Any:
    """
    Convert a value according to a resolved type annotation.
    Union members are attempted from left to right.
    """
    if isinstance(expected_type, InitVar):
        expected_type = expected_type.type
    if isinstance(expected_type, (str, ForwardRef)):
        raise ConversionError(f"{path}: unresolved forward reference {expected_type!r}")
    origin = get_origin(expected_type)
    arguments = get_args(expected_type)
    if expected_type is Any:
        return value
    if origin is Annotated:
        return convert_value(value, arguments[0], localns=localns, path=path)
    if origin in UNIONS:
        errors = []
        for option in arguments:
            try:
                return convert_value(value, option, localns=localns, path=path,)
            except (TypeError, ValueError) as error:
                errors.append(str(error))
        raise ConversionError(f"{path}: {value!r} cannot become {expected_type}: " + "; ".join(errors))
    if origin is Literal:
        for option in arguments:
            try:
                result = convert_value(value, type(option), localns=localns, path=path,)
                if type(result) is type(option) and result == option:
                    return option
            except (TypeError, ValueError):
                pass
        raise ConversionError(f"{path}: {value!r} is not one of {arguments}")
    if expected_type is type(None):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        raise ConversionError(f"{path}: expected None")
    if matches_type(value, expected_type):
        return value
    if isinstance(expected_type, type) and is_dataclass(expected_type):
        if not isinstance(value, Mapping):
            raise ConversionError(f"{path}: expected a mapping for {expected_type.__name__}")
        return dataclass_from_mapping(expected_type, value, localns=localns, _path=path)
    if origin is tuple:
        if isinstance(value, STRINGS) or not isinstance(value, Iterable):
            raise ConversionError(f"{path}: expected a non-string iterable")
        value = tuple(value)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            item_types = (arguments[0],) * len(value)
        elif not arguments:
            item_types = (Any,) * len(value)
        elif len(value) != len(arguments):
            raise ConversionError(f"{path}: expected {len(arguments)} tuple items, got {len(value)}")
        else:
            item_types = arguments
        return tuple(convert_value(item, item_type, localns=localns, path=f"{path}[{index}]",) 
                     for index, (item, item_type) in enumerate(zip(value, item_types)))
    if origin in MAPPINGS | {dict}:
        if not isinstance(value, Mapping):
            raise ConversionError(f"{path}: expected a mapping")
        key_type, item_type = arguments or (Any, Any)
        return {convert_value(key, key_type, localns=localns, path=f"{path}.<key>",): 
                convert_value(item, item_type, localns=localns, path=f"{path}[{key!r}]",)
                for key, item in value.items()}
    if origin in SETS | {set, frozenset}:
        if isinstance(value, STRINGS) or not isinstance(value, Iterable):
            raise ConversionError(f"{path}: expected a non-string iterable")
        item_type = arguments[0] if arguments else Any
        constructor = (frozenset if origin is frozenset else set)
        return constructor(convert_value(item, item_type, localns=localns, path=f"{path}[{index}]",) 
                           for index, item in enumerate(value))
    if origin in SEQUENCES | {list}:
        if isinstance(value, STRINGS) or not isinstance(value, Iterable):
            raise ConversionError(f"{path}: expected a non-string iterable")
        item_type = arguments[0] if arguments else Any
        return [convert_value(item, item_type, localns=localns, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if expected_type is bool:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
        if type(value) in (int, float) and value in (0, 1):
            return bool(value)
        raise ConversionError(f"{path}: cannot convert {value!r} to bool")
    if isinstance(expected_type, type):
        try:
            return expected_type(value)
        except (TypeError, ValueError) as error:
            raise ConversionError(f"{path}: cannot convert {value!r} to {expected_type.__name__}") from error
    raise ConversionError(f"{path}: unsupported annotation {expected_type!r}")


def dataclass_from_mapping(cls: type[T], values: Mapping[str, Any], *, localns: Mapping[str, Any] | None = None,
                           reject_unknown: bool = True, _path: str | None = None) -> T:
    """
    Create a dataclass from a mapping.
    When localns is omitted, the immediate caller's locals are used.
    """
    if not isinstance(cls, type) or not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")
    if not isinstance(values, Mapping):
        raise TypeError(f"Expected a mapping, got {type(values).__name__}")
    localns = (_caller_locals() if localns is None else dict(localns))
    hints = _resolve_hints(cls, localns)
    parameters = inspect.signature(cls).parameters
    path = _path or cls.__qualname__
    unknown = set(values) - set(parameters)
    if reject_unknown and unknown:
        names = ", ".join(sorted(map(str, unknown)))
        raise ConversionError(f"{path}: unknown field(s): {names}")
    converted = {
        name: convert_value(value, hints.get(name, parameters[name].annotation), localns=localns, path=f"{path}.{name}")
        for name, value in values.items()
        if name in parameters
    }
    try:
        return cls(**converted)
    except TypeError as error:
        raise ConversionError(f"{path}: could not construct {cls.__qualname__}: {error}") from error


def dataclass_from_args(cls: type[T], *args: Any, localns: Mapping[str, Any] | None = None,
                        reject_unknown: bool = True, **kwargs: Any,) -> T:
    """
    Constructor-like version of dataclass_from_mapping.
    Example:
        dataclass_from_args(SomeClass, "16")
    """
    localns = _caller_locals() if localns is None else dict(localns)
    positional_names = [parameter.name for parameter in inspect.signature(cls).parameters.values()
                        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD,)]
    if len(args) > len(positional_names):
        raise ConversionError(f"{cls.__qualname__}: too many positional arguments")
    for name, value in zip(positional_names, args):
        if name in kwargs:
            raise ConversionError(f"{cls.__qualname__}: multiple values for {name!r}")
        kwargs[name] = value
    return dataclass_from_mapping(cls, kwargs, localns=localns, reject_unknown=reject_unknown)


def infer_field_type(name: str, value: Any) -> Any | None:
    """
    Infer a dataclass annotation from a default value.

    Returns None for attributes that should not become dataclass fields.
    """
    def is_derived(v) -> bool:
        return getattr(value, '__is_derived_field__', False)

    # Methods, properties, nested classes, etc. should not become fields.
    if isinstance(value, (staticmethod, classmethod, property)):
        return None

    # Handle derived fields separately.
    if is_derived(value):
        # if value.result_type is not None:
        #     return value.result_type

        try:
            return_type = get_type_hints(value.factory).get("return")
        except (NameError, TypeError):
            return_type = None

        if return_type is None:
            raise TypeError(
                f"Cannot infer the type of derived field {name!r}. "
                "Provide result_type=... or an explicit annotation."
            )

        return return_type

    if callable(value):
        return None

    # Handle dataclasses.field(...).
    if isinstance(value, Field):
        if value.default is not MISSING:
            if value.default is None:
                raise TypeError(f"Cannot infer the type of field {name!r} from None. Provide an explicit annotation.")
            return type(value.default)
        if value.default_factory is not MISSING:
            factory = value.default_factory
            # field(default_factory=list) -> list
            if isinstance(factory, type):
                return factory
            try:
                return_type = get_type_hints(factory).get("return")
            except (NameError, TypeError):
                return_type = None
            if return_type is None:
                raise TypeError(
                    f"Cannot infer the type of field {name!r} from its "
                    "default_factory. Annotate the field or annotate the "
                    "factory's return type."
                )
            return return_type
        return None

    if value is None:
        raise TypeError(
            f"Cannot infer the type of field {name!r} from None. "
            "Provide an explicit annotation such as "
            f"{name}: str | None = None."
        )

    return type(value)


