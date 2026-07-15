from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import torch.utils.data
import functools
import torch
from collections.abc import Mapping


# <editor-fold desc="Utility Functions">
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def create_loaders(training, *other_datasets, batch_size=32, shuffle=True):
    result = []
    for index, dataset in enumerate([training, *other_datasets]):
        result.append(torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            # Drop last if training otherwise keep last
            drop_last=True if index == 0 else False,))

# </editor-fold>


# <editor-fold desc="Slot-in Functions">

def update_schedulers(args: 'Arguments', epoch: int):
    for scheduler in args.schedulers:
        scheduler.step()

# </editor-fold>


@dataclass
class Arguments:
    model: torch.nn.Module
    # Either a common loss function from torch.nn.modules.loss or a custom one cf https://saturncloud.io/blog/custom-loss-function-in-pytorch-a-comprehensive-guide/
    loss_function: 'Loss Object'
    optimizer: torch.optim.Optimizer
    training_set: torch.utils.data.dataset.Subset
    validation_set: torch.utils.data.dataset.Subset
    training_loader: torch.utils.data.dataloader.DataLoader
    validation_loader: torch.utils.data.dataloader.DataLoader
    max_epochs: int
    # Training Saves will be save path + '_training'
    save_path: Path
    # Function that uses this arguments class and the current epoch to do whatever you want:
    #   if you have schedulers you must step them in the epochal update
    #   any other argument parameters that you want to update you may do so if you would like
    #   or if you would like a read-out per epoch you can put that here
    epochal_update: Callable[['Arguments', int], None] = update_schedulers
    stop_condition: Callable[['Arguments', int], bool] = lambda args, epoch: False
    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = field(default_factory=lambda: [])

    from_existing_model: bool = False

    device: torch.device = device
    best_validation_loss: float = float("inf")
    start_epoch: int = 0

    _epochal_validation_loss: int = field(default=0, init=False)
    _epochal_num_of_items_evaluated: int = field(default=0, init=False)

    kwargs: dict = field(default_factory=dict)

    @property
    def checkpoint_path(self) -> Path:
        return Path(str(self.save_path) + "_training")

    @property
    def epochal_validation_mean_loss(self):
        if self._epochal_num_of_items_evaluated > 0:
            return self._epochal_validation_loss / self._epochal_num_of_items_evaluated
        return -1

    def update_epochal_validation_loss(self, batch_loss: torch.Tensor, batch_size: int):
        self._epochal_num_of_items_evaluated += batch_size
        # Only works for mean loss aggregate
        self._epochal_validation_loss += batch_loss.item() * batch_size

    def __post_init__(self):

        self.device = torch.device(self.device)

        model_devices = {parameter.device for parameter in self.model.parameters()}
        model_devices.update(buffer.device for buffer in self.model.buffers())
        if model_devices and model_devices != {self.device}:
            raise ValueError(
                f"Model parameters/buffers are on {model_devices}, "
                f"but training device is {self.device}. "
                "Move the model before constructing the optimizer."
            )

        if self.from_existing_model:
            load_training_checkpoint(self)
        validate_optimizer_device(self.model, self.optimizer)

    # noinspection PyAttributeOutsideInit
    def start_evaluating(self):
        self.model.eval()
        self._epochal_validation_loss = 0
        self._epochal_num_of_items_evaluated = 0


# <editor-fold desc="Loop Sub-functions">
def train(args: Arguments, epoch):
    args.model.train()
    for problems, labels in data_iterator(args.training_loader, args.device):
        args.optimizer.zero_grad(set_to_none=True)
        outputs = args.model(problems)
        loss = args.loss_function(outputs, labels)
        loss.backward()
        args.optimizer.step()  # Maybe make the above a closure?


def validate(args: Arguments, epoch):
    args.start_evaluating()
    with torch.inference_mode():
        for problems, labels in data_iterator(args.validation_loader, args.device):
            outputs = args.model(problems)
            loss = args.loss_function(outputs, labels)
            args.update_epochal_validation_loss(loss, infer_batch_size(problems))

def save_model_for_inference(model: torch.nn.Module, path):
    torch.save(model.state_dict(), path)

def save_model_for_training(args: Arguments, epoch):
    torch.save({
        "epoch": epoch,
        "model_state_dict": args.model.state_dict(),
        "optimizer_state_dict": args.optimizer.state_dict(),
        "scheduler_state_dicts": [scheduler.state_dict() for scheduler in args.schedulers],
        "best_validation_loss": args.best_validation_loss,
    },
        args.checkpoint_path
    )

def save_logic(args: Arguments, epoch):
    if (args.epochal_validation_mean_loss < args.best_validation_loss) or (args.best_validation_loss == float('inf')):
        args.best_validation_loss = args.epochal_validation_mean_loss
        # save_model_for_inference(args.model, args.save_path)
        save_model_for_training(args, epoch)

def load_training_checkpoint(args: Arguments):
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Training checkpoint not found: {args.checkpoint_path}"
        )
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)
    args.model.load_state_dict(checkpoint["model_state_dict"])
    args.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    saved_scheduler_states = checkpoint.get("scheduler_state_dicts", [])

    if len(saved_scheduler_states) != len(args.schedulers):
        raise ValueError(f"The checkpoint contains {len(saved_scheduler_states)} scheduler states, but Arguments contains {len(args.schedulers)} schedulers.")

    for scheduler, scheduler_state in zip(args.schedulers, saved_scheduler_states):
        scheduler.load_state_dict(scheduler_state)

    args.best_validation_loss = checkpoint.get("best_validation_loss", float("inf"))
    # The saved epoch has already completed, so resume from the next one.
    args.start_epoch = checkpoint["epoch"] + 1
    print(
        f"Resumed training from epoch {args.start_epoch}. "
        f"Best validation loss: {args.best_validation_loss:.6f}"
    )
# </editor-fold>


def loop(args: Arguments):
    for epoch in range(args.start_epoch, args.max_epochs):
        train(args, epoch)
        validate(args, epoch)
        args.epochal_update(args, epoch)
        save_logic(args, epoch)
        if args.stop_condition(args, epoch):
            break


