import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import torch.utils.data
import functools
import torch
from collections.abc import Mapping
import time


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

def create_loaders(training, *other_datasets, batch_size=32):
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
                pin_memory=pin_memory
            )
        )
    return result

# </editor-fold>


# <editor-fold desc="Slot-in Functions">
def early_stop(
        patience: int = 5,
        minimum_loss_improvement: float = 0.0,
        initial_epoch_buffer: int = 100,
        print_result: bool = True,
        counter_key: str = "_epochs_without_improvement",
):
    def stop_condition(args: "Arguments", epoch: int) -> bool:
        if not math.isfinite(args.epochal_validation_mean_loss):
            raise RuntimeError(
                f"Validation loss became nonfinite at epoch {epoch + 1}: "
                f"{args.epochal_validation_mean_loss}"
            )
        if args.epochal_validation_mean_loss < args.best_validation_loss - minimum_loss_improvement:
            # Best Loss
            args.kwargs[counter_key] = 0
            # Num Epochs Without Improvement
        else:
            args.kwargs[counter_key] = args.kwargs.get(counter_key, 0) + 1

        epochs_without_improvement = args.kwargs[counter_key]
        if epochs_without_improvement >= patience and epoch >= initial_epoch_buffer:
            if print_result:
                print(f"Early stopping at epoch {epoch + 1}: validation loss did not improve for {patience} epochs. "
                      f"Best validation loss: {args.best_validation_loss:.6f}")
            return True

        return False

    return stop_condition

def update_schedulers(args: "Arguments", epoch: int):
    for scheduler in args.schedulers:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(args.epochal_validation_mean_loss)
        else:
            scheduler.step()

def epoch_logger(args: "Arguments", epoch: int) -> list[str]:
    """
    epoch #/ total,
    train loss,
    val loss,
    best val loss,
    loss gap,
    val delta loss
    is new best or # without improvement,
    learning rates,
    epoch time,
    avg time,
    total time,
    """
    current_val_loss = args.epochal_validation_mean_loss
    current_training_loss = args.epochal_training_mean_loss
    loss_gap = current_val_loss - current_training_loss
    # previous_training_loss = args.kwargs.get("_previous_training_loss")
    previous_val_loss = args.kwargs.get("_previous_validation_loss")
    previous_time = args.kwargs.get("_epoch_log_time")
    completed_epochs_this_run = args.kwargs.get("_completed_epochs_this_run", 0) + 1
    args.kwargs["_completed_epochs_this_run"] = completed_epochs_this_run
    now = time.perf_counter()
    # if args.kwargs.get('_initial_start_time') is None: args.kwargs['_initial_start_time'] = now
    duration = None if previous_time is None else now - previous_time
    # delta_training_loss = None if previous_training_loss is None else current_training_loss - previous_training_loss
    delta_val_loss = None if previous_val_loss is None else current_val_loss - previous_val_loss
    is_new_best = current_val_loss < args.best_validation_loss
    effective_best = min(current_val_loss, args.best_validation_loss)
    learning_rates = [group["lr"] for group in args.optimizer.param_groups]

    parts = [
        f"Epoch: {epoch + 1}/{args.max_epochs}",
        f"training loss: {args.epochal_training_mean_loss:.6f}",
        f"validation loss: {current_val_loss:.6f}",
        f"best validation loss: {effective_best:.6f}",
        f"loss gap: {loss_gap:.6f}"
    ]
    # if delta_training_loss is not None:
    #     parts.append(f"change: {delta_training_loss:+.6f}")
    if delta_val_loss is not None:
        parts.append(f"change: {delta_val_loss:+.6f}")
    if is_new_best:
        parts.append(f"is new best validation loss: {is_new_best}")
    elif args.kwargs.get('_epochs_without_improvement') is not None:
        parts.append(f'epochs without improvement: {args.kwargs.get('_epochs_without_improvement')}')

    if len(learning_rates) == 1:
        parts.append(f"lr: {learning_rates[0]:.3e}")
    else:
        formatted_lrs = ", ".join(f"{lr:.3e}" for lr in learning_rates)
        parts.append(f"lr: [{formatted_lrs}]")

    if duration is not None:
        parts.append(f"time: {duration:.2f}s")

    if args.kwargs['_initial_start_time'] != now:
        elapsed_time = now - args.kwargs['_initial_start_time']
        parts.append(f'average epoch time: {elapsed_time / completed_epochs_this_run}')
        parts.append(f'total run time: {elapsed_time}')

    args.kwargs["_previous_training_loss"] = current_training_loss
    args.kwargs["_previous_validation_loss"] = current_val_loss
    args.kwargs["_epoch_log_time"] = now

    return parts

def default_epochal_update(args: "Arguments", epoch: int):
    update_schedulers(args, epoch)
    print('\n-----------------------------------------------------------------------------------\n')
    print("\n".join(epoch_logger(args, epoch)))

# </editor-fold>


@dataclass
class Arguments:
    model: torch.nn.Module
    # Either a common loss function from torch.nn.modules.loss or a custom one cf https://saturncloud.io/blog/custom-loss-function-in-pytorch-a-comprehensive-guide/
    loss_function: 'Loss Object'
    optimizer: torch.optim.Optimizer
    training_set: torch.utils.data.dataset.Dataset | torch.utils.data.dataset.Subset
    validation_set: torch.utils.data.dataset.Dataset | torch.utils.data.dataset.Subset
    training_loader: torch.utils.data.dataloader.DataLoader
    validation_loader: torch.utils.data.dataloader.DataLoader
    max_epochs: int
    # Training Saves will be save path + '_training'
    save_path: Path
    # Function that uses this arguments class and the current epoch to do whatever you want:
    #   if you have schedulers you must step them in the epochal update
    #   any other argument parameters that you want to update you may do so if you would like
    #   or if you would like a read-out per epoch you can put that here
    epochal_update: Callable[['Arguments', int], None] = default_epochal_update
    stop_condition: Callable[['Arguments', int], bool] = lambda args, epoch: False
    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = field(default_factory=lambda: [])

    from_existing_model: bool = False

    device: torch.device = device
    best_training_loss: float = float("inf")
    best_validation_loss: float = float("inf")
    start_epoch: int = 0

    _epochal_training_loss: float = field(default=.0, init=False)
    _epochal_validation_loss: float = field(default=.0, init=False)
    _epochal_num_of_items_evaluated_training: int = field(default=0, init=False)
    _epochal_num_of_items_evaluated_validation: int = field(default=0, init=False)

    kwargs: dict = field(default_factory=dict)

    @property
    def checkpoint_path(self) -> Path:
        return self.save_path.parent / f'{self.save_path.stem}_training{self.save_path.suffix}'

    @property
    def epochal_validation_mean_loss(self):
        if self._epochal_num_of_items_evaluated_validation > 0:
            return self._epochal_validation_loss / self._epochal_num_of_items_evaluated_validation
        return float("nan")

    @property
    def epochal_training_mean_loss(self) -> float:
        if self._epochal_num_of_items_evaluated_training > 0:
            return (
                self._epochal_training_loss
                / self._epochal_num_of_items_evaluated_training
            )
        return float("nan")

    def update_epochal_training_loss(self, batch_loss: torch.Tensor,batch_size: int):
        self._epochal_training_loss += batch_loss.detach().item() * batch_size
        self._epochal_num_of_items_evaluated_training += batch_size

    def update_epochal_validation_loss(self, batch_loss: torch.Tensor, batch_size: int):
        self._epochal_num_of_items_evaluated_validation += batch_size
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

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def start_training(self):
        self.model.train()
        self._epochal_training_loss = .0
        self._epochal_num_of_items_evaluated_training = 0

    # noinspection PyAttributeOutsideInit
    def start_evaluating(self):
        self.model.eval()
        self._epochal_validation_loss = .0
        self._epochal_num_of_items_evaluated_validation = 0

    def initialize_logging_metrics(self):
        start_time = time.perf_counter()
        self.kwargs["_initial_start_time"] = start_time
        self.kwargs["_epoch_log_time"] = start_time
        self.kwargs["_completed_epochs_this_run"] = 0


# <editor-fold desc="Loop Sub-functions">
def train(args: Arguments, epoch):
    args.start_training()
    for problems, labels in data_iterator(args.training_loader, args.device):
        args.optimizer.zero_grad(set_to_none=True)
        outputs = args.model(problems)
        loss = args.loss_function(outputs, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at epoch {epoch + 1}: {loss.detach().item()}")
        loss.backward()
        args.optimizer.step()
        # Maybe make the above a closure?
        args.update_epochal_training_loss(loss, infer_batch_size(problems))

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
    args.initialize_logging_metrics()
    for epoch in range(args.start_epoch, args.max_epochs):
        train(args, epoch)
        validate(args, epoch)
        should_stop = args.stop_condition(args, epoch)
        args.epochal_update(args, epoch)
        save_logic(args, epoch)
        if should_stop:
            break


