import math
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Callable, Iterator, Any
import torch.utils.data
import torch
import time
import utility


# <editor-fold desc="Slot-in Functions">
early_stop_key = '_epochs_without_improvement'
def early_stop(
        patience: int = 5,
        minimum_loss_improvement: float = 0.0,
        initial_epoch_buffer: int = 100,
        print_result: bool = True,
):
    def stop_condition(args: "Arguments", epoch: int) -> bool:
        if not math.isfinite(args.epochal_validation_mean_loss):
            raise RuntimeError(
                f"Validation loss became nonfinite at epoch {epoch + 1}: "
                f"{args.epochal_validation_mean_loss}"
            )
        if args.epochal_validation_mean_loss < args.best_validation_loss - minimum_loss_improvement:
            # Num Epochs Without Improvement
            args.kwargs[early_stop_key] = 0
        else:
            args.kwargs[early_stop_key] = args.kwargs.get(early_stop_key, 0) + 1

        epochs_without_improvement = args.kwargs[early_stop_key]
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

def update_batch_scheduler(args: "Arguments"):
    for scheduler in args.batch_schedulers:
        scheduler.step()

def get_epoch_logger_dict(args: "Arguments", epoch: int) -> dict[str, Any]:
    """
    Epoch: The count of the current epoch,
    Items in Epoch Training: # items evaluated in training
    Items in Epoch Validation: # items evaluated in validation
    Training Loss: training loss in current epoch,
    Validation Loss: validation loss in current epoch,
    Loss Gap: epochal training loss - epochal validation loss
    Best Training Loss: ...
    Best Validation Loss: ...
    Change in Validation Loss: validation loss in previous epoch minus the current validation loss
    Epochs Without Improvement: Count of epochs where the validation loss has not gotten lower
    Learning Rates: A list of learning rates for each parameter group,
    Epoch Time: The amount of time taken to run one epoch,
    Epoch Time Average: The average amount of time taken to run one epoch,
    Total Run Time: Total time take since the start of the current training session,
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
    effective_best_training = min(current_training_loss, args.best_training_loss)
    effective_best = min(current_val_loss, args.best_validation_loss)
    learning_rates = [
        lr for group in args.optimizer.param_groups
        if (lr := group.get("lr", group.get("learning_rate"))) is not None
    ]

    logging_dict: dict[str, Any] = {
        "Epoch": epoch + 1,
        "Items in Epoch Training": args.epochal_num_of_items_evaluated_training,
        "Items in Epoch Validation": args.epochal_num_of_items_evaluated_validation,
        "Training Loss": args.epochal_training_mean_loss,
        "Validation Loss": current_val_loss,
        "Current Loss Gap": loss_gap,
        "Best Training Loss": effective_best_training,
        "Best Validation Loss": effective_best,
    }
    # if delta_training_loss is not None:
    #     parts.append(f"change: {delta_training_loss:+.6f}")
    if delta_val_loss is not None:
        logging_dict['Change in Validation Loss'] = delta_val_loss
    # if is_new_best:
    #     logging_dict['New Best Validation Loss'] = is_new_best
    # elif args.kwargs.get(early_stop_key) is not None:
    logging_dict['Epochs Without Improvement'] = args.kwargs.get(early_stop_key, 0)

    logging_dict['Learning Rates'] = learning_rates

    if duration is not None:
        logging_dict['Epoch Time'] = utility.format_seconds_into_time(duration)

    if args.kwargs['_initial_start_time'] != now:
        elapsed_time = now - args.kwargs['_initial_start_time']
        logging_dict['Epoch Time Average'] = utility.format_seconds_into_time(elapsed_time / completed_epochs_this_run)
        logging_dict['Total Run Time'] = utility.format_seconds_into_time(elapsed_time)

    args.kwargs["_previous_training_loss"] = current_training_loss
    args.kwargs["_previous_validation_loss"] = current_val_loss
    args.kwargs["_epoch_log_time"] = now

    return logging_dict

def default_epochal_update(args: "Arguments", epoch: int):
    update_schedulers(args, epoch)
    print('\n-----------------------------------------------------------------------------------\n')
    print("\n".join([f'{k}: {v}' for k, v in get_epoch_logger_dict(args, epoch).items()]))

def default_batch_update(args: "Arguments", epoch: int, batch_info: "BatchInformation"):
    update_batch_scheduler(args)
    args.update_epochal_training_loss(batch_info)

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
    save_path: Path
    load_path: Path = None
    # Function that uses this arguments class and the current epoch to do whatever you want:
    #   if you have schedulers you must step them in the epochal update
    #   any other argument parameters that you want to update you may do so if you would like
    #   or if you would like a read-out per epoch you can put that here
    epochal_update: Callable[['Arguments', int], None] = default_epochal_update
    batchal_update: Callable[['Arguments', int, 'BatchInformation'], None] = default_batch_update
    stop_condition: Callable[['Arguments', int], bool] = lambda args, epoch: False
    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = field(default_factory=lambda: [])
    batch_schedulers: list[torch.optim.lr_scheduler.LRScheduler] = field(default_factory=lambda: [])

    device: torch.device = utility.device
    best_training_loss: float = float("inf")
    best_validation_loss: float = float("inf")
    start_epoch: int = 0

    _epochal_training_loss: float = field(default=.0, init=False)
    _epochal_validation_loss: float = field(default=.0, init=False)
    _epochal_num_of_items_evaluated_training: int = field(default=0, init=False)
    _epochal_num_of_items_evaluated_validation: int = field(default=0, init=False)

    max_batches: int | None = None

    kwargs: dict = field(default_factory=dict)

    @property
    def last_checkpoint_path(self) -> Path:
        return self.save_path.parent / f"{self.save_path.stem}_last{self.save_path.suffix}"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.save_path.parent / f"{self.save_path.stem}_best{self.save_path.suffix}"

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

    @property
    def epochal_num_of_items_evaluated_training(self):
        return self._epochal_num_of_items_evaluated_training

    @property
    def epochal_num_of_items_evaluated_validation(self):
        return self._epochal_num_of_items_evaluated_validation

    def update_epochal_training_loss(self, batch_info: 'BatchInformation'):
        self._epochal_training_loss += batch_info.loss.detach().item() * batch_info.batch_size
        self._epochal_num_of_items_evaluated_training += batch_info.batch_size

    def update_epochal_validation_loss(self, batch_loss: torch.Tensor, batch_size: int):
        self._epochal_num_of_items_evaluated_validation += batch_size
        # Only works for mean loss aggregate
        self._epochal_validation_loss += batch_loss.item() * batch_size

    def __post_init__(self):

        self.device = torch.device(self.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        model_devices = {parameter.device for parameter in self.model.parameters()}
        model_devices.update(buffer.device for buffer in self.model.buffers())
        if model_devices and model_devices != {self.device}:
            raise ValueError(
                f"Model parameters/buffers are on {model_devices}, "
                f"but training device is {self.device}. "
                "Move the model before constructing the optimizer."
            )
        if self.load_path:
            load_training_checkpoint(self, self.load_path)
        utility.validate_optimizer_device(self.model, self.optimizer)
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

    def cap_iterator(self, iterator) -> Iterator:
        if self.max_batches:
            return islice(iterator, self.max_batches)
        return iterator


class BatchInformation:
    def __init__(self, problems, labels, outputs, loss):
        self.problems = problems
        self.labels = labels
        self.outputs = outputs
        self.loss = loss
        self.batch_size = utility.infer_batch_size(problems)


# <editor-fold desc="Loop Sub-functions">
def train(args: Arguments, epoch):
    args.start_training()
    for problems, labels in args.cap_iterator(utility.data_iterator(args.training_loader, args.device)):
        args.optimizer.zero_grad(set_to_none=True)

        outputs = args.model(problems)
        loss = args.loss_function(outputs, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at epoch {epoch + 1}: {loss.detach().item()}")
        loss.backward()
        args.optimizer.step()
        args.batchal_update(args, epoch, BatchInformation(problems, labels, outputs, loss))


def validate(args: Arguments, epoch):
    args.start_evaluating()
    with torch.inference_mode():
        for problems, labels in args.cap_iterator(utility.data_iterator(args.validation_loader, args.device)):
            outputs = args.model(problems)
            loss = args.loss_function(outputs, labels)
            args.update_epochal_validation_loss(loss, utility.infer_batch_size(problems))

def save_model_for_inference(model: torch.nn.Module, path):
    torch.save(model.state_dict(), path)


arguments_to_save_for_training = [early_stop_key,]
def save_model_for_training(args: Arguments, epoch, path):
    kwargs_to_save = {key: args.kwargs[key] for key in arguments_to_save_for_training}
    temporary_path = path.with_suffix(path.suffix + ".temporary")

    torch.save({
        "epoch": epoch,
        "model_state_dict": args.model.state_dict(),
        "optimizer_state_dict": args.optimizer.state_dict(),
        "epochal_scheduler_state_dicts": [scheduler.state_dict() for scheduler in args.schedulers],
        "batch_scheduler_state_dicts": [scheduler.state_dict() for scheduler in args.batch_schedulers],
        "best_training_loss": args.best_training_loss,
        "best_validation_loss": args.best_validation_loss,
        "kwargs": kwargs_to_save
    },
        temporary_path
    )
    temporary_path.replace(path)

def save_logic(args: Arguments, epoch):
    if (args.epochal_validation_mean_loss < args.best_validation_loss) or (args.best_validation_loss == float('inf')):
        args.best_validation_loss = args.epochal_validation_mean_loss
        # save_model_for_inference(args.model, args.save_path)
        save_model_for_training(args, epoch, args.best_checkpoint_path)
        save_model_for_inference(args.model, args.save_path)
    save_model_for_training(args, epoch, args.last_checkpoint_path)

def load_training_checkpoint(args: Arguments, path):
    """
    Modifies Arguments in place to resume training.

    "epoch": epoch,
    "model_state_dict": args.model.state_dict(),
    "optimizer_state_dict": args.optimizer.state_dict(),
    "scheduler_state_dicts": [scheduler.state_dict() for scheduler in args.schedulers],
    "best_training_loss": args.best_training_loss,
    "best_validation_loss": args.best_validation_loss,
    "kwargs": kwargs_to_save
    :return: None
    """
    if not path.exists():
        raise FileNotFoundError(f"Training checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=args.device, weights_only=True)
    # The saved epoch has already completed, so resume from the next one.
    args.start_epoch = int(checkpoint["epoch"] + 1)
    args.model.load_state_dict(checkpoint["model_state_dict"])
    args.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # <editor-fold desc="Load Schedulers">
    saved_scheduler_states = checkpoint.get("epochal_scheduler_state_dicts", [])
    saved_batch_scheduler_states = checkpoint.get("batch_scheduler_state_dicts", [])

    if len(saved_scheduler_states) != len(args.schedulers):
        raise ValueError(f"The checkpoint contains {len(saved_scheduler_states)} scheduler states, but Arguments contains {len(args.schedulers)} schedulers.")
    if len(saved_batch_scheduler_states) != len(args.batch_schedulers):
        raise ValueError(f"The checkpoint contains {len(saved_batch_scheduler_states)} scheduler states, but Arguments contains {len(args.batch_schedulers)} schedulers.")

    for scheduler, scheduler_state in zip(args.schedulers, saved_scheduler_states):
        scheduler.load_state_dict(scheduler_state)

    for scheduler, scheduler_state in zip(args.batch_schedulers, saved_batch_scheduler_states):
        scheduler.load_state_dict(scheduler_state)
    # </editor-fold>

    args.best_training_loss = checkpoint.get("best_training_loss", float("inf"))
    args.best_validation_loss = checkpoint.get("best_validation_loss", float("inf"))
    args.kwargs.update(checkpoint.get('kwargs', dict()))

    print(f"Resumed training from epoch {args.start_epoch}. Best validation loss: {args.best_validation_loss:.6f}")
# </editor-fold>


def loop(args: Arguments):
    args.initialize_logging_metrics()
    for epoch in range(args.start_epoch, args.max_epochs):
        train(args, epoch)
        validate(args, epoch)
        # noinspection PyArgumentList
        should_stop = args.stop_condition(args, epoch)
        args.epochal_update(args, epoch)
        save_logic(args, epoch)
        if should_stop:
            break


