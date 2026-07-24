import math
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Callable, Iterator, Any
import torch.utils.data
import torch
import time
import utility
import csv

# <editor-fold desc="Slot-in Functions">
early_stop_key = '_epochs_without_improvement'
last_validation_loss_key = '_previous_validation_loss'

grad_scale_stats_key = '_gradient_scale_stats'


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


def save_log(epoch_logger_dict: dict[str, Any], log_file: Path):
    io_of_logfile: list[list[Any]] = []
    temporary_path = log_file.with_suffix('.temporarycsv')
    if log_file.exists():
        with open(log_file, 'r', newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                io_of_logfile.append(row)
    else:
        log_file.parent.mkdir(exist_ok=True)
        io_of_logfile.append(list(epoch_logger_dict.keys()))
    io_of_logfile.append([str(value) for value in epoch_logger_dict.values()])
    io_of_logfile = [list_ for list_ in io_of_logfile if len(list_) > 0]
    # print(io_of_logfile)
    with open(temporary_path, 'w', newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(io_of_logfile)

    temporary_path.replace(log_file)


def update_grad_scale_list(args: "Arguments", current_grad_scale: torch.Tensor | None):
    """
    Increments the gradient stats dict, and creates it if it doesn't exist.
    :param args: Arguments
    :param current_grad_scale: Tensor Scalar float32
    :return: None
    """
    if args.maximum_gradient_scale is None or current_grad_scale is None:
        return
    # Deprecated comments but keeping because the information is useful
    # Convert it to a Non-Tensor Object first then add to list
    # .detach() ensures the logging value is disconnected from autograd,
    # while .item() converts the scalar CUDA tensor into a normal Python number.

    # One caveat: .item() synchronizes the CPU and GPU each batch. For a small proof-of-concept training run,
    # that is unlikely to matter much. For maximum performance, accumulate the statistics as tensors and call
    # .item() only once per epoch, or maintain running sum/max/count values rather than storing every batch value.
    stats_dtype = torch.float32 if current_grad_scale.dtype in (torch.float16, torch.bfloat16) else current_grad_scale.dtype
    current_grad_scale = current_grad_scale.detach().to(stats_dtype)

    stats = args.kwargs.get(grad_scale_stats_key)

    if stats is None:
        args.kwargs[grad_scale_stats_key] = {
            "sum": current_grad_scale.clone(),
            "max": current_grad_scale.clone(),
            "clipped": (current_grad_scale > args.maximum_gradient_scale).to(torch.int64),
            "count": 1,
        }
    else:
        stats["sum"] += current_grad_scale
        stats["max"] = torch.maximum(stats["max"], current_grad_scale)
        stats["clipped"] += current_grad_scale > args.maximum_gradient_scale
        stats["count"] += 1


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
    # previous_training_loss = args.kwargs.get("_previous_training_loss")
    previous_time = args.kwargs.get("_epoch_log_time")
    args.kwargs["_completed_epochs_this_run"] = args.kwargs.get("_completed_epochs_this_run", 0) + 1
    # delta_training_loss = None if previous_training_loss is None else current_training_loss - previous_training_loss
    delta_val_loss = current_val_loss - args.kwargs.get(last_validation_loss_key) if args.kwargs.get(
        last_validation_loss_key) is not None else None
    learning_rates = [
        lr for group in args.optimizer.param_groups
        if (lr := group.get("lr", group.get("learning_rate"))) is not None
    ]
    gradient_stats = args.kwargs.get(grad_scale_stats_key)
    now = time.perf_counter()
    elapsed_time = now - args.kwargs['_initial_start_time']

    logging_dict: dict[str, Any] = {
        "Epoch": epoch + 1,
        "Items in Epoch Training": args.epochal_num_of_items_evaluated_training,
        "Items in Epoch Validation": args.epochal_num_of_items_evaluated_validation,
        "Training Loss": args.epochal_training_mean_loss,
        "Validation Loss": current_val_loss,
        "Current Loss Gap": current_val_loss - current_training_loss,
        "Best Training Loss": min(current_training_loss, args.best_training_loss),
        "Best Validation Loss": min(current_val_loss, args.best_validation_loss),
        'Change in Validation Loss': delta_val_loss,
        'Epochs Without Improvement': args.kwargs.get(early_stop_key, 0),
        'Learning Rates': learning_rates,
        'Epoch Time': utility.format_seconds_into_time(now - previous_time) if previous_time is not None else None,
        'Epoch Time Average': utility.format_seconds_into_time(elapsed_time / args.kwargs["_completed_epochs_this_run"]),
        'Total Run Time': utility.format_seconds_into_time(elapsed_time),
    }

    if all([args.maximum_gradient_scale is not None, gradient_stats, gradient_stats["count"]]):
            logging_dict.update({
                "Gradient Scale Mean": (gradient_stats["sum"] / gradient_stats["count"]).item(),
                "Gradient Scale Max Calculated": gradient_stats["max"].item(),
                "Gradient Percent Clipped": (gradient_stats["clipped"] / gradient_stats["count"]).item(),
            })

    args.kwargs["_previous_training_loss"] = current_training_loss
    args.kwargs["_previous_validation_loss"] = current_val_loss
    args.kwargs["_epoch_log_time"] = now

    return logging_dict


def logging_main(args: "Arguments", epoch: int):
    epoch_logger_dict = get_epoch_logger_dict(args, epoch)
    print('\n-----------------------------------------------------------------------------------\n')
    print("\n".join([f'{k}: {v}' for k, v in epoch_logger_dict.items()]))
    if args.log_path is not None:
        save_log(epoch_logger_dict, args.log_path)


def default_epochal_update(args: "Arguments", epoch: int):
    update_schedulers(args, epoch)
    logging_main(args, epoch)


def default_batch_update(args: "Arguments", epoch: int, batch_info: "BatchInformation"):
    update_batch_scheduler(args)
    update_grad_scale_list(args, batch_info.gradient_scale)
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
    log_path: Path | None = None
    # Function that uses this arguments class and the current epoch to do whatever you want:
    #   if you have schedulers you must step them in the epochal update
    #   any other argument parameters that you want to update you may do so if you would like
    #   or if you would like a read-out per epoch you can put that here
    epochal_update: Callable[['Arguments', int], None] = default_epochal_update
    batchal_update: Callable[['Arguments', int, 'BatchInformation'], None] = default_batch_update
    stop_condition: Callable[['Arguments', int], bool] = lambda args, epoch: False
    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = field(default_factory=lambda: [])
    batch_schedulers: list[torch.optim.lr_scheduler.LRScheduler] = field(default_factory=lambda: [])
    maximum_gradient_scale: float | None = 1.0

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
        self.kwargs[grad_scale_stats_key] = None

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

    def scale_gradients_down(self) -> torch.Tensor | None:
        """
        This operation is in place it scales down each of the gradients by max scale / original scale. Only if the the
        original scale is larger though. cf `utility.scale_down_large_gradients`
        :return: (None | Tensor Scalar torch.float32) Original gradient scale
        """
        if self.maximum_gradient_scale:
            return utility.scale_down_large_gradients(self.model, self.maximum_gradient_scale)
        return None

class BatchInformation:
    def __init__(self, problems, labels, outputs, loss, gradient_norm=None):
        self.problems = problems
        self.labels = labels
        self.outputs = outputs
        self.loss = loss
        self.batch_size = utility.infer_batch_size(problems)
        self.gradient_scale = gradient_norm


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
        gradient_norm = args.scale_gradients_down()
        args.optimizer.step()

        args.batchal_update(args, epoch, BatchInformation(problems, labels, outputs, loss, gradient_norm))


def validate(args: Arguments, epoch):
    args.start_evaluating()
    with torch.inference_mode():
        for problems, labels in args.cap_iterator(utility.data_iterator(args.validation_loader, args.device)):
            outputs = args.model(problems)
            loss = args.loss_function(outputs, labels)
            args.update_epochal_validation_loss(loss, utility.infer_batch_size(problems))


def save_model_for_inference(model: torch.nn.Module, path):
    torch.save(model.state_dict(), path)


arguments_to_save_for_training = [early_stop_key, last_validation_loss_key]


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
        raise ValueError(
            f"The checkpoint contains {len(saved_scheduler_states)} scheduler states, but Arguments contains {len(args.schedulers)} schedulers.")
    if len(saved_batch_scheduler_states) != len(args.batch_schedulers):
        raise ValueError(
            f"The checkpoint contains {len(saved_batch_scheduler_states)} scheduler states, but Arguments contains {len(args.batch_schedulers)} schedulers.")

    for scheduler, scheduler_state in zip(args.schedulers, saved_scheduler_states):
        scheduler.load_state_dict(scheduler_state)

    for scheduler, scheduler_state in zip(args.batch_schedulers, saved_batch_scheduler_states):
        scheduler.load_state_dict(scheduler_state)
    # </editor-fold>

    args.best_training_loss = checkpoint.get("best_training_loss", float("inf"))
    args.best_validation_loss = checkpoint.get("best_validation_loss", float("inf"))
    args.kwargs.update(checkpoint.get('kwargs', dict()))

    print(f"Resumed training from epoch {args.start_epoch} - Best validation loss: {args.best_validation_loss:.6f} - "
          f"Last validation loss: {args.kwargs[last_validation_loss_key]}")


# </editor-fold>


def loop(args: Arguments):
    print('Initializing model loop')
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
