from pathlib import Path
from datasets import load_dataset
import dataset_code
import training_framework
import nn_modules
import autograd_functions
import optimizer
import torch
import utility

def project_root() -> Path:
    current_file_path = Path(__file__).resolve()
    return current_file_path.parent.parent

def make_adamw_parameter_groups(model: torch.nn.Module, weight_decay: float = 1e-2) -> list[dict]:
    """
    The idea is biases don't need weight decay because they are relatively fixed and won't accumulate gradients.
    Layer norm parameters also don't need weight decay because they are supposed to learn magnitude.
    Both sets of parameters are broadcast so their size is 1 because of that you can use the following filter to check
    for which parameters require weight decay. `if parameter.ndim >= 2`
    :param model:
    :param weight_decay:
    :return:
    """
    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    return [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]


# # The loss is a scalar averaged over each token and over all the batches
# loss = autograd_functions.softmaxed_cross_entropy.apply(final_logits, expected_outputs)
# return final_logits, loss
data_dir = project_root() / 'data'
# selection_seed: int = 42

class Config(utility.CommandLineArguments):
    save_path: Path

    load_path: Path | None = None
    seed: int = 42
    context_length: int = 2 ** 8
    max_epochs: int = 10_000
    starting_learning_rate: float = 3e-3
    minimum_learning_rate: float = 3e-5
    dataset_cache_dir: Path | None = None
    training_stride: int = utility.derived(lambda self: self.context_length // 2)
    validation_stride: int = utility.derived(lambda self: self.context_length)
    max_documents_training: int = 10_000
    max_documents_validation: int = 1_000
    batch_size = 2**5
    loader_num_workers: int = 2
    embedding_dim: int = 256
    num_heads: int = utility.derived(lambda self: self.embedding_dim // 2**6)  # --> 8-6 -> 2**2
    num_blocks: int = 4
    # Should Device be here?
    patience: int = 50
    weight_decay: float = .001
    max_batches: int | None = None


if __name__ == '__main__':
    config = Config.from_command_line()
    dataset = load_dataset("roneneldan/TinyStories", cache_dir=config.dataset_cache_dir)
    training_set = dataset_code.TransformerTextDataset(
        dataset,
        split="train",
        text_column="text",  # Automatically inferred here
        context_length=config.context_length,
        stride=config.training_stride,
        max_documents=config.max_documents_training,  # Start small for a proof of concept
        selection_seed=config.seed,
    )

    validation_set = dataset_code.TransformerTextDataset(
        dataset,
        split="validation",
        context_length=config.context_length,
        stride=config.validation_stride,
        max_documents=config.max_documents_validation,
        selection_seed=config.seed,
    )

    training_loader, validation_loader = utility.create_loaders(
        training_set,
        validation_set,
        number_of_workers=config.loader_num_workers,
        batch_size=config.batch_size

    )  # test_set)

    model = nn_modules.GPTModel(
        vocab_size=training_set.vocabulary_size,
        embedding_dimension=config.embedding_dim,
        max_sequence_length=config.context_length,
        total_blocks=config.num_blocks,
        num_heads=config.num_heads,
    )
    model = utility.to_device(model)
    optim = optimizer.AdamW(make_adamw_parameter_groups(model, weight_decay=config.weight_decay), learning_rate=config.starting_learning_rate)

    skeleton = training_framework.Arguments(
        model=model,
        loss_function=autograd_functions.softmaxed_cross_entropy.apply,
        optimizer=optim,
        training_set=training_set,
        validation_set=validation_set,
        training_loader=training_loader,
        validation_loader=validation_loader,
        max_epochs=config.max_epochs,
        save_path=config.save_path,
        load_path=config.load_path,
        # epochal_update= ,
        stop_condition=training_framework.early_stop(patience=config.patience),
        schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(optim, config.max_epochs, eta_min=config.minimum_learning_rate, last_epoch=-1)],
        max_batches=config.max_batches
    )

    training_framework.loop(skeleton)
