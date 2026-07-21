from datasets import load_dataset
import dataset_code
import training_framework
import nn_modules
import autograd_functions
import optimizer
import torch
from pathlib import Path

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

context_length = 2**8
max_epochs = 10_000
data_dir = project_root() / 'data'
starting_learning_rate = 3e-3
minimum_learning_rate = 3e-5

if __name__ == '__main__':
    ds = load_dataset("roneneldan/TinyStories")
    training_set = dataset_code.TransformerTextDataset(
        ds,
        split="train",
        text_column="text",  # Automatically inferred here
        context_length=context_length,
        stride=context_length // 2,
        max_documents=10_000,  # Start small for a proof of concept
        selection_seed=42,
    )

    validation_set = dataset_code.TransformerTextDataset(
        ds,
        split="validation",
        context_length=context_length,
        stride=context_length,
        max_documents=1_000,
        selection_seed=42,
    )

    training_loader, validation_loader = training_framework.create_loaders(
        training_set, validation_set,  # test_set
    )

    model = nn_modules.GPTModel(
        vocab_size=training_set.vocabulary_size,
        embedding_dimension=256,
        max_sequence_length=context_length,
        total_blocks=4,
        num_heads=4,
    )
    model = training_framework.to_device(model)
    optim = optimizer.AdamW(make_adamw_parameter_groups(model, weight_decay=1e-3), learning_rate=starting_learning_rate)

    skeleton = training_framework.Arguments(
        model=model,
        loss_function=autograd_functions.softmaxed_cross_entropy.apply,
        optimizer=optim,
        training_set=training_set,
        validation_set=validation_set,
        training_loader=training_loader,
        validation_loader=validation_loader,
        max_epochs=max_epochs,
        save_path=data_dir / 'model/checkpoint.pt',
        # load_path=data_dir / 'model_alt_folder/checkpoint_last.pt',
        # epochal_update= ,
        stop_condition=training_framework.early_stop(patience=50),
        schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs, eta_min=minimum_learning_rate, last_epoch=-1)],
        max_batches=2
    )

    training_framework.loop(skeleton)
