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


# # The loss is a scalar averaged over each token and over all the batches
# loss = autograd_functions.softmaxed_cross_entropy.apply(final_logits, expected_outputs)
# return final_logits, loss

context_length = 2**8
max_epochs = 10_000
data_dir = project_root() / 'data'

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
    optim = optimizer.AdamW(model.parameters())

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
        # epochal_update= ,
        stop_condition=training_framework.early_stop(patience=50),
        schedulers=[torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs, eta_min=1e-3, last_epoch=-1)],
    )

    training_framework.loop(skeleton)
