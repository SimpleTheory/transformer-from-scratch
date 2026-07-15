import dataset_code
import training_framework
import nn_modules
import autograd_functions
import optimizer
from pathlib import Path

def project_root() -> Path:
    current_file_path = Path(__file__).resolve()
    return current_file_path.parent.parent


# # The loss is a scalar averaged over each token and over all the batches
# loss = autograd_functions.softmaxed_cross_entropy.apply(final_logits, expected_outputs)
# return final_logits, loss

data_dir = project_root() / 'data'
documents = [path.read_text(encoding="utf-8") for path in (data_dir / 'dataset').glob("*.txt")]

context_length = 2**9

training_set, validation_set, test_set = dataset_code.TransformerTextDataset.create_splits(
    documents,
    context_length,
    train_ratio=.8,
    validation_ratio=.1,
    seed=30
)
training_loader, validation_loader, test_loader = training_framework.create_loaders(
    training_set, validation_set, test_set
)


model = nn_modules.GPTModel(
    vocab_size=training_set.vocabulary_size,
    embedding_dimension=768,
    max_sequence_length=context_length,
    total_blocks=12,
    num_heads=12,
)

skeleton = training_framework.Arguments(
    model=training_framework.to_device(model),
    loss_function=training_framework.to_device(autograd_functions.softmaxed_cross_entropy.apply),
    optimizer=optimizer.AdamW(model.parameters()),
    training_set=training_set,
    validation_set=validation_set,
    training_loader=training_loader,
    validation_loader=validation_loader,
    max_epochs=10_000,
    save_path=data_dir / 'model/checkpoint.pt',
    # epochal_update= ,
    # stop_condition= ,
    # schedulers= ,
)


if __name__ == '__main__':
    training_framework.loop(skeleton)
