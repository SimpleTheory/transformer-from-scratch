import torch
import dataset_code
import training_framework
import nn_modules
import autograd_functions
import optimizer

# # The loss is a scalar averaged over each token and over all the batches
# loss = autograd_functions.softmaxed_cross_entropy.apply(final_logits, expected_outputs)
# return final_logits, loss
dataset = dataset_code.TransformerTextDataset.from_file()
model = nn_modules.GPTModel(
    vocab_size=dataset.vocabulary_size,
    embedding_dimension=768,
    max_sequence_length=2**8,
    total_blocks=,
    num_heads=12,
)

skeleton = training_framework.Arguments(
    model=training_framework.to_device(),
    loss_function=training_framework.to_device(autograd_functions.softmaxed_cross_entropy.apply),
    optimizer=optimizer.AdamW(

    )
)
