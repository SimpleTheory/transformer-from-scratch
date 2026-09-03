from transformer_from_scratch.base_objects.nn_module_modernizations import *
from transformer_from_scratch.base_objects.autograd_functions import *

class TransformerBlock(torch.nn.Module):
    # TODO class level doc comment
    # Made with projecting back to the embedding dim in mind
    def __init__(
            self,
            embedding_dimension: int,
            num_of_heads: int,
            columns: int = None,
            # skip_attention_layer_norm: bool = True,
            ff_intermediate_columns: int = None,
            ff_total_experts: int = 8,
            ff_experts_to_accept: int = 2,
            dropout_probability: float = 0.1
    ):
        # TODO init level doc comment explaining the params
        # INCLUDE IN DOC COMMENT Columns % Num of heads must == 0
        # INCLUDE IN DOC COMMENT Input should be (batch size, sequence length, embedding dimension)
        super().__init__()

        # <editor-fold desc="Param Initialization">
        # self.skip_attention_layer_norm = skip_attention_layer_norm
        # if not skip_attention_layer_norm:
        self.ff_intermediate_columns = embedding_dimension * 4 if ff_intermediate_columns is None else ff_intermediate_columns
        # </editor-fold>

        self.dropout = InvertedDropout(dropout_probability)
        self.layer_norm_1 = LayerNorm(embedding_dimension)
        self.attention_block = MultiHeadAttention(embedding_dimension, num_of_heads, columns, True)
        self.layer_norm_2 = LayerNorm(embedding_dimension)

        # Decide on which of these to use at first
        self.feed_forward = DoubleLinearApplied(embedding_dimension, self.ff_intermediate_columns, embedding_dimension)
        # self.feed_forward = MoEDoubleLinearApplied(
        #     ff_total_experts,
        #     ff_experts_to_accept,
        #     embedding_dimension,
        #     self.ff_intermediate_columns,
        # )

    def forward(self, input: torch.Tensor):
        # Input must be size (Batch Size, Sequence Length, Embedding Dimensions)
        # Create a copy to preserve original input for debugging
        input_copy = input

        # Residual connections: instead of taking the result as the next step in the pipeline we add it back to the input
        # That way we have a build up of everything that came before and at least some context of what things were...allegedly
        # Theoretically concatenation (adding the output as dimensions) would work, but it would blow up the size of the model
        # While this has trade-offs it makes it easier to not overfit and keeps the model size low. This is done to every
        # operation in the block, basically "preserving" everything that came before it. Because of that though, the sizes
        # of the outputs and the input need to be the same.

        # <editor-fold desc="Skip Attention Layer Logic (Commented Out)">
        # if self.skip_attention_layer_norm:
        #     # Don't apply layer norm on the first pass from the embeddings
        #     input_copy = input_copy + self.attention_block(input_copy)

        # else:
        # </editor-fold>

        # Residual connection, apply the multihead attention to the layer normalized input to stabilize it.
        # This is needed in case of noise from the previous operations due to the residual connection concept.
        input_copy = input_copy + self.dropout(self.attention_block(self.layer_norm_1(input_copy)))

        # Mix and expand the output from the attention with a linear layer then reproject it down to size `embedding_dim`
        # with a second linear layer. Since the linear layers are connected to each other they are split with an activation
        # function to introduce non-linearity. Otherwise, they'd be equivalent to one giga linear layer.
        input_copy = input_copy + self.dropout(self.feed_forward(self.layer_norm_2(input_copy)))

        return input_copy

class GPTModel(torch.nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_dimension: int,
            max_sequence_length: int,
            total_blocks: int,
            num_heads: int,
            ff_intermediate_columns_multiplier: int = 4,
            dropout_probability: float = 0.1,
            tie_weights: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length
        self.total_blocks = total_blocks
        self.ff_intermediate_columns_multiplier = ff_intermediate_columns_multiplier

        self.token_embeddings = EmbeddingLayer(vocab_size, embedding_dimension)
        self.positional_embeddings = EmbeddingLayer(max_sequence_length, embedding_dimension)
        self.dropout = InvertedDropout(dropout_probability)

        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(
                embedding_dimension=embedding_dimension,
                num_of_heads=num_heads,
                ff_intermediate_columns=embedding_dimension * ff_intermediate_columns_multiplier,
                dropout_probability=dropout_probability
                # skip_attention_layer_norm=False,
            )
            for _ in range(total_blocks)
        ])
        self.final_layer_norm = LayerNorm(embedding_dimension)
        # Weight tying is using the same table for token embedding and for the linear_to_vocab layer.
        # The idea being that you save a lot of parameters since the (vocab_size, embedding_dim) table is ginormous.
        # It usually performs worse than having an independent specialized table for the task though.
        self.tie_weights = tie_weights
        if tie_weights:
            # Traditionally weight-tyings did not have a bias, but honestly the param cost is small, and it learns independently
            # of the tied matrix. So there is really no reason not to include a bias.

            # Creating the layer like this is fine because the embedding matrix is already (vocab_size, embedding_dim)
            # which is already (out_features, in_features).
            self.linear_to_vocab = LinearLayer(self.token_embeddings.embedding_matrix, torch.nn.Parameter(torch.zeros(vocab_size)))
        else:
            self.linear_to_vocab = LinearLayer.from_feature_counts(embedding_dimension, vocab_size)

    def forward(self, inputs):
        """
        token_ids
          ↓
        token embedding + positional embedding
          ↓
        TransformerBlocks
          ↓
        final LayerNorm
          ↓
        linear projection to vocab size
          ↓
        logits: (Batch Size, Sequence Length, Vocab Size)

        the model returns:
            logits.shape == (batch_size, sequence_length, vocab_size)
        Each row along the last dimension is saying:
            "For this token position, how likely is each possible next token?"
        For example logits[0, 5] means:
            The model's raw scores for the next token after position 5 in batch item 0.

        :param inputs: (Batch Size, Sequence Length)
            Contains integer token ids
        :return: Tensor(Batch Size, Sequence Length, Vocab Size) | ..., loss: float
        """
        batch_size, sequence_length = inputs.shape
        # Checking input to see if its valid
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"sequence_length={sequence_length} exceeds max_sequence_length={self.max_sequence_length}"
            )
        # Create a sequence with of the data length and then embed each thing in the sequence.
        # So essentially you have an embedding for position 0, 1, 2, 3, etc...
        # The idea being you can use this to represent position here its just done by adding it to the token embeddings,
        # but really you can do anything. For example concatenate it and then project it to the needed size or anything you can imagine.
        # Pytorch equivalent of [_ for _ in range(sequence_length)]
        position_ids = torch.arange(sequence_length, device=inputs.device)

        token_vectors = self.token_embeddings(inputs)
        position_vectors = self.positional_embeddings(position_ids)
        # (Batch Size, Sequence Length, Embedding Dimensions)
        data_to_work_on = self.dropout(token_vectors + position_vectors)
        for block in self.transformer_blocks:
            data_to_work_on = block(data_to_work_on)

        # Normalize the data one more time and project to the vocab size to get the logits, where each
        # logit represents: at the current batch for the current token position how likely is any of the next tokens
        # in the vocab size.
        # (Batch Size, Sequence Length, Vocab Size)
        final_logits = self.linear_to_vocab(
            self.final_layer_norm(data_to_work_on)
        )
        return final_logits

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int):
        """
        For a single prompt batch size should be 1.

        token_ids:
            Shape: (batch_size, current_sequence_length)
        Returns:
            Shape: (batch_size, current_sequence_length + max_new_tokens)
        """

        for _ in range(max_new_tokens):
            # How indexing works for tensors is each argument is another dimension
            # So for instance in a 2D tensor, tensor[3, 5] means take the element at the third row and the 5th column.
            # Likewise, you can also apply slices to these elements so in the same vein tensor[3, :] means take all the columns of the 3rd row
            # In this case tokens is (batch_size, current_sequence_length) so this means
                # Over all the batches (which are the rows thus the first :) slice the columns up to -self.max_sequence_length
                # Effectively cropping the batches to the max sequence length in case they are longer than that
            # This is important since by generating a response we are effectively "lengthening" the sequence length as well
            token_ids_cropped = token_ids[:, -self.max_sequence_length:]

            # (Batch Size, Sequence Length, Vocab Size)
            # For example logits[0, 5] means: The model's raw scores for the next token after position 5 in batch item 0.
            # self(...) works because you are essentially just calling the model this is a part of.
            next_token_scores = self(token_ids_cropped)

            # Again same tensor indexing paradigm
            # 1. Over all the batches, 2. Take the last part of the sequence, 3. and grab the scores for the next token across the whole vocab size
            # (Batch Size, Vocab Size)
            last_token_scores = next_token_scores[:, -1, :]

            # We do softmax to do log based probabilities because the model is based off of that
            next_token_probabilities = autograd_functions.softmax_with_kwarg(last_token_scores, dim=-1)

            # How torch.multinomial works is for 1D it samples (weighted random) that list, for 2D it samples every row
            # Additionally, the sample returns as the index.
            # Here for (Batch Size, Vocab Size) you get one sample for every batch.
            # And then it returns the index of the sampled item, which just so happens to equal the token id.
            # This in turn effectively gives the next token over any given batch's probability range
            # For example if a batch has probability [.1, .2, .7] one of those items relative the weights will be chosen:
                # With the above batch -> 0 10%, 1 20%, 2 70%
            next_token = torch.multinomial(next_token_probabilities, num_samples=1)

            # Here we add the new token to the original token_ids over every batch, so that the next iteration of the loop
            # can use this new modified token ids to make its guess.
            # Cat here is specifically (batch size, 1) catting to (batch size, sequence length) so its adding 1 (aka the new token)
            # to sequence length.
            token_ids = torch.cat([token_ids, next_token], dim=1)

        # Now we return the token ids with all the new tokens in them. For actually displaying the results though like on
        # a chatbot you could hide the original prompt or the whole context with some regular code.
        return token_ids