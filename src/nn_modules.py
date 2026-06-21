import torch
import autograd_functions
import math

"""
... Then build Modules:

--Linear
--LayerNorm
--FeedForward
--Embedding
--SingleHead Attention
MultiHeadAttention
TransformerBlock
GPT


# Sample Linear layer to see how to apply nn.Module
class MyLinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.biases = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, inputs):
        return wx_plus_b.apply(inputs, self.weights, self.biases)

Notes:
    Don't use @dataclass for nn.Modules (while learning pytorch) because order matters when super init is called and the attributes are defined


"""
class LinearLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Layer wx+b where
            x (..., in_features)
            w (out_features, in_features)
            b (out_features,)

        Where
            `in features` should be the trailing dimension of the input of this layer

            `out features` should be the amount of columns you want to add or remove from the matrix in the pipeline (after the transformation)

        """
        super().__init__()
        # Todo layer initialization because randn's std is too big leading to big gradients and inefficient learning
        # Generic layer init is torch.randn(...) / sqrt(in_features)
        # Since relu makes it smaller by about half with relu it tends to be ... * (2/sqrt(in_features))
        self.weights = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.biases = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, inputs: torch.Tensor):
        return autograd_functions.wx_plus_b.apply(inputs, self.weights, self.biases)

class DoubleLinearApplied(torch.nn.Module):
    def __init__(self, in_columns: int, intermediate_columns: int, out_columns: int):
        """
        A macro for a double linear layer with relu activation functions
        """
        super().__init__()
        # TODO: Layer init scaling cf linear layer
        self.weights_first = torch.nn.Parameter(torch.randn(intermediate_columns, in_columns))
        self.biases_first = torch.nn.Parameter(torch.zeros(intermediate_columns))

        self.weights_second = torch.nn.Parameter(torch.randn(out_columns, intermediate_columns))
        self.biases_second = torch.nn.Parameter(torch.zeros(out_columns))

    def forward(self, inputs: torch.Tensor):
        # Memory allocation might be inefficient here but this code is better for clarity in the meantime
        intermediate_space = autograd_functions.wx_plus_b.apply(inputs, self.weights_first, self.biases_first)
        intermediate_space = autograd_functions.relu.apply(intermediate_space)
        result = autograd_functions.wx_plus_b.apply(intermediate_space, self.weights_second, self.biases_second)
        # result = autograd_functions.relu.apply(result)
        return result

class LayerNorm(torch.nn.Module):
    def __init__(self, trailing_dim_of_input: int):
        super().__init__()
        # At the beginning it should normalize without scaling so setting the weights to one and the biases to 0 will
        # allow it to at first normalize and then learn to reapply the scale
        self.weights = torch.nn.Parameter(torch.ones(trailing_dim_of_input))
        self.biases = torch.nn.Parameter(torch.zeros(trailing_dim_of_input))

    def forward(self, inputs):
        return autograd_functions.layer_normalization.apply(inputs, self.weights, self.biases)

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dimensions: int):
        super().__init__()
        # TODO Arbitrarily the randomized values to be lower for more stable gradients (gpt recommends .02)
        self.embedding_matrix = torch.nn.Parameter(torch.randn(vocab_size, embedding_dimensions))

    def forward(self, tokens):
        """
        For the tokens, each element is a token ID and each row is one sequence.
        Creates a new tensor where each corresponding token id is replaced with its embedding vector.

        :param tokens: (batch_size, sequence_length)
        !TOKENS MUST BE AN INT TENSOR! usually dtype=torch.long (otherwise the indexing won't work)
        :return: (batch_size, sequence_length, embedding_dimensions)
        """
        return autograd_functions.embedding_function.apply(tokens, self.embedding_matrix)

class SingleHeadAttention(torch.nn.Module):
    def __init__(self, embedding_dim: int, columns: int = None):
        """
        Todo write doc comments explaining single head attention
        :param embedding_dim:
        :param columns: Hidden space of Q, K, V. Also the trailing dim of the new output, by default it is the same as
        the embedding_dim value
        """
        super().__init__()
        self.columns: int = columns if columns else embedding_dim
        if self.columns < 0:
            raise ValueError(f'Dimension size cannot be negative {columns=}')

        # Need to have init dim be embedding_dim to @ the input
        # TODO SCALE (gpt recommends doing same scale as func just do 1/sqrt(embedding dim aka the last row of the input here))
        self.query_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns))
        self.key_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns))
        self.value_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns))

        self.query_bias = torch.nn.Parameter(torch.zeros(self.columns))
        self.key_bias = torch.nn.Parameter(torch.zeros(self.columns))
        self.value_bias = torch.nn.Parameter(torch.zeros(self.columns))

    def forward(self, inputs, mask=True):
        """
        :param inputs: (Batch Size, Sequence Length, Embedding Dim)
        :param mask: Whether or not a mask should be applied to the attention matrix (so that tokens can't see into the future)
        :return: (Batch Size, Sequence Length, Columns)
        """
        # Each is now (Batch Size, Sequence Length, Columns)
        query = autograd_functions.wx_plus_b.apply(inputs, self.query_weights, self.query_bias, normal=True)
        key = autograd_functions.wx_plus_b.apply(inputs, self.key_weights, self.key_bias, normal=True)
        value = autograd_functions.wx_plus_b.apply(inputs, self.value_weights, self.value_bias, normal=True)

        # The final attention matrix should be (Seq Len, Seq Len) (essentially every token by every token)
        # To get that we need to transpose either the Q or K (by convention the K) to get
            # (Seq Len, Columns) @ (Columns, Seq Len)
        # .transpose(-2, -1) transposes the 2nd to last dimension with the last dimension
            # so (768, 12, 4) -> (768, 4, 12)
        # Seq Col @ Col Seq -> (Seq, Seq)
        attention_matrix = query @ key.transpose(-2, -1)
        # Scaled by the size of the hidden space (for some reason)
        attention_matrix /= math.sqrt(self.columns)
        if mask:
            attention_matrix = apply_mask(attention_matrix)
        # Apply softmax to get a score for each token x token that the value matrix can use
        attention_matrix = autograd_functions.softmax.apply(attention_matrix, dim=-1)
        return attention_matrix @ value

def apply_mask(attention_matrix: torch.Tensor) -> torch.Tensor:
        # Attention matrix is (.. ,seq_len, seq_len)
        sequence_length = attention_matrix.shape[-1]
        # Shape: (T, T)
        # .tril() stands for lower triangle, it divides the matrix in half across the diagonal and everything in the
        # upper right half is set to 0 by default, here because dtype=bool it is set to false
        # torch.tril(torch.ones(4, 4, dtype=torch.bool)) becomes:
        #     [ True, False, False, False],
        #     [ True,  True, False, False],
        #     [ True,  True,  True, False],
        #     [ True,  True,  True,  True]
        # Basically because the matrix is (seq, seq) every row stands for one token of the input so by doing a mask
        # like this you are preventing the token from accessing information about tokens ahead of it
        # so the first token (which is the first row) can only look at itself,
        # and the second token in the second row can look at itself and the last one etc.
        mask = torch.tril(torch.ones(sequence_length, sequence_length, device=attention_matrix.device, dtype=torch.bool))

        # Broadcasts from (T, T) to (B, T, T)
        # ~ Means not so it will flip the matrix
        #   So:
        #     [True, False, False]        [False,  True,  True]
        #     [True,  True, False]   ->   [False, False,  True]
        #     [True,  True,  True]        [False, False, False]

        # And where the value is now true it will fill with negative infinity
        # the reason being after softmax it essentially becomes 0
        return attention_matrix.masked_fill(~mask, float("-inf"))
