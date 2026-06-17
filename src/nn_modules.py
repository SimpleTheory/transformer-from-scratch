import torch
import autograd_functions

"""
... Then build Modules:

--Linear
--LayerNorm
--FeedForward
--Embedding
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
    Don't use @dataclass for nn.Modules (while learning pytorch) because order matter on when super init is called and the attributes are defined


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


