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
--MultiHeadAttention
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
    A generic linear layer at init should scaled by the 1/sqrt(in_features), though this could change from the activation function for example:
            Xavier/Glorot style, often for tanh/sigmoid-ish balanced layers
                W = torch.randn(in_features, out_features) * math.sqrt(2 / (in_features + out_features))
            Kaiming/He style, often for ReLU networks
                W = torch.randn(in_features, out_features) * math.sqrt(2 / in_features)
            etc...
    The reason being that the activation function gates a lot of the output so the initial scale should be different

"""
class LinearLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, initialization_scaling: float = None):
        """
        Layer initialization defaults to `1/sqrt(in_features)`
        Layer wx+b where
            x (..., in_features)
            w (out_features, in_features)
            b (out_features,)

        Where
            `in features` should be the trailing dimension of the input of this layer

            `out features` should be the amount of columns you want to add or remove from the matrix in the pipeline (after the transformation)

        """
        super().__init__()
        # Layer initialization because randn's std is too big leading to big gradients and inefficient learning
        # Generic layer init is torch.randn(...) / sqrt(in_features)
        if initialization_scaling is None:
            initialization_scaling = 1 / math.sqrt(in_features)
        self.initialization_scaling = initialization_scaling
        self.weights = torch.nn.Parameter(torch.randn(out_features, in_features)) * initialization_scaling
        self.biases = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, inputs: torch.Tensor):
        return autograd_functions.wx_plus_b.apply(inputs, self.weights, self.biases)

class DoubleLinearApplied(torch.nn.Module):
    def __init__(
            self,
            in_columns: int,
            intermediate_columns: int,
            out_columns: int,
            activation_func: torch.autograd.Function = autograd_functions.gelu,
            initialization_scaling: float = None
    ):
        """
        A macro for a double linear layer with an activation function
        """
        super().__init__()

        self.activation_func: torch.autograd.Function = activation_func
        # Using kaiming-he scaling because activation function will likely be relu or gelu
        self.initialization_scaling = initialization_scaling if initialization_scaling is not None else (2/math.sqrt(in_columns))
        self.weights_first = torch.nn.Parameter(torch.randn(intermediate_columns, in_columns)) * self.initialization_scaling
        self.biases_first = torch.nn.Parameter(torch.zeros(intermediate_columns))

        # Regular init scaling here because no activation function after
        self.weights_second = torch.nn.Parameter(torch.randn(out_columns, intermediate_columns)) / math.sqrt(intermediate_columns)
        self.biases_second = torch.nn.Parameter(torch.zeros(out_columns))

    def forward(self, inputs: torch.Tensor):
        # Memory allocation might be inefficient here but this code is better for clarity in the meantime
        intermediate_space = autograd_functions.wx_plus_b.apply(inputs, self.weights_first, self.biases_first)
        intermediate_space = self.activation_func.apply(intermediate_space)
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
        # Scale the matrix by an arbitrary scalar for the randomized weights to be lower for more stable gradients (gpt recommends .02)
        # Maybe parametrize it
        self.embedding_matrix = torch.nn.Parameter(torch.randn(vocab_size, embedding_dimensions)) * .02

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
        if self.columns <= 0:
            raise ValueError(f'Dimension size must be over 0 {columns=}')

        # Need to have init dim be embedding_dim to @ the input
        # Also need to scale the randomly initialized weights for more stable training at the beginning
        # SCALE (gpt recommends doing same scale as func just do 1/sqrt(embedding dim aka the in_features))
        self.query_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns)) / math.sqrt(embedding_dim)
        self.key_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns)) / math.sqrt(embedding_dim)
        self.value_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns)) / math.sqrt(embedding_dim)

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
        # Return (Batch Size, Sequence Length, Columns)
        return attention_matrix @ value

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding_dim: int, num_of_heads: int, columns: int = None, project_to_embedding_dim: bool = True):
        """
        Todo write doc comments explaining the concept behind multihead attention
        :param embedding_dim: The embedding_dim/channels/hidden_space/whatever you want to call it of the input
        :param num_of_heads: The number of heads to split the columns to. The following must be true (self.columns % num_of_heads == 0)
        :param columns: Hidden space of Q, K, V. Also the trailing dim of the new output, by default it is the same as
        the embedding_dim value
        :param project_to_embedding_dim: If `True` projects the final return of the forward pass to (batch_size, sequence_length, embedding_dim)
        if `False` the forward pass returns (batch_size, sequence_length, columns)
        """
        super().__init__()

        # <editor-fold desc="Input Validation & Attribute Creation">
        if not isinstance(embedding_dim, int):
            raise TypeError("embedding_dim must be an int")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if columns is not None and not isinstance(columns, int):
            raise TypeError("columns must be an int or None")
        self.columns: int = columns if columns is not None else embedding_dim
        if self.columns <= 0:
            raise ValueError(f'Dimension size must be over 0 {self.columns=}')

        if not isinstance(num_of_heads, int):
            raise TypeError("num_heads must be an int")
        if num_of_heads <= 0:
            raise ValueError("num_heads must be positive")
        if num_of_heads > self.columns:
            raise ValueError(f"num_of_heads {num_of_heads} cannot be greater than columns {self.columns}")
        if self.columns % num_of_heads != 0:
            raise ValueError(f'Columns {self.columns} is not divisible by {num_of_heads}, the modulo is {self.columns % num_of_heads}')

        self.embedding_dim = embedding_dim
        self.num_of_heads = num_of_heads
        self.dimensions_per_head = self.columns // num_of_heads
        self.final_linear_layer_projection_dimensions = embedding_dim if project_to_embedding_dim else self.columns

        # </editor-fold>

        # Need to have init dim be embedding_dim to @ the input
        # Also need to scale the randomly initialized weights for more stable training at the beginning
        # SCALE (gpt recommends doing same scale as func just do 1/sqrt(embedding dim aka the in_features))
        self.query_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns)) / math.sqrt(embedding_dim)
        self.key_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns)) / math.sqrt(embedding_dim)
        self.value_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns)) / math.sqrt(embedding_dim)
        # Here the in feature is self.columns
        self.final_linear_weights = torch.nn.Parameter(torch.randn(self.columns, self.final_linear_layer_projection_dimensions)) / math.sqrt(self.columns)

        self.query_bias = torch.nn.Parameter(torch.zeros(self.columns))
        self.key_bias = torch.nn.Parameter(torch.zeros(self.columns))
        self.value_bias = torch.nn.Parameter(torch.zeros(self.columns))
        self.final_linear_bias = torch.nn.Parameter(torch.zeros(self.final_linear_layer_projection_dimensions))

    def forward(self, inputs, mask=True):
        # Assuming that input is (Batch Size, Sequence Length, Embedding Dimensions)
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected input last dim to be init's embedding_dim {self.embedding_dim}, got {inputs.shape[-1]}")

        # Each is now (Batch Size, Sequence Length, Columns)
        query: torch.Tensor = autograd_functions.wx_plus_b.apply(inputs, self.query_weights, self.query_bias, normal=True)
        key = autograd_functions.wx_plus_b.apply(inputs, self.key_weights, self.key_bias, normal=True)
        value = autograd_functions.wx_plus_b.apply(inputs, self.value_weights, self.value_bias, normal=True)

        # Retrieve other useful info
        batch_size, sequence_length, columns = query.shape

        # Split the columns into a number of heads
        # For example if num_of_heads = 4 and columns is 24 (..., 24) -> (..., 4, 6)
        # Each is now (Batch Size, Sequence Length, Number of Heads, Dimensions Per Head)
        query = query.view(batch_size, sequence_length, self.num_of_heads, self.dimensions_per_head)
        key = key.view(batch_size, sequence_length, self.num_of_heads, self.dimensions_per_head)
        value = value.view(batch_size, sequence_length, self.num_of_heads, self.dimensions_per_head)

        # Since heads are really just a second batch we need to move them back to do the operation on the hidden dimension
        # aka dim_per_head and the sequence length, to still get an attention matrix (seq_len, seq_len) just over the
        # number_of_dims given on that specific head. This is because the @ is batched over batch size and heads to do
        # (..., seq_len, dim_per_head) @ (..., seq_len, dim_per_head).T
        # Each is now (Batch Size, Number of Heads, Sequence Length, Dimensions Per Head)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Now we calculate the attention matrix (because this is batched over the batches and the num of heads, every single one
        # is done in this line below). We have to transpose key to make the operation
        # (..., seq_len, dim_per_head) @ (..., dim_per_head, seq_len) -> (..., seq_len, seq_len)
        # The shape of this is (batch_size, num_of_heads, sequence_length, sequence_length)
        head_separated_attention_matrix = query @ key.transpose(-2, -1)

        # Normally you'd scale off of columns but because each mini batch (each head) has a hidden space of dimensions_per_head
        # you scale it off of that.
        head_separated_attention_matrix /= math.sqrt(self.dimensions_per_head)

        # Apply the mask
        if mask:
            head_separated_attention_matrix = apply_mask(head_separated_attention_matrix)

        # Softmax the attention matrices
        head_separated_attention_matrix = autograd_functions.softmax.apply(head_separated_attention_matrix, dim=-1)

        # Get the results per head
        # Shape is (batch_size, num_of_heads, sequence_length, dimensions_per_head)
        head_separated_results = head_separated_attention_matrix @ value

        # Now we need to recombine the heads to get the full embedding dimension back (basically we separated them earlier,
        # now we are recombining).
        # First we need to move num_of_heads to the end right before dimensions_per_head to combine them
        # Shape is now (batch_size, sequence_length, num_of_heads, dimensions_per_head)
        head_separated_results = head_separated_results.transpose(1, 2)

        # We then recombine num_of_heads & dimensions_per_head
        # Shape is (batch_size, sequence_length, columns)
        combined_results = head_separated_results.reshape(batch_size, sequence_length, columns)

        # Multihead attention usually has a final linear layer to combine the results of all the heads and also to project
        # the output to an expected output like the embedding_dimension for the residual connections.
        # (AKA just adding the result of this to the original input, but in order to do that they need to be the same shape).
        # Final shape (batch_size, sequence_length, columns) or (..., embedding_dim) depending on this param in the init `project_to_embedding_dim`
        # AKA embedding_dim instead of columns if project_to_embedding_dim is True
        return autograd_functions.wx_plus_b.apply(combined_results, self.final_linear_weights, self.final_linear_bias, normal=True)

class TransformerBlock(torch.nn.Module):
    def __init__(self):
        # TODO 3 (doc comment)
        super().__init__()
        # TODO
    def forward(self, input):
        # TODO 2



def apply_mask(attention_matrix: torch.Tensor) -> torch.Tensor:
        # Attention matrix is (... , Sequence Length, Sequence Length)
        sequence_length = attention_matrix.shape[-1]
        # Shape: (Sequence Length, Sequence Length)
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

        # Broadcasts from (Sequence Length, Sequence Length) to (Batch Size, Sequence Length, Sequence Length)
        # ~ Means not so it will flip the matrix
        #   So:
        #     [True, False, False]        [False,  True,  True]
        #     [True,  True, False]   ->   [False, False,  True]
        #     [True,  True,  True]        [False, False, False]

        # And where the value is now true it will fill with negative infinity
        # the reason being after softmax it essentially becomes 0
        return attention_matrix.masked_fill(~mask, float("-inf"))
