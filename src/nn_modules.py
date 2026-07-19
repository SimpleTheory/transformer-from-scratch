import torch
import autograd_functions
import math

"""
... Then build Modules:

--Linear
--LayerNorm
--FeedForward
--MoEFeedForward
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
    Syntax Warning: The initialization must take place within the nn.Parameter(...) otherwise it loses its parameter status

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
        self.weights = torch.nn.Parameter(torch.randn(out_features, in_features) * initialization_scaling)
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
        A macro for a double linear layer with an activation function. The idea is to blow up the hidden space to let the
        model make better decisions about the results of attention then shrink it back down to the original size. In
        practice, it would look something like:

        (Batch, Sequence Len, Embedding) -> (Batch, Sequence Len, Embedding * 4) -> (Batch, Sequence Len, Embedding)
        """
        super().__init__()

        self.activation_func: torch.autograd.Function = activation_func
        # Using kaiming-he scaling because activation function will likely be relu or gelu
        self.initialization_scaling = initialization_scaling if initialization_scaling is not None else (math.sqrt(2/in_columns))
        self.weights_first = torch.nn.Parameter(torch.randn(intermediate_columns, in_columns) * self.initialization_scaling)
        self.biases_first = torch.nn.Parameter(torch.zeros(intermediate_columns))

        # Regular init scaling here because no activation function after
        self.weights_second = torch.nn.Parameter(torch.randn(out_columns, intermediate_columns) / math.sqrt(intermediate_columns))
        self.biases_second = torch.nn.Parameter(torch.zeros(out_columns))

    def forward(self, inputs: torch.Tensor):
        # Memory allocation might be inefficient here but this code is better for clarity in the meantime
        intermediate_space = autograd_functions.wx_plus_b.apply(inputs, self.weights_first, self.biases_first)
        intermediate_space = self.activation_func.apply(intermediate_space)
        result = autograd_functions.wx_plus_b.apply(intermediate_space, self.weights_second, self.biases_second)
        # result = autograd_functions.relu.apply(result)
        return result

class MoEDoubleLinearApplied(torch.nn.Module):
    def __init__(self,
                 total_number_of_experts: int,
                 experts_to_accept: int,
                 in_columns: int,
                 intermediate_columns: int,
                 activation_func: torch.autograd.Function = autograd_functions.gelu,
                 out_columns: int = None,
                 initialization_scaling: float = None
                 ):
        super().__init__()
        # Keep dimensionality by default for residuals
        if out_columns is None:
            out_columns = in_columns
        # Kaiming initialization for gelu style default activation functions
        if initialization_scaling is None:
            initialization_scaling = math.sqrt(2/in_columns)
        if experts_to_accept > total_number_of_experts:
            raise ValueError(f"{experts_to_accept=} cannot be greater than {total_number_of_experts=}")
        self.total_number_of_experts = total_number_of_experts
        self.experts_to_accept = experts_to_accept
        self.in_columns = in_columns
        self.intermediate_columns = intermediate_columns
        self.activation_func = activation_func
        self.out_columns = out_columns
        self.initialization_scaling = initialization_scaling
        # This must be a ModuleList so PyTorch registers the experts' parameters.
        # Otherwise, optimizer.parameters(), state_dict(), etc. will not include them.
        self.experts = torch.nn.ModuleList([
            DoubleLinearApplied(in_columns, intermediate_columns, out_columns, activation_func, initialization_scaling)
            for _ in range(total_number_of_experts)
        ])
        self.gate = LinearLayer(in_columns, total_number_of_experts)

    def forward(self, input_tensor):
        """
        :param input_tensor: Batch Size, Sequence Length, Embedding Dimensions
        :return: Batch Size, Sequence Length, Out Columns
        """
        batch_size, sequence_length, embedding_dims = input_tensor.shape
        # Sanity check on input
        if embedding_dims != self.in_columns:
            raise ValueError(
                f"Expected input last dimension {self.in_columns}, got {embedding_dims}"
            )
        # Reshape batch * seq_len (henceforth called N), embedding
        n = batch_size * sequence_length
        flattened_input = input_tensor.reshape(-1, embedding_dims)
        # This will be (N, total_number_of_experts)
            # We do this to get the scores for each element in the input
        scores = self.gate(flattened_input)
        # This will be (N, number_of_experts_to_use)
        # The filtered scores are the k highest scores so: [0, .1, .2, .3, .4] where k=2 -> [.3, .4] & [3, 4]
        # We take their row index (as in the indices of the values of that specific row)
        # because that will be the same index of the expert in self.experts
        filtered_scores, index_of_expert_to_use = torch.topk(
            scores,
            k=self.experts_to_accept,
            dim=-1,
        )
        # Once filtered we convert the values to weights of how much that expert should apply
        weights_per_expert_per_n = autograd_functions.softmax_with_kwarg(filtered_scores, dim=-1)

        # The idea in the next segment is we want to flatten everything because the same token will be routed to many different places
        # We want it such that we can have a data structure that can keep track of:
            # 1. Which element it was (token_indices)
            # 2. Which expert to use (flattened_expert_index)
            # 3. What weight to apply to the result (flattened_weights)
        # This works because we are making three tables with identical sizes (n, num_of_experts_to_use)

        token_indices = (
            # Like Python range(n)
            torch.arange(n, device=input_tensor.device)
            # Turn the range into a column (so (n,) -> (n,1))
            .unsqueeze(1)
            # This method takes a dimension of 1 and duplicates x times (though this is done as a view to save memory)
            # So now you will have `experts_to_accept` copies of the range column
            .expand(n, self.experts_to_accept)
            # Flatten the range to one dimension so if `experts_to_accept` was 3 then [0,0,0,1,1,1,...]
            .reshape(-1)
        )
        flattened_weights = weights_per_expert_per_n.reshape(-1)
        flattened_expert_index = index_of_expert_to_use.reshape(-1)

        # We can then sort each of them by the experts
        # and then batch by expert for efficiency reasons
        sort_index = torch.argsort(flattened_expert_index)
        token_indices = token_indices[sort_index]
        flattened_weights = flattened_weights[sort_index]
        flattened_expert_index = flattened_expert_index[sort_index]

        # Gets the count for each index at its position replacing misses with a 0
        # For example say total experts is 5
            # flattened expert index [2, 3, 4, 0, 0, 0, 3]
            # return [3, 0, 1, 2, 1]
            #         0  1  2  3  4  -- it is the count of how often the indices show up in the source
        # We need this basically to use slicing to batch by the experts, if I know 3 elements are going to expert 0
        # Then I can slice each of the tensors equally by 3
        hits_per_expert = torch.bincount(
            flattened_expert_index,
            minlength=self.total_number_of_experts,
        ).tolist()

        # Create an output buffer
        result = torch.zeros(n, self.out_columns, device=input_tensor.device, dtype=input_tensor.dtype)

        start = 0
        for current_expert, hits in enumerate(hits_per_expert):
            if hits == 0:
                continue
            # Create a slice object for the current expert
            end = start + hits
            current_slice = slice(start, end)

            # Slice `token_indices` list and `flattened_weights` list as mentioned above using the above slice obj
            current_token_indices = token_indices[current_slice]
            current_flattened_weights = flattened_weights[current_slice]
            # This is the list of the actual token vectors retrieved via the token_index
            tokens_sent_to_this_expert = flattened_input[current_token_indices]  # (hits, embedding_dim)

            # Get the experts output and apply the weight
            current_expert_output = self.experts[current_expert](tokens_sent_to_this_expert)  # (hits, out_columns) ideally embedding_dim
            current_expert_output = current_expert_output * current_flattened_weights.unsqueeze(-1)  # Columnize them because its one weight per row

            # At each token's index add its output to that index in the buffer
            result.index_add_(
                dim=0,
                index=current_token_indices,
                source=current_expert_output,
            )

            # Adjust the start for the next slice
            start = end

        # Reshape the buffer to (Batch Size, Sequence Length, Out Columns)
        return result.reshape(batch_size, sequence_length, self.out_columns)

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
    def __init__(self, vocab_size: int, embedding_dimensions: int, initializer=.02):
        super().__init__()
        # Scale the matrix by an arbitrary scalar for the randomized weights to be lower for more stable gradients (gpt recommends .02)
        # Maybe parametrize it
        self.embedding_matrix = torch.nn.Parameter(torch.randn(vocab_size, embedding_dimensions) * initializer)

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
        Todo write about the concepts behind single head attention
        :param embedding_dim:
        :param columns: Hidden space of Q, K, V. Also the trailing dim of the new output, by default it is the same as
        the embedding_dim value
        """
        super().__init__()
        self.columns: int = columns if columns else embedding_dim
        if self.columns <= 0:
            raise ValueError(f'Dimension size must be over 0 {columns=}')

        # Need to have init dim be embedding_dim to @ the input
        # The randomly initialized weights are scaled for more stable training at the beginning
        # gpt recommends using the default scale of 1/sqrt(embedding dim aka the in_features)
            # this is usually the default scale for weights that aren't passed through activation functions later
        self.query_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns) / math.sqrt(embedding_dim))
        self.key_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns) / math.sqrt(embedding_dim))
        self.value_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns) / math.sqrt(embedding_dim))

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
        query = autograd_functions.wx_plus_b_with_kwarg(inputs, self.query_weights, self.query_bias, normal=True)
        key = autograd_functions.wx_plus_b_with_kwarg(inputs, self.key_weights, self.key_bias, normal=True)
        value = autograd_functions.wx_plus_b_with_kwarg(inputs, self.value_weights, self.value_bias, normal=True)

        # The final attention matrix should be (Seq Len, Seq Len) (essentially every token by every token)
        # To get that we need to transpose either the Q or K (by convention the K) to get
            # (Seq Len, Columns) @ (Columns, Seq Len)
        # .transpose(-2, -1) transposes the 2nd to last dimension with the last dimension
            # so (768, 12, 4) -> (768, 4, 12)
        # (Seq Len, Columns) @ (Columns, Seq Len) -> (Seq, Seq)
        attention_matrix = query @ key.transpose(-2, -1)
        # Scaled by the size of the hidden space (for some reason)(, avoided inplace operation for pytorch debugging)
        attention_matrix = attention_matrix / math.sqrt(self.columns)
        if mask:
            attention_matrix = apply_mask(attention_matrix)
        # Apply softmax to get a score for each (token x token) that the value matrix can use
        attention_matrix = autograd_functions.softmax_with_kwarg(attention_matrix, dim=-1)

        # Multiply the scores of each token x token to the value matrix
        # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, columns)

        return attention_matrix @ value  # Return (Batch Size, Sequence Length, Columns)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embedding_dim: int, num_of_heads: int, columns: int = None, project_to_embedding_dim: bool = True):
        """
        Todo write about the concepts behind multihead attention
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
        self.query_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns) / math.sqrt(embedding_dim))
        self.key_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns) / math.sqrt(embedding_dim))
        self.value_weights = torch.nn.Parameter(torch.randn(embedding_dim, self.columns) / math.sqrt(embedding_dim))
        # Here the in feature is self.columns
        self.final_linear_weights = torch.nn.Parameter(torch.randn(self.columns, self.final_linear_layer_projection_dimensions) / math.sqrt(self.columns))

        self.query_bias = torch.nn.Parameter(torch.zeros(self.columns))
        self.key_bias = torch.nn.Parameter(torch.zeros(self.columns))
        self.value_bias = torch.nn.Parameter(torch.zeros(self.columns))
        self.final_linear_bias = torch.nn.Parameter(torch.zeros(self.final_linear_layer_projection_dimensions))

    def forward(self, inputs, mask=True):
        # Assuming that input is (Batch Size, Sequence Length, Embedding Dimensions)
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected input last dim to be init's embedding_dim {self.embedding_dim}, got {inputs.shape[-1]}")

        # Each is now (Batch Size, Sequence Length, Columns)
        query: torch.Tensor = autograd_functions.wx_plus_b_with_kwarg(inputs, self.query_weights, self.query_bias, normal=True)
        key = autograd_functions.wx_plus_b_with_kwarg(inputs, self.key_weights, self.key_bias, normal=True)
        value = autograd_functions.wx_plus_b_with_kwarg(inputs, self.value_weights, self.value_bias, normal=True)

        # Retrieve other useful info
        batch_size, sequence_length, columns = query.shape

        # Split the columns into a number of heads
        # For example if num_of_heads = 4 and columns is 24 (..., 24) -> (..., 4, 6)
        # Each is now (Batch Size, Sequence Length, Number of Heads, Dimensions Per Head)
        query = query.view(batch_size, sequence_length, self.num_of_heads, self.dimensions_per_head)
        key = key.view(batch_size, sequence_length, self.num_of_heads, self.dimensions_per_head)
        value = value.view(batch_size, sequence_length, self.num_of_heads, self.dimensions_per_head)

        # Since heads are really just a second batch we need to move them back to do the operation on the hidden dimension
        # aka dimensions_per_head. By doing this we can still get an attention matrix (seq_len, seq_len) just over the
        # dimensions_per_head of that specific head.
        #  This is because the @ is batched over batch size and heads to do
        # (..., seq_len, dimensions_per_head) @ (..., seq_len, dimensions_per_head).T

        # Each is now (Batch Size, Number of Heads, Sequence Length, Dimensions Per Head)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Now we calculate the attention matrix, because the key tensor and query tensor are double batched (over the batches, num of heads),
        # every single attention matrix is computed in the line below.

        # We have to transpose key to make the following operation valid:
        # (..., seq_len, dim_per_head) @ (..., dim_per_head, seq_len) -> (..., seq_len, seq_len)

        # The shape of this is (batch_size, num_of_heads, sequence_length, sequence_length)
        head_separated_attention_matrix = query @ key.transpose(-2, -1)

        # Normally you'd scale off of columns but because each mini batch (each head) has a hidden space of dimensions_per_head
        # you scale it off of that.
        head_separated_attention_matrix = head_separated_attention_matrix / math.sqrt(self.dimensions_per_head)

        # Apply the mask
        if mask:
            head_separated_attention_matrix = apply_mask(head_separated_attention_matrix)

        # Softmax the attention matrices
        head_separated_attention_matrix = autograd_functions.softmax_with_kwarg(head_separated_attention_matrix, dim=-1)

        # Get the results per head
        # Shape is (Batch Size, Number of Heads, Sequence Length, Dimensions Per Head)
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
        # (Residual connections just means adding the result of this to the original input, but in order to do that they need to be the same shape).
        # Final shape (batch_size, sequence_length, columns) or (..., embedding_dim) depending on this param in the init `project_to_embedding_dim`
        # AKA embedding_dim instead of columns if project_to_embedding_dim is True
        return autograd_functions.wx_plus_b_with_kwarg(combined_results, self.final_linear_weights, self.final_linear_bias, normal=True)

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
            ff_experts_to_accept: int = 2
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
        input_copy = input_copy + self.attention_block(self.layer_norm_1(input_copy))

        # Mix and expand the output from the attention with a linear layer then reproject it down to size `embedding_dim`
        # with a second linear layer. Since the linear layers are connected to each other they are split with an activation
        # function to introduce non-linearity. Otherwise, they'd be equivalent to one giga linear layer.
        input_copy = input_copy + self.feed_forward(self.layer_norm_2(input_copy))

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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length
        self.total_blocks = total_blocks
        self.ff_intermediate_columns_multiplier = ff_intermediate_columns_multiplier

        self.token_embeddings = EmbeddingLayer(vocab_size, embedding_dimension)
        self.positional_embeddings = EmbeddingLayer(max_sequence_length, embedding_dimension)

        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(
                embedding_dimension=embedding_dimension,
                num_of_heads=num_heads,
                ff_intermediate_columns=embedding_dimension * ff_intermediate_columns_multiplier
                # skip_attention_layer_norm=False,
            )
            for _ in range(total_blocks)
        ])
        self.final_layer_norm = LayerNorm(embedding_dimension)
        self.linear_to_vocab = LinearLayer(embedding_dimension, vocab_size)

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
        data_to_work_on = token_vectors + position_vectors
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
    def generate(
            self,
            token_ids: torch.Tensor,
            max_new_tokens: int,
    ):
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
