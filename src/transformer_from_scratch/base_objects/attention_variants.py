"""
GHQ Grouped Head Query -> Share KV (and their caches) across multiple heads such that say 8 heads use the same kv weights
so the only things computed are the current words kv vectors to add to their respective tables and each query when creating
those tensors.

MLA Multihead Latent Attention -> Compress the KVQ into a smaller matrix and derive them from there, takes less space than the former
but more compute, however since you control the intermediate representation's size that changeable and it's better at scale since
in compression you might lose a lot of information and make the model harder to train. The compression fact does mess with Rope though
so they usually do partial rope where they apply it in a special way and compress that separately then recombine everything in the end.

Sparse Attention/SLA Attention and all subvariants -> Different ways of manipulating the mask so instead of
a casual no future mask you might do a window around the current token, you might have a gate like a small MLP or a similarity score
that disables certain columns you might insert additional context tokens or compress tokens, or have a sliding
window but let some tokens see globally, and many more changes like this.
Essentially its manipulating the attention matrix itself and its mask.

Gated Attention -> Compute a fourth G matrix from input aside from KQV (or compress in MLA) to use as a gate and you use
a more traditional layernorm for KV (at some point) because of the instability caused to the gradients by the gate the
article also mentioned they tend to use partial rope as well. Usually used in hybrid attention for the full block

Delta Net Hybrid Attention -> In the model use full attention blocks only sometimes and different ultra light weight non attention
architectures in other blocks to speed up model and reduce size. Usually they use Delta Net as their alternate architecture
which uses a table (sequence length, some embedding size) and they use that to pass on info. They then modify the values
in the table on the fly with each new entry relative to what the new entry is. In more detail, the new entry creates an update table which
it uses to update the block's original table essentially using itself as the thing that needed to be predicted along
with appending itself to the memory table.
"""
import torch
import transformer_from_scratch.base_objects.autograd_functions_modernization as autograd_functions
import transformer_from_scratch.base_objects.nn_module_modernizations as nn_modules
import math


class GHQCache:
    def __init__(self, key: torch.Tensor, value: torch.Tensor):
        self.key = key
        self.value = value

    def increment_key(self, new_key):
        self.key = torch.cat([self.key, new_key], -2)

    def increment_value(self, new_value):
        self.value = torch.cat([self.value, new_value], -2)

    def current_position(self):
        # Size of dim Sequence Length
        return self.key.shape[-2]

class GHQ(torch.nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_of_heads: int,
            num_of_kv_groups: int,
            rope_parameters: autograd_functions.RopeParameters,
            dimensions_per_head: int | None = None,
            columns: int = None,
            project_to_embedding_dim: bool = True,
            rope_dimensions: int = None,
            use_qk_norm: bool = True,
    ):
        """
        Todo write about GHQ
        :param embedding_dim: The embedding_dim/channels/hidden_space/whatever you want to call it of the input
        :param num_of_heads: The number of heads to split the columns to. The following must be true (self.columns % num_of_heads == 0)
        :param num_of_kv_groups: The number of KV groups to across all the different heads must be `Heads % KV Groups == 0`
        :param columns: Hidden space of Q, K, V. Also the trailing dim of the new output, by default it is the same as
        the embedding_dim value
        :param project_to_embedding_dim: If `True` projects the final return of the forward pass to (batch_size, sequence_length, embedding_dim)
        if `False` the forward pass returns (batch_size, sequence_length, columns)
        :param rope_dimensions: The count of dimensions to apply rope on must be even and must be `<= embedding_dim // num_of_heads`
        :param use_qk_norm: If True will create RMSNorms for Q and K and it will apply them in the forward pass after they are created
        """
        super().__init__()

        # <editor-fold desc="Attribution">
        self.columns: int = columns if columns is not None else embedding_dim
        self.embedding_dim = embedding_dim
        self.num_of_heads = num_of_heads
        self.num_kv_groups = num_of_kv_groups
        self.final_linear_layer_projection_dimensions = embedding_dim if project_to_embedding_dim else self.columns
        # This is done so that this nn.Module class is not recognized as belonging here. Since it is really just a pointer
        # coming in from the model. (As I am passing a user created object which in python by default passes a pointer)
        object.__setattr__(self, "rope_parameters", rope_parameters)
        self.use_qk_norm = use_qk_norm
        # </editor-fold>

        # <editor-fold desc="Input Validation">
        if not isinstance(embedding_dim, int):
            raise TypeError("embedding_dim must be an int")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if columns is not None and not isinstance(columns, int):
            raise TypeError("columns must be an int or None")
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
        if not isinstance(num_of_kv_groups, int):
            raise TypeError("num_of_kv_groups must be an int")

        if any((
                num_of_kv_groups > num_of_heads,
                num_of_kv_groups < 1,
                num_of_heads % num_of_kv_groups != 0
        )):
            raise ValueError(f'Invalid number of KV groups {num_of_kv_groups} relative to number of heads {num_of_heads}.')
        if rope_dimensions is not None and not isinstance(rope_dimensions, int):
            raise TypeError("dim_to_apply_rope_on must be an int or None")
        # </editor-fold>

        # <editor-fold desc="Derived Attributes and their Validation">
        # <editor-fold desc="Dimensions Per Head Logic with Columns">
        # The idea here being your dimensions per head are dependent on your columns which are essentially the hidden space
        # of these operations. As such they need to be explicitly dependent on each other.
        if dimensions_per_head is None:
            self.columns = embedding_dim if columns is None else columns
            if self.columns % num_of_heads != 0:
                raise ValueError(...)
            self.dimensions_per_head = self.columns // num_of_heads
        else:
            self.dimensions_per_head = dimensions_per_head
            expected_columns = num_of_heads * dimensions_per_head
            if columns is not None and columns != expected_columns:
                raise ValueError(f"columns={columns} but num_heads * head_dim={expected_columns}")
            self.columns = expected_columns
        # </editor-fold>

        self.group_size = num_of_heads // num_of_kv_groups

        self.rope_dim = self.dimensions_per_head if rope_dimensions is None else rope_dimensions

        if any((self.rope_dim % 2 != 0, self.rope_dim > self.dimensions_per_head, self.rope_dim <= 0)):
            raise ValueError(f'Invalid dimensions specified for RoPE {rope_dimensions},'
                             f' given head dimension size of {self.dimensions_per_head}')
        # </editor-fold>

        # <editor-fold desc="Layers">
        self.query_weights = nn_modules.LinearLayer.from_feature_counts(embedding_dim, self.columns, bias=False)
        self.key_weights = nn_modules.LinearLayer.from_feature_counts(embedding_dim, self.num_kv_groups * self.dimensions_per_head, bias=False)
        self.value_weights = nn_modules.LinearLayer.from_feature_counts(embedding_dim, self.num_kv_groups * self.dimensions_per_head, bias=False)
        # Here the in feature is self.columns
        self.final_linear_weights = nn_modules.LinearLayer.from_feature_counts(self.columns, self.final_linear_layer_projection_dimensions, bias=False)

        if use_qk_norm:
            self.query_norm = nn_modules.RMSNorm(self.dimensions_per_head)
            self.key_norm = nn_modules.RMSNorm(self.dimensions_per_head)
        # </editor-fold>

    # noinspection DuplicatedCode
    def forward(self, inputs, cache=None, use_cache=False, mask=True) -> tuple[torch.Tensor, GHQCache]:
        # <editor-fold desc="Create QKV">
        # Now forward is only expecting the newest token(s) in the sequence
        # Assuming that input is (Batch Size, Sequence Length, Embedding Dimensions)
        if inputs.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected input last dim to be init's embedding_dim {self.embedding_dim}, got {inputs.shape[-1]}")

        # Each is now (Batch Size, Newest Tokens, Columns)
        query = self.query_weights(inputs)
        new_keys = self.key_weights(inputs)
        new_values = self.value_weights(inputs)
        batch_size, newest_tokens, columns = query.shape
        # </editor-fold>

        # <editor-fold desc="Resizing">
        # Split the columns into a number of heads
        # For example if num_of_heads = 4 and columns is 24 (..., 24) -> (..., 4, 6)
        # Query is now (Batch Size, Sequence Length, Number of Heads, Dimensions Per Head)
        # Key & Value are now (Batch Size, Sequence Length, KV Groups, Dimensions Per Head)
        query = query.view(batch_size, newest_tokens, self.num_of_heads, self.dimensions_per_head)
        new_keys = new_keys.view(batch_size, newest_tokens, self.num_kv_groups, self.dimensions_per_head)
        new_values = new_values.view(batch_size, newest_tokens, self.num_kv_groups, self.dimensions_per_head)

        # Since heads are really just a second batch we need to move them back to do the operation on the hidden dimension
        # aka dimensions_per_head. By doing this we can still get an attention matrix (seq_len, seq_len) just over the
        # dimensions_per_head of that specific head.
        #  This is because the @ is batched over batch size and heads to do
        # (..., seq_len, dimensions_per_head) @ (..., seq_len, dimensions_per_head).T

        # Q is now (Batch Size, Number of Heads, Sequence Length, Dimensions Per Head)
        # KV is now (Batch Size, KV Groups, Sequence Length, Dimensions Per Head)
        query = query.transpose(1, 2)
        new_keys = new_keys.transpose(1, 2)
        new_values = new_values.transpose(1, 2)
        # </editor-fold>

        # <editor-fold desc="Apply QK Norm and Rope">
        current_offset_for_rope = cache.current_position() if (use_cache and cache is not None) else 0

        if self.use_qk_norm:
            query = self.query_norm(query)
            new_keys = self.key_norm(new_keys)

        new_keys = autograd_functions.apply_partial_rope(self.rope_dim, new_keys, *self.rope_parameters, current_offset_for_rope)
        query = autograd_functions.apply_partial_rope(self.rope_dim, query, *self.rope_parameters, current_offset_for_rope)
        # </editor-fold>

        # <editor-fold desc="Update Cache">
        if cache is None and use_cache:
            cache = GHQCache(new_keys, new_values)
            key = cache.key
            value = cache.value
        elif use_cache:
            # (Batch Size, KV Groups, Sequence Length, Dimensions Per Head)
            # Thus since we want to increment the Sequence Length we concatenate on the penultimate dimension `-2`
            # cache.key = torch.cat([cache.key, new_keys], -2)
            # cache.value = torch.cat([cache.value, new_values], -2)
            cache.increment_key(new_keys)
            cache.increment_value(new_values)

            key = cache.key
            value = cache.value
        else:
            key = new_keys
            value = new_values
        # </editor-fold>

        # <editor-fold desc="Expand KV Groups to match Queries">
        # Before modifying Key is: (batch_size, kv_groups, seq_len, head_dim)
        # This duplicates the tensor at the specified dimension such that you copy each part of it `self.group_size` times
        # Suppose `self.group_size` is 4, and you have two KV Groups [K0, K1]
        # Then the result will give you (batch_size, kv_groups * group_size, seq_len, head_dim) where KV Groups will be:
        #   -> [K0, K0, K0, K0, K1, K1, K1, K1]
        # This example lets a single K group attend to 4 queries while the rest of the attention calculations stay the same as MHA
        key = key.repeat_interleave(self.group_size, dim=1)
        value = value.repeat_interleave(self.group_size, dim=1)
        # </editor-fold>

        # <editor-fold desc="Attention Calculation">
        # Now we calculate the attention matrix, because the key tensor and query tensor are double batched (over the batches, num of heads),
        # every single attention matrix is computed in the line below.

        # We have to transpose key to make the following operation valid:
        # (..., new_tokens, dim_per_head) @ (..., dim_per_head, seq_len) -> (..., new_tokens, seq_len)

        # The shape of this is (batch_size, num_of_heads, new_tokens, sequence_length)
        head_separated_attention_matrix = query @ key.transpose(-2, -1)

        # Normally you'd scale off of columns but because each mini batch (each head) has a hidden space of dimensions_per_head
        # you scale it off of that.
        head_separated_attention_matrix = head_separated_attention_matrix / math.sqrt(self.dimensions_per_head)

        # Apply the mask
        if mask:
            head_separated_attention_matrix = nn_modules.apply_causal_mask_with_cache(head_separated_attention_matrix)

        # Softmax the attention matrices
        head_separated_attention_matrix = autograd_functions.softmax_with_kwarg(head_separated_attention_matrix, dim=-1)

        # Get the results per head (N is the Newest Tokens)
        # (B, H, N, SL) @ (B, H, SL, Dh) -> (B, H, N, Dh)
        head_separated_results = head_separated_attention_matrix @ value

        # Now we need to recombine the heads to get the full embedding dimension back (basically we separated them earlier,
        # now we are recombining).
        # First we need to move num_of_heads to the end right before dimensions_per_head to combine them
        # Shape is now (batch_size, newest_tokens, num_of_heads, dimensions_per_head)
        head_separated_results = head_separated_results.transpose(1, 2)

        # We then recombine num_of_heads & dimensions_per_head
        # Shape is (batch_size, newest_tokens, columns)
        combined_results = head_separated_results.reshape(batch_size, newest_tokens, columns)
        # </editor-fold>

        # Multihead attention usually has a final linear layer to combine the results of all the heads and also to project
        # the output to an expected output like the embedding_dimension for the residual connections.
        # (Residual connections just means adding the result of this to the original input, but in order to do that they need to be the same shape).
        # Final shape (batch_size, sequence_length, columns) or (..., embedding_dim) depending on this param in the init `project_to_embedding_dim`
        # AKA embedding_dim instead of columns if project_to_embedding_dim is True
        return self.final_linear_weights(combined_results), cache
