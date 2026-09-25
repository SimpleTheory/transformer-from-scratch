import torch.nn

from transformer_from_scratch.base_objects.autograd_functions import *
from typing import NamedTuple

class rms_norm(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            input_tensor: torch.Tensor,
            weights: torch.Tensor,
            tiny_num_to_avoid_dev_by_0: float = 1e-6):
        """
        ---------------------------------------------
        RMSNorm
        element / sqrt(avg(x**2)) * weight
                • where avg is: sum(x**2) / num_elements

        LayerNorm
        ((element - mean) / std_dev) * weight + bias
                • where std_dev is sqrt(sum(x - mean)**2 / num_elements)

        Basically you take out the mean and the bias (relative to the original LayerNorm formula).
        The idea is that you don't need to recenter via the mean and as such since you are not offsetting the
        original vector you don't need a learned bias. Just remove the scale and re-add it with the weight.

        This has been shown to have the same performance as `LayerNorm` with fewer parameters. The idea being the important
        part of `LayerNorm` was just rescaling, and re-centering didn't actually help all that much.
        ---------------------------------------------

        :param input_tensor: (batch_size, rows, columns)
                         aka (batch size, seq len, embedding dim)
        :param weights: (columns,)
        :param tiny_num_to_avoid_dev_by_0: Constant
        :return: (batch_size, rows, columns)
        """
        # sum(x**2) / count(x)
        mean_square = input_tensor.pow(2).mean(dim=-1, keepdim=True)
        mean_square_sqrt_over_1 = torch.rsqrt(mean_square + tiny_num_to_avoid_dev_by_0)
        input_over_mean_sqrt = input_tensor * mean_square_sqrt_over_1
        ctx.save_for_backward(input_over_mean_sqrt, mean_square_sqrt_over_1, weights)
        return input_over_mean_sqrt * weights

    @staticmethod
    def backward(ctx, output_gradients):
        input_over_mean_sqrt, mean_square_sqrt_over_1, weights = ctx.saved_tensors
        # TODO: Calculate gradient of each parameter and return in order above!
        # In trying to get the gradient over the mean we kinda have to work backwards to get the gradient of each intermediary
        # step which would entail:
            # gradient of input_over_mean_sqrt,
            # gradient of the sum(input**2) / count(input)
            # gradient of input from the dividend,
        # We do a small workaround though instead of calculating the 2nd point we project it onto the original result and subtract
        # it from the gradient given in the 1st, then we use that result to calculate the 3rd point.

        # Gradient with respect to input_over_mean_sqrt
        grad_input_over_mean_sqrt = output_gradients * weights

        # Figure out how much of the gradient is trying to push the input in the
        # same direction it already points, which would only change its overall size.
        projection = (grad_input_over_mean_sqrt * input_over_mean_sqrt).mean(dim=-1, keepdim=True)

        # Turn that amount back into a vector pointing along the normalized input, then subtract it from the incoming gradient.
        # This removes the part of the gradient that only tries to make the entire input vector bigger or smaller,
        # since RMSNorm would normalize that change away. Essentially undoing `sqrt(avg(x**2))`
        # Essentially:
            # remaining_gradient = what_the_gradient_wants - vector_scaling_part
        resized_gradient = grad_input_over_mean_sqrt - input_over_mean_sqrt * projection

        # Finally account for the original division by RMS. Basically after the above calculation we can treat `mean_square_sqrt_over_1`
        # as a constant and get the gradient of the input by taking the calculated resized gradient we have so far and multiplying
        # by this.
        gradient_input = resized_gradient * mean_square_sqrt_over_1

        # Gradient with respect to the weights
        # Sum over every dimension, but leave the columns. So each column's value will be the sum over all the
        # row's at all the batches who have that column. The idea being the value of the weight is being multiplied to
        # each element of every row and every batch where that column is. So the total gradient of the weight is the sum
        # of each of those.
        dims_to_sum = tuple(range(output_gradients.ndim - 1))
        gradient_weights = (output_gradients * input_over_mean_sqrt).sum(dim=dims_to_sum)

        return gradient_input, gradient_weights, None

# <editor-fold desc="Rope">
class RopeParameters(torch.nn.Module):
    def __init__(self, cos: torch.Tensor, sin: torch.Tensor):
        super().__init__()

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def __iter__(self):
        yield self.cos
        yield self.sin

def compute_basic_rope_params(embedding_dimension: int, context_length: int, theta_base: int = 10000, dtype=torch.float32) -> RopeParameters:
    """
    Basically the formula is (broadcast)
        position / [θ**(i/embedding_dimension) for i in range(0, embedding_dim, 2)]
        -> [
              position / θ**(0/embedding_dimension),
              position / θ**(2/embedding_dimension),
              position / θ**(4/embedding_dimension),
              ...
            ]
    Before returning you duplicate it so:
        -> [
          position / θ**(0/embedding_dimension),
          ...,
          position / θ**(0/embedding_dimension),
          ...
        ]
    Then you return 2 copies one applying cos to each element and one applying sin

    :param embedding_dimension: The embedding dimensions of the head you are applying rope to (or at least the
    dimensions you are applying rope on in the case of partial rope) MUST BE EVEN!
    :param context_length: Total context length to get possible positional values.
    :param theta_base: What constant to use for theta in the equation above.
    :param dtype: What Dtype to use should stay at F32 for operational consistency, can be converted to something else later.
    :return: (cos, sin), each with shape (context_length, embedding_dimension).
    """
    if embedding_dimension % 2 != 0:
        raise ValueError('The RoPE dimension must be even because half of the dimensions are rotated against the other half.')

    # [10,000**(0/512), 10,000**(2/512), ... ] (embedding_dim/2,)
    inv_freq = 1.0 / (
            theta_base ** (
            torch.arange(0, embedding_dimension, 2, dtype=dtype).float()  # range(0, embedding_dim, 2)
            / embedding_dimension)
    )
    positions = torch.arange(context_length, dtype=dtype)  # range(context_length) (context_length,)

    # The idea is to be able to do broadcast multiplication as follows:

    # positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    #   -> (context_length, 1) * (1, embedding_dim/2)
    #   -> (context_length, embedding_dim/2)

    # Thus as an example:
    # positions.unsqueeze(1) =
    # [
    #     [0],
    #     [1],
    #     [2]
    # ]
    # inv_freq.unsqueeze(0) =
    # [
    #     [1.0, 0.1]
    # ]
    # and therefore:
    # angles =
    # [
    #     [0 * 1.0, 0 * 0.1],
    #     [1 * 1.0, 1 * 0.1],
    #     [2 * 1.0, 2 * 0.1]
    # ]
    # The reason the broadcast works is because of the unsqueezes. Reshaping the lists to have size 1 across different dimensions of a table
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)

    # Duplicate the angles so that (given our previous example)
    # angles = [
    #     [0.0, 0.0],        [0.0, 0.0, 0.0, 0.0],
    #     [1.0, 0.1],   ->   [1.0, 0.1, 1.0, 0.1],
    #     [2.0, 0.2]         [2.0, 0.2, 2.0, 0.2]
    # ]
    # Thus it is no longer (context_length, embedding_dim/2), but rather (context_length, embedding_dim)
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)  # Apply cos to every value in the tensor
    sin = torch.sin(angles)  # Apply sin to every value in the tensor

    return RopeParameters(cos, sin)

def apply_rope(input_tensor, cos, sin, offset=0):
    """
    The useful property of rope is that when the rotated query and key are used in the attention dot product:
        q' = R(mθ)q
        k' = R(nθ)k

        q' · k'
        = qᵀ R(mθ)ᵀ R(nθ) k
        = qᵀ R((n - m)θ) k

    Therefore, although Q and K are individually rotated using their
    absolute positions m and n, their attention score naturally depends
    on the RELATIVE position (n - m).

    As such the final result of the similarity score between vector Q & K will be adjusted insofar as the relative positions between
    K and M for each angle at each of the different embedding dimensions. That also means however that you are forcing embedding dimension
    pairs to learn that specific positional information as opposed to just any general information. This can have unintended
    consequences and is the reason why partial rope exists as well (to let some embedding dimensions learn independent of position).

    In the end with rope the model would get something like:
        content similarity
        +
        how that content aligns under distance 3
        at many different positional frequencies
    or “Distance 3 should mean the same positional relationship everywhere in the sequence, but how much that relationship matters depends on the actual Q/K content.”
    (at least allegedly)

    rotation is
         cosθ, -sinθ,    @        x1
         sinθ,  cosθ              x2
                   ↓
            x1cos - x2sin
            x1sin + x2cos

    :param input_tensor: Tensor to be rotated
    :param cos: the cos frequencies for this tensor
    :param sin: the sin frequencies for this tensor
    :param offset: What tokens to apply rope to, important for KV caching. Since if you already have 100 tokens computed
    and you get 4 more just do offset=100 and sequence_length=4. That way you'll get the result for tokens 100, 101, 102, and 103.
    :return:
    """
    batch_size, num_heads, sequence_length, head_dim = input_tensor.shape
    if head_dim % 2 != 0:
        raise ValueError('The RoPE dimension must be even because half of the dimensions are rotated against the other half.')
    x1, x2 = input_tensor.split(head_dim // 2, dim=-1)
    current_sequence_slice = slice(offset, offset + sequence_length)

    # Transform shape into:  (1, 1, sequence_length, head_dim)
    # this is done to broadcast it into each batch and each head
    cos = cos[current_sequence_slice, :].unsqueeze(0).unsqueeze(0)
    sin = sin[current_sequence_slice, :].unsqueeze(0).unsqueeze(0)

    # with input [a, b, c, d] -> [-c, -d, a, b]
    rotated = torch.cat((-x2, x1), dim=-1)

    # so with input [a, b, c, d] & rotated_input [-c, -d, a, b]
    # [
    #     a * cos - c * sin,
    #     b * cos - d * sin,
    #     c * cos + a * sin,
    #     d * cos + b * sin
    # ]
    # So this is equivalent to the rotation matrix where (a,c) and (b,d) are pairs
    result = (input_tensor * cos) + (rotated * sin)

    return result.to(dtype=input_tensor.dtype)

def apply_partial_rope(rope_dim: int, input_tensor, cos, sin, offset=0, return_separately=False):
    # Split into two tensors divided along the embedding dimension
    # The idea is to apply rope to one but not the other
    rope_part = input_tensor[..., :rope_dim]
    untouched_part = input_tensor[..., rope_dim:]

    rope_part = apply_rope(
        rope_part,
        cos,
        sin,
        offset
    )
    if return_separately:
        return rope_part, untouched_part
    return torch.cat((rope_part, untouched_part), dim=-1)
# </editor-fold>

class KVCache:
    def __init__(self, number_of_layers):
        self.number_of_layers = number_of_layers
        self.cache = [None] * number_of_layers

    def get(self, layer_index):
        return self.cache[layer_index]

    def update(self, layer_index, value):
        self.cache[layer_index] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for layer in range(len(self.cache)):
            self.cache[layer] = None

    def __getitem__(self, item):
        return self.get(item)

    def __setitem__(self, key, value):
        return self.update(key, value)
