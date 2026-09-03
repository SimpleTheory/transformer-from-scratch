from autograd_functions import *

class rms_norm(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            input_tensor: torch.Tensor,
            weights: torch.Tensor,
            tiny_num_to_avoid_dev_by_0: float = 1e-5):
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
