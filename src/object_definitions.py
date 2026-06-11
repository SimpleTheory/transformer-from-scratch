import torch
from torch.autograd.function import FunctionCtx

"""
(ChatGPT's Syllabus Instructions)
If I were doing this project myself, I'd implement the Functions in roughly this order:

LinearFunction (wx_plus_b)
ReLUFunction
SoftmaxFunction
CrossEntropyFunction
LayerNormFunction
EmbeddingFunction

Then build Modules:

Linear
LayerNorm
Embedding
FeedForward
MultiHeadAttention
TransformerBlock
GPT
"""

# When done write a doc string on how everything works together
# TODO: Write doc strings on the concept of the shapes of gradients relative a functions outputs and inputs


class relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input_tensor: torch.Tensor):
        ctx.save_for_backward(input_tensor)
        return torch.where(input_tensor > 0, input_tensor, 0)

    @staticmethod
    def backward(ctx: FunctionCtx, output_gradients):
        input_tensor, = ctx.saved_tensors
        # Where the argument is greater than 0 it will return true and 0 or less will return false
        # .to (arg_dtype is going to be some float) will convert the bools into float 1.0s and 0.0s
        # Thus the elements that are over 0 will be preserved (element * 1.0) when multiplied by the output gradients
        # and those below 0 will be canceled through * 0.0
        return output_gradients * (input_tensor > 0).to(input_tensor.dtype)

class wx_plus_b(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor):
        """
        :param ctx: Pytorch Context
        :param inputs: Tensor(batch_size, in_features)
        :param weights: Tensor(out_features, in_features)
        :param biases: Tensor(out_features,)
        :return: Tensor(batch_size, out_features)
        """
        ctx.save_for_backward(inputs, weights, biases)
        # (b,i) @ (i, o) + (o,)
        return (inputs @ weights.T) + biases

    @staticmethod
    def backward(ctx, output_gradients):
        """
        This is one of the most important insights in backprop.
        The shape of grad_output is always:
        Exactly the same shape as the output of the forward pass.

        :param ctx: Pytorch Context
            weights: Tensor(out_features, in_features)
            inputs:  Tensor(batch_size, in_features)
            biases:  Tensor(out_features,)
        :param output_gradients: Tensor(batch_size, out_features)
        :return:
        """

        inputs, weights, biases = ctx.saved_tensors

        # (o, i) = (b, o).T (b,i)
        grad_weights = output_gradients.T @ inputs
        # (b, i) = (b, o) (o, i)
        grad_inputs = output_gradients @ weights
        # (b, o) sum across all o's leaving only (o,)
        # When you condense on a dimension you are left with that dimension
            # Ex: (3, 2)
            #  [2, 5],
            #  [3, 7],
            #  [4, 1]
            # Condensed by column (which is 2)
            # 2+3+4, 5+7+1 = (9,13) ---> (,2)

        # Basically since you run the operation across different samples you have to condense the total gradient of each
        # sample into an "overall" gradient for the batch. The gradient for any given operation would still be that element's gradient * 1
        # but across the whole batch it is the sum of each element's gradient in the batch

        # Any time there is a broadcast addition this would be the derivative
        grad_biases = output_gradients.sum(dim=0)

        return grad_inputs, grad_weights, grad_biases

class softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        # TODO: List matrix sizes
        # TODO: Subtract by max in tensor to prevent overflow since softmax cares about relative differences
        # [e**current / sum([e**i for i in list_of_num]) for current in list_of_num]
        # Scalar operation being applied to the whole tensor
        input_tensor_with_each_element_to_the_e = torch.e ** input_tensor

        # Below each element to the e is now being divided by the sum of its row
        # (In the divisor) For each row, sum across the columns, and keep the result as a column-shaped tensor.
        # dim -1 is the dimension being summed across (you are summing across the columns here, thus condensing the rows)
        # Without keepdim it would flatten the values to one row, keepdim keeps each row here but with only its condensed value left.
        result = input_tensor_with_each_element_to_the_e / input_tensor_with_each_element_to_the_e.sum(dim=-1, keepdim=True)

        # Save the result because it is needed because softmax's derivative implementation in pytorch is wierd
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, output_gradients):
        # TODO: List matrix sizes
        # I kind of understand the derivative here with the jacobian matrix across the values relative to each other
        # since they affect each other. But I don't fully understand what is going on and how this secondary application
        # works for the chain rule. Though some of the fundamental concepts like multiplying the out gradient across the inputs
        # and chaining it on the output makes some conceptual sense.
        softmax_output, = ctx.saved_tensors
        return softmax_output * (
                output_gradients
                # Summed across columns (thus condensing each row into one value), keeping each row as its own row
                - (output_gradients * softmax_output).sum(dim=-1, keepdim=True)
        )

class layer_normalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input_tensor: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor, tiny_num_to_avoid_dev_by_0: float = 1e-5):
        """
        ((element - mean) / std_dev) * weight + bias

        :param input_tensor: (batch_size, features, feature_count)
        :param weights: (feature_count,)
        :param biases: (feature_count,)
        :param tiny_num_to_avoid_dev_by_0: Constant
        :return: (batch_size, features, feature_count)
        """
        elements_minus_mean = input_tensor - input_tensor.mean(dim=-1, keepdim=True)
        # Unbiased for variance changes equation to have a + 1 because of sample size vs true population.
        std_dev = torch.sqrt(input_tensor.var(dim=-1, keepdim=True, unbiased=False) + tiny_num_to_avoid_dev_by_0)
        normalized_tensor = elements_minus_mean / std_dev
        ctx.save_for_backward(normalized_tensor, std_dev, weights,)
        # * Because the weights aren't interconnected it's a broadcast
        return normalized_tensor * weights + biases

    @staticmethod
    def backward(ctx, output_gradients):
        """
        original input: (batch_size, features, feature_count)

        std_dev: (batch_size, features, 1) (One std_dev value per feature)

        weights: (feature_count,) (Weights broadcast per feature)
        :param output_gradients: same as original input
        :return:
        """
        normalized_tensor, std_dev, weights, = ctx.saved_tensors

        # * Here because it is * (a broadcast) in the forward pass
        gradient_normalized_tensor = weights * output_gradients

        # I don't fully understand this derivative but the basic idea is that a change in x affects three things:
            # 1. itself (in x - x_mean)
            # 2. The mean (in ...)
            # 3. The std_dev (sqrt(x_var))
        # So the gradient is essentially computing what a small change in the input would do to those 3 parts of the eq
        gradient_input = (
                # This part computes dx/dx_mean
                gradient_normalized_tensor - gradient_normalized_tensor.mean(dim=-1, keepdim=True)
                # This part computes the scale that was removed in the original function by using the gradient of the output
                # and the output to reapply the scale. The mean is then used for the scale removed from each element to be reapplied,
                # because the scale is reapplied on a feature wide level, not per element.
                - normalized_tensor * (gradient_normalized_tensor * normalized_tensor).mean(dim=-1, keepdim=True)
                # Rescale by std_dev
                ) / std_dev
        # Sum is to get the sum of all the gradients for each element across batch
        # (0, 1) means to sum across all batches and all rows: ex for sum every row[0] for over every batch to condense to a single value
        # That will leave only one value the gradient sum across every row, leaving only columns
        gradient_weights = (normalized_tensor * output_gradients).sum(dim=(0, 1))
        # One bias value is shared across all batch/feature positions, so all those gradient contributions accumulate into that one bias gradient.
        gradient_bias = output_gradients.sum(dim=(0, 1))
        # Have to include a None return because of tiny_num, though it is not being trained
        return gradient_input, gradient_weights, gradient_bias, None

class embedding_functions(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids, embedding_tensor):
        """

        :param token_ids: (batch_size, sequence_length) Each column contains a token ID
        :param embedding_tensor: (vocab_size, embedding_dimensions)
        :return: (batch_size, sequence_length, embedding_dimensions) Every scalar in the sequence length is replaced
        one for one with its corresponding embedding vector.
        """
        ctx.save_for_backward(token_ids)
        ctx.embedding_shape = embedding_tensor.shape
        # Indexes everything in the last dimension of token_ids (the scalar token ids) and replaces them with the
        # corresponding embedding dimensions in its index in vocab size
        return embedding_tensor[token_ids]

    @staticmethod
    def backward(ctx, output_gradients: torch.Tensor):
        """

        token_ids: (batch_size, sequence_length) Each column contains a token ID

        ctx.embedding_shape: tuple with numbers representing (vocab_size, embedding_dimensions)

        :param output_gradients: (batch_size, sequence_length, embedding_dimensions)
                    for example with (2,3,4):
                        2 batches × (at) 3 token positions × (have) 4 gradient values
        :return:
        """
        tokens_ids = ctx.saved_tensors
        embedding_gradients = torch.zeros_like(ctx.embedding_shape, device=output_gradients.device, dtype=output_gradients.dtype)

        # Args:
            # 1. Which dimension to add into, here add into the rows of embedding gradient (basically into the vocab size
                # because each value in the embedding dimension will get its own gradient)
            # 2. Index of which rows should be added into and their order,
                # basically a list of tokens in order for what their gradient should be
            # 3. The gradients that should be added into each value of the embedding dimension.
                # Basically we take the values in the order that they should have been given in the forward pass and those gradients
                # will then be added back into the corresponding vocab's gradient given in arg 2.
                    # Reshape here is reshaping to (batch_size * sequence_length) essentially flattening that part which is the order
                    # in which forward looked up each token
        embedding_gradients.index_add(0, tokens_ids.reshape(-1), output_gradients.reshape(-1, output_gradients.shape[-1]))

        # The tokens don't have a gradient since the model isn't changing them (at least in this architecture)
        return None, embedding_gradients
        
        