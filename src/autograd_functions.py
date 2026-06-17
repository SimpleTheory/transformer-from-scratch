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

# TODO: Write doc strings on the concept of the shapes of gradients relative a functions outputs and inputs (maybe in readme)


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
        ... = all the leading dimensions

        Batch size doesn't matter because you are still multiplying them all by the same Weight matrix anyways
        The matrix multiplication is batched so for every one in the batch it will do this same operation.

        :param inputs: Tensor(..., in_features)
        :param weights: Tensor(out_features, in_features)
        :param biases: Tensor(out_features,)
        :return: Tensor(..., out_features)
        """
        ctx.save_for_backward(inputs, weights, biases)
        # (b,i) @ (i, o) + (o,)
        return (inputs @ weights.T) + biases

    @staticmethod
    def backward(ctx, output_gradients):
        """
        "This is one of the most important insights in backprop.
        The shape of grad_output is always:
        Exactly the same shape as the output of the forward pass." (ChatGPT)

        Basically:
            the shape of the return of forward == the backwards output gradient
            `forward(input).shape == output_gradient.shape`

            &

            the shape of the inputs (and their order in pytorch) == the shape of the return of the tensors in backward (and their order)
            `[input_1.shape, input_2.shape, ...] == [tensor.shape for tensor in backward(ctx, output_gradient)]`

        :param ctx: Pytorch Context
            weights: Tensor(out_features, in_features)
            inputs:  Tensor(batch_size, in_features)
            biases:  Tensor(out_features,)
        :param output_gradients: Tensor(batch_size, out_features) Same dimensions as return in forward

        :return: input_gradients: Tensor(batch_size, in_features) Same dimensions as inputs in forward
                 weight_gradients: Tensor(out_features, in_features)
                 bias_gradients: Tensor(out_features,)
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
    def forward(ctx, input_tensor: torch.Tensor):
        """
        The softmax formula is:
            [e**current / sum([e**i for i in list_of_num]) for current in list_of_num]

        :param input_tensor: Any tensor
        :return: Same shape with softmax applied across each row
        """
        # Shift the input by the maximum value to prevent overflow and imprecision from floats. Basically, `e**input` can be very large
        # and softmax only cares about relative magnitude. So shifting everything over equally prevents the overflow but
        # keeps the output the same. If there are large negative numbers its ok exponentiating by negatives gives you
        # a small number.
        shifted = input_tensor - input_tensor.max(dim=-1, keepdim=True).values

        # Scalar operation being applied to the whole tensor
        # input_tensor_with_each_element_to_the_e = torch.e ** input_tensor

        # This line is the same as the line above however I am using `.exp()` because the implementation of this is faster
        # and more stable due to more diligent work on the float value of the e constant in different circumstance.
        input_tensor_with_each_element_to_the_e = shifted.exp()

        # Below each element to the e is now being divided by the sum of its row

        # (In the divisor) For each row, sum across the columns, and keep the result as a column-shaped tensor.
        # `dim -1` is the dimension being summed across (you are summing across the columns here, thus condensing the rows into a single summed value).
        # Without `keepdim` it would flatten the values to one row, `keepdim` keeps each row here but with only its condensed value left.
        result = input_tensor_with_each_element_to_the_e / input_tensor_with_each_element_to_the_e.sum(dim=-1, keepdim=True)

        # Save the result because it is needed because softmax's derivative implementation in pytorch is wierd
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, output_gradients):
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
    def forward(
            ctx: FunctionCtx,
            input_tensor: torch.Tensor,
            weights: torch.Tensor,
            biases: torch.Tensor,
            tiny_num_to_avoid_dev_by_0: float = 1e-5):
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
       Backprop through normalization:
            1. Start with the gradient after weight scaling.
            2. Remove the part caused by the mean, since changing one element changes the row mean.
            3. Remove the part caused by the variance/std, since changing one element changes the row scale.
            4. Divide by std_dev because the forward pass divided by std_dev.

        input_tensor: (batch_size, features, feature_count)
        std_dev: (batch_size, features, 1) (One std_dev value per feature/row)
        weights: (feature_count,) (Weights broadcast per feature)
        :param output_gradients: (batch_size, features, feature_count)
        :return:
        """
        normalized_tensor, std_dev, weights, = ctx.saved_tensors

        # 1. * Here because it is * (a broadcast) in the forward pass
        gradient_normalized_tensor = weights * output_gradients

        gradient_input = (
                # 2. Remove the part caused by the mean, since changing one element changes the row mean.
                gradient_normalized_tensor - gradient_normalized_tensor.mean(dim=-1, keepdim=True)
                # 3. This part computes the scale that was removed in the original function by using the gradient of the output
                # and the output to reapply the scale. The mean is then used for the scale removed from each element to be reapplied,
                # because the scale is reapplied on a feature wide level, not per element.
                - normalized_tensor * (gradient_normalized_tensor * normalized_tensor).mean(dim=-1, keepdim=True)
                #  4. Rescale by std_dev
                ) / std_dev

        # This creates a tuple such that, if
            # output_gradients.shape == (2, 5, 4), then
            # dims_to_sum == (0, 1)
        # Basically creates a tuple of the dims to sum over (all of them except the last one as seen above)
        dims_to_sum = tuple(range(output_gradients.ndim - 1))

        # Sum is to get the sum of all the gradients for each element across each batch
        # This will sum everything (including batches, rows, etc.) except the columns basically leaving one value per column
        gradient_weights = (normalized_tensor * output_gradients).sum(dim=dims_to_sum)
        # One bias value is shared across all batch/feature positions, so all those gradient contributions accumulate into that one bias gradient.
        gradient_bias = output_gradients.sum(dim=dims_to_sum)
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
        embedding_gradients = torch.zeros(ctx.embedding_shape, device=output_gradients.device, dtype=output_gradients.dtype)

        # Args:
            # 1. Which dimension to add into, here add into the rows (basically into the vocab size
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

class cross_entropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probabilities, targets):
        """
        Basically what's happening in the formula everything is going to cancel out except for the target ()
        then you take the natural log of the target's probability * -1 (The negative is because a log of decimals gives
        negative numbers, and you need the loss to be bigger than 0 (since 0 is the "ideal" output), basically the same
        as doing absolute value though).

        tldr: sum(-target * ln(probability)) (where target on everything but 1 element is 0)
        tldrest: -ln(correct_probability)

        :param ctx:
        :param probabilities: (Batch, Classes) Output probabilities from the model
        :param targets: (Batch, Classes) Tensor with 0s for everything except 1 on target
        :return: (Batch,) One loss score per batch
        """
        ctx.save_for_backward(probabilities, targets)
        return -(targets * probabilities.log()).sum(dim=-1).mean()

    @staticmethod
    def backward(ctx, output_gradients):
        """
        Add an extra dimension to broadcast to each element: (batch,) unsqueeze -> (batch, 1)
        Original function (target always 1 or 0) is:
            -target * ln(probability)
        Therefore given the fact that the derivative of ln(x) == 1/x the derivative with respect to probability is:
            -target * 1/probability
        Then apply chain rule to each element (broadcast):
            output_gradient (for that batch) * -target/probability

        One note this implementation is numerically unstable because an element of probability can be 0
        That (and the fact that it makes the derivative way easier) is why it's usually wrapped in softmax

        probabilities: (Batch, Classes)
        targets: (Batch, Classes)
        :param output_gradients: (Batch,)
        :return: gradient_probabilities (Batch, Class), None (Targets don't need gradients they are the expected value)
        """
        probabilities, targets = ctx.saved_tensors
        gradient_probabilities = output_gradients.unsqueeze(-1) * (-targets / probabilities)

        return gradient_probabilities, None

class softmaxed_cross_entropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, results_of_model_tensor, targets):
        """
        cross_entropy over a sample's total outputs is (because only one target is active):
            `sum(-target * ln(probability))`
        softmax makes each probability:
            `e**current / sum([e**i for i in list_of_num]`
        therefore if you apply softmax to the probabilities beforehand you get:
           `sum(-target * (ln(e**current / sum([e**i for i in list_of_num])))`
        simplify with log rule `log(a/b) == log(a) - log(b)`:
            `sum(-target * (ln(e**current) - ln(sum([e**i for i in list_of_num]))))`
        and again:
            `sum(-target * (current - ln(sum([e**i for i in list_of_num]))))`
        multiply target into the parenthesis
            `sum(-target * current - (-target * ln(sum([e**i for i in list_of_num]))))`
        split with the idea that `sum(a+b) == sum(a) + sum(b)`
        it is also + because you are doing `- -target * ...` which is a double minus
            `sum(-target * current) + sum(target * ln(sum([e**i for i in list_of_num]))`
        because the ln term is constant with respect to target meaning it is always the same number no matter what the target
        is we can factor it out of the sum (basically ab + cb == b(a+c))
            `sum(-target * current) + sum(target) * ln(sum([e**i for i in list_of_num]))`
        we know that the list of targets is all 0s except for 1, `so sum(target) == 1`:
            `sum(-target * current) + 1 * ln(sum([e**i for i in list_of_num]))`
        and since only 1 target has 1 and the rest are 0s `sum(-target * current) == -current`:
            `-correct_class + ln(sum([e**i for i in list_of_num]))`
        aka
            `ln(sum([e**i for i in list_of_num])) - correct class' value`
        :param results_of_model_tensor: (batch_size, sequence_length, vocab_size)
        :param targets: (batch_size, sequence_length)
        :return: Scalar mean of loss per token per batch
        """
        # Shift to prevent overflow
        max_values = results_of_model_tensor.max(dim=-1, keepdim=True).values
        shifted = results_of_model_tensor - max_values

        # Find the correct class' value
        # gather has 2 arguments
            # 1. the dimension to index (index across the columns (basically indexing columns in a row))
            # 2. a list of indices to get from (each row)
                # Unsqueeze turns the list into a column so you get one value to index per row
                # and each row of the list corresponds to one of the source
            # Then returns a new matrix with equal rows (1st arg) and one column where each value is the one indexed from the index column (2nd arg)
        list_of_correct_values_per_batch = results_of_model_tensor.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # Sum dim=-1 condenses across the columns leaving one value for each row
        # This is a constant subtraction (1 constant across each batch, so for the batch it is a broadcast)
        # For this implementation you have to readd the shifted's magnitude to each number to get the true result
        # The softmax will handle the shifted the same regardless, but you want to readd the max value for the loss
        loss_per_token = torch.log(torch.exp(shifted).sum(dim=-1)) + max_values.squeeze(-1) - list_of_correct_values_per_batch

        ctx.save_for_backward(shifted, targets)

        # Mean loss for the whole batch
        return loss_per_token.mean()

    @staticmethod
    def backward(ctx, output_gradients):
        """
        shifted: (batch_size, sequence_length, vocab_size)
        targets: (batch_size, sequence_length)

        :param output_gradients: Scalar Value 1.0 (unless I add more stuff upstream like a loss modifier or a combo loss,
        then it would inherit the gradient relative how it'd affect the loss mod/combo loss)
        :return: same as shifted, None (Same as inputs from forward)
        """
        shifted, targets, = ctx.saved_tensors
        # See softmax's forward
        shifted_exp = shifted.exp()
        shifted_exp_sum = shifted_exp.sum(dim=-1, keepdim=True)
        probabilities = shifted_exp / shifted_exp_sum
        # Unsqueeze the targets so instead of a list of targets you get one column of targets (one per row)
        index = targets.unsqueeze(-1)
        # (arg src) Make a -1 tensor with the same shape as the index
        # What the index and dim are telling you here is for each row on which column are we going to add that specific row's -1 (from src)
        # since dim=columns here its for each row determine the column(s) being added to and you add the equivalent value from src (since index and src have the same dim)
        # in practice this indexes the target for each row and does -1 on its probability
        # The extra _ after `...add_` is pytorch convention for an inplace operation
        # Basically: probability -= target
        probabilities.scatter_add_(
            dim=-1,
            index=index,
            src=-torch.ones_like(index, dtype=shifted.dtype)
        )
        # The derivative of mean (cf forward pass `return loss_per_token.mean()`) of any element used in the mean is that
        # 1 / total num of elements
        # For example, original equation: (x1 + x2 + x3 + x4) / 4
        # If you change any given x a little bit the outcome will be (the change)/4 (ie change by one -> output change 1/4)
        # When you apply the derivative to each element it ends up being: probabilities / total num of tokens
        # We can get total token count with targets because there every token is accounted for with a 0 or 1
        probabilities /= targets.numel()

        # Chain rule scalar multiplication
        probabilities *= output_gradients

        # probabilities should be renamed after scatter_add_ to gradient_probabilities since that - operation is literally the
        # gradient (at least before applying the derivative of the mean)

        # Targets has no gradient so this is the return
        return probabilities, None

