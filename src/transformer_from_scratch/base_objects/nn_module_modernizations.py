import typing

import torch.nn

from nn_modules import *
import autograd_functions_modernization


class GatedFFN(torch.nn.Module):
    """
    Gated Feed Forward is basically
        up_scale = input @ up_weights
        gate = activation_func(input @ gate_weights)
        intermediate = up_scale * gate
        result = intermediate @ down_weights

    Where as before it was (regular FFN) was:
        intermediate = activation_func(input @ up_weights)
        result = intermediate @ down_weights

    The difference being instead of having the up_weights be responsible for:
        • What should be active
        • and what it should learn

    Those responsibilities are now separated into distinct layer that way:
        up_weights - can focus on what it should learn
        gate_weights - can focus on what it should expose to the next layer

    Separating these responsibilities into distinct layers makes each layer better at its specialization and has shown
    better performance in studies when compared to the original FFN.


    -------------
    Some Extra Notes:
        • Not all implementations have biases because in very large transformers with large datasets the offset are already learned
          through all the combinations of all the weights making the biases superfluous. Roughly in the range of 100Ms of parameters
          approaching 1B and 100s of GBs of training data approaching TBs, though there is no real scientific cut-off
          these numbers are just speculation. If one were to start dropping biases, this module would be a good candidate.
        • Not all implementations have activation functions since the gate weights can just learn to diminish certain features
          without the activation function. This approach is called a bi-linear layer, but it also has the side effect
          of it not merely being gating signal but also a multiplier. Which may cause over reliance on features as it
          multiplies them or soft-kills features while reducing their relevance relative the multiplied features.

    """
    def __init__(
            self,
            in_columns: int,
            # This can be a lot of different possible numbers, 8/3 is somewhat of a good starting place for now
            intermediate_columns: int | float = 8/3,
            intermediate_columns_as_multiplier: bool = True,
            # Default should be same as in columns
            out_columns: int = None,
            # Set to None to make it a bilinear layer
            activation_func:
                type[torch.autograd.Function] |
                typing.Callable[[torch.Tensor], torch.Tensor] |
                None
            = autograd_functions.silu,
            # set to 1.0 if you don't want init scaling: default is kaiming-he
            # generic_initialization_scaling: float = None,
            gate_initialization_scaling: float = None,
            bias=True
    ):
        """
        A macro for a triple linear layer where on acts as a gate with an activation function. The idea is to blow up the hidden space to let the
        model make better decisions about the results of attention. That hidden space is then gated and shrunk back down to its original size. 
        In practice, it would look something like:

        (Batch, Sequence Len, Embedding) ->   (Batch, Sequence Len, Embedding * 3) -> (Batch, Sequence Len, Embedding)
                                    activation(Batch, Sequence Len, Embedding * 3) -↗
        
        :param in_columns: INT number of columns in input tensor
        :param intermediate_columns: INT number of columns in intermediary space OR INT to multiply with in columns to determine the former 
        :param intermediate_columns_as_multiplier: if True sets the `intermediate_columns` param to be an `INT` to
         multiply with `in_columns` to determine the actual number of intermediate_columns
        :param out_columns: INT number of columns of the output tensor, if None it will be equal to `in_columns`
        :param activation_func: Activation func to use with the `gate layer`. If None it will be bilinear with the `up layer`
        # :param generic_initialization_scaling: Constant to scale random init weights for up and down.
        default if None is 1/sqrt(in_columns)
        :param gate_initialization_scaling: Constant to scale random init weights for gate.
        default if None is math.sqrt(2/in_columns)
        :param bias: Bool if False the layers will not have bias parameters, generally should only be False on bigger models 
        (1B+ Param, 1TB+ Dataset) default is True
        """
        super().__init__()

        out_columns = in_columns if out_columns is None else out_columns
        if intermediate_columns_as_multiplier:
            intermediate_columns = round(intermediate_columns * in_columns)

        # <editor-fold desc="Attribution">
        self.in_columns = in_columns
        self.out_columns = out_columns
        self.intermediate_columns = intermediate_columns
        self.intermediate_columns_as_multiplier = intermediate_columns_as_multiplier
        self.activation_func = self.get_activation_function(activation_func)
        # self.generic_initialization_scaling = generic_initialization_scaling if generic_initialization_scaling is not None else 1/math.sqrt(in_columns)
        # Using kaiming-he scaling because activation function will likely be relu or gelu or silu
        self.gate_initialization_scaling = gate_initialization_scaling if gate_initialization_scaling is not None else (math.sqrt(2/in_columns))
        self.bias = bias
        # </editor-fold>

        # Divided by sqrt(in_columns) for generic initialization, down_weights have this as well
        self.up_weights = torch.nn.Parameter(torch.randn(intermediate_columns, in_columns) / math.sqrt(in_columns))
        # Init scaling here because of activation function, though this can be changed by the parameters
        self.gate_weights = torch.nn.Parameter(torch.randn(intermediate_columns, in_columns) * self.gate_initialization_scaling)
        self.down_weights = torch.nn.Parameter(torch.randn(out_columns, intermediate_columns) / math.sqrt(intermediate_columns))

        # Bias initialization
        if bias:
            self.up_biases = torch.nn.Parameter(torch.zeros(intermediate_columns))
            self.gate_biases = torch.nn.Parameter(torch.zeros(intermediate_columns))
            self.down_biases = torch.nn.Parameter(torch.zeros(out_columns))

    def forward(self, input_tensor):
        """
        This is just the below but more verbose and with biases if they were added:
            up_scale = input @ up_weights
            gate = activation_func(input @ gate_weights)
            intermediate = up_scale * gate
            result = intermediate @ down_weights
        :param input_tensor: (..., in columns)
        :return: (..., out columns)
        """
        up_scale = autograd_functions.wx_plus_b_with_kwarg(input_tensor, self.up_weights, self.get_attr('up_biases'))
        gate = self.activation_func(
            autograd_functions.wx_plus_b_with_kwarg(input_tensor, self.gate_weights, self.get_attr('gate_biases'))
        )
        intermediate = up_scale * gate

        return autograd_functions.wx_plus_b_with_kwarg(intermediate, self.down_weights, self.get_attr('down_biases'))

    @staticmethod
    def get_activation_function(activation) -> typing.Callable[[torch.Tensor], torch.Tensor]:
        # No activation -> identity function
        if activation is None:
            return lambda x: x

        # Custom torch.autograd.Function -> use .apply()
        if isinstance(activation, type) and issubclass(activation, torch.autograd.Function):
            return activation.apply

        # Regular function, nn.Module instance, etc.
        if callable(activation):
            return activation

        raise TypeError(f"Activation must be callable, a torch.autograd.Function subclass, or None. Got {type(activation)}.")

    def get_attr(self, attr, default=None):
        return getattr(self, attr, default)


class RMSNorm(torch.nn.Module):
    """
    To see the details about RMSNorm please refer to `autograd_functions_modernization.rms_norm`
    """
    def __init__(self, trailing_dim_of_input: int, tiny_num_to_avoid_dev_by_0=1e-5):
        super().__init__()
        self.trailing_dim_of_input = trailing_dim_of_input
        self.tiny_num_to_avoid_dev_by_0 = tiny_num_to_avoid_dev_by_0
        # The weight is initialized to 1 to start with no scale after the normalization by default.
        # From there it can learn the scale that it should reapply.
        self.weights = torch.nn.Parameter(torch.ones(trailing_dim_of_input))

    def forward(self, input_tensor):
        return autograd_functions_modernization.rms_norm.apply(input_tensor, self.weights, self.tiny_num_to_avoid_dev_by_0)


class MistralStyleMoEGatedFFN(torch.nn.Module):
    def __init__(
            self,
            total_number_of_experts: int,
            experts_to_accept: int,
            in_columns: int,
            intermediate_columns: int | float = 8 / 3,
            intermediate_columns_as_multiplier: bool = True,
            out_columns: int = None,  # Default should be same as in columns
            activation_func=autograd_functions.silu,
            bias: bool = True
    ):
        super().__init__()

        if experts_to_accept > total_number_of_experts:
            raise ValueError(f"{experts_to_accept=} cannot be greater than {total_number_of_experts=}")
        if total_number_of_experts < 1:
            raise ValueError(f'Your total number of experts {total_number_of_experts} must be greater than 1')
        if not 1 <= experts_to_accept <= total_number_of_experts:
            raise ValueError(f'This statement at MoE is False: 1 <= experts_to_accept <= total_number_of_experts\n'
                             f'1 <= {experts_to_accept} <= {total_number_of_experts}')

        self.total_number_of_experts = total_number_of_experts
        self.experts_to_accept = experts_to_accept
        # Bias is False here because a bias (in this topk MoE implementation) gives each expert a global preference independent of the token.
        self.gate = LinearLayer.from_feature_counts(in_columns, total_number_of_experts, bias=False)

        self.experts = torch.nn.ModuleList([
            GatedFFN(in_columns, intermediate_columns, intermediate_columns_as_multiplier, out_columns, activation_func, bias)
            for _ in range(total_number_of_experts)
        ])

    def __getattr__(self, item):
        err = AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")
        try:
            return super().__getattr__(item)
        except AttributeError:
            pass  # It's not a standard PyTorch attribute, look at the experts next
        if '_modules' in self.__dict__ and 'experts' in self._modules and len(self.experts) > 0:
            try:
                return getattr(self.experts[0], item)
            except IndexError as e:
                raise AttributeError(f'{item} was not found in {type(self).__name__}, however the experts list is'
                                     f' currently empty. If this was an expert attribute make sure to populate the'
                                     f' experts before accessing this attribute.') from e
            except AttributeError as e:
                raise err from e
        raise err

    @property
    def out_columns(self) -> int:
        return self.experts[0].out_columns

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param input_tensor: Batch Size, Sequence Length, Embedding Dimensions
        :return: Tuple(
            Tensor(Batch Size, Sequence Length, Out Columns),
            Scalar as 0 dimensional Tensor: (Auxiliary Loss)
        )
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
        )

        # Create an output buffer
        result = torch.zeros(n, self.out_columns, device=input_tensor.device, dtype=input_tensor.dtype)

        start = 0
        for current_expert, hits in enumerate(hits_per_expert.tolist()):
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

        # Calculate Auxiliary / Load Balancing Loss
        auxiliary_loss = self.load_balancing_loss(
            hits_per_expert,
            self.get_router_probabilities(scores),
        )

        # Reshape the buffer to (Batch Size, Sequence Length, Out Columns)
        return result.reshape(batch_size, sequence_length, self.out_columns), auxiliary_loss

    @staticmethod
    def get_router_probabilities(scores: torch.Tensor) -> torch.Tensor:
        """
        Softmax the scores to get the probability of each router for each token.
        Where every element is the score that token gets for that specific router. Hence, the var name `scores`.
        :param scores: (Total Tokens, Total Experts) The score the gate gives for each expert.
        :return: (...) Softmaxed for each token giving you a probability for each router at each token.
        """
        return autograd_functions.softmax_with_kwarg(
            scores,
            dim=-1,
        )

    def load_balancing_loss(
            self,
            hits_per_expert: torch.Tensor,
            router_probabilities: torch.Tensor,
            # coefficient: float = .01,
    ) -> torch.Tensor:
        # router_probabilities is (Total Tokens, Total Experts), thus
        total_token_count = router_probabilities.shape[0]
        # The amount of times an expert was selected because each token goes to the top-k experts,
        # hence: total_token_count * experts_to_accept
        total_times_an_expert_was_selected = total_token_count * self.experts_to_accept

        # This is a fraction for each expert re: the total amount of times it was selected / total selections
        ratios_per_expert = hits_per_expert / total_times_an_expert_was_selected

        # This is the average per router not per token take  for example
        #            R0    R1    R2    R3
        # Token 0: [0.70, 0.10, 0.10, 0.10]
        # Token 1: [0.50, 0.20, 0.20, 0.10]
        # Token 2: [0.60, 0.10, 0.20, 0.10]

        # This will give the average (across the rows aka) for each column giving a list of (R,) where each value is that R's average
        # This differs from the above in that this is the average (softmax) probability any given token had to go to a router
        # Additionally this is the differentiable path our loss will learn from: (E = total experts) (N = Sequence Length)
                # gate: weights & bias
                # ↓
                # scores = self.gate(flattened_input)(N, E)
                # ↓
                # router_probabilities = softmax(scores)(N, E)
                # ↓
                # average_router_probability = mean(dim=0)(E,)
        # The other var `ratios_per_expert` is undifferentiable because it is intrinsically linked to top-k which is discrete
        average_router_probability = router_probabilities.mean(dim=0)

        # The reason you multiply by the number of experts is `ratios_per_expert` (ideally) is 1/E and `avg_router_probability` is (ideally) 1/E as well.
        # Though, that correlates your load balancing loss on E so let's do a walk through given an ideal situation:

        # In an ideal setting
        # ratios = [0.25, 0.25, 0.25, 0.25]
        # probs = [0.25, 0.25, 0.25, 0.25]
        # Multiply them elementwise
        # [0.0625, 0.0625, 0.0625, 0.0625] aka [1/E**2]
        # Sum the list (because the list is size E when you sum it is effectively the same as condensing all the values into 1 and doing * e)
        # 0.25 aka (1/E)
        # However notice that the loss number achieved here is still intrinsically correlated with our total number of experts
        # Thus to make it independent we have to multiply by E.
        # Even though the final ideal loss ends up being 1 gradient descent looks for changes that make improvements
        # Since there are no improvements it will stay there.
        auxiliary_loss = self.total_number_of_experts * torch.sum(ratios_per_expert * average_router_probability)

        # Returns Scalar as 0 dimensional Tensor
        return auxiliary_loss  # * coefficient
