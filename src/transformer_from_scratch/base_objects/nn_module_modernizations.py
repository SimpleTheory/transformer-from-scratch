from nn_modules import *


class MoEDoubleLinearApplied(torch.nn.Module):
    # TODO: This MoE FF is ungated it is currently upscale then activate then downscale, SwiGated is closer to:
    # up = upscale(input)
    # gate = activation_func(gate_layer(input))
    # result = downscale(gate * up) # Elementwise multiplication
    # TODO: This MoE needs a load balancing loss to make sure the inputs don't all go to the same expert

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
