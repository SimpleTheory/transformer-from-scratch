import typing
import torch
import transformer_from_scratch.base_objects.autograd_functions as autograd_functions
from transformer_from_scratch.base_objects.utility import create_weights, create_biases
import math

"""
Notes:
    - Don't use @dataclass for nn.Modules (while learning pytorch) because order matters when super init is called
    and the attributes are defined
"""

class LinearLayer(torch.nn.Module):
    """
        Layer wx+b where
        x (..., in_features)
        w (out_features, in_features)
        b (out_features,)

        Where
            `in features` should be the trailing dimension of the input of this layer

            `out features` should be the amount of columns you want to add or remove from the matrix in the pipeline (after the transformation)

        The bias `b` can be disabled to just have a `wx` by setting it to None
    """
    def __init__(self, weights: torch.Tensor, biases: torch.Tensor | None):
        """
        :param weights: Must be (out_features, in_features)
        :param biases: (out_features,) or None
        """
        super().__init__()
        self.weights = weights
        self.biases = biases

    @classmethod
    def from_feature_counts(
            cls, in_features: int,
            out_features: int,
            activation: str | typing.Callable = None,
            bias: bool = True,
            initialization_scaling: float = None,
            dtype=None,
            **kwargs,
    ):
        # Layer initialization defaults to `1/sqrt(in_features)`
        # Layer initialization because randn's std is too big leading to big gradients and inefficient learning
        # Generic layer init is torch.randn(...) / sqrt(in_features)
        if initialization_scaling is not None:
            activation = 'static'
        return cls(
            weights=create_weights(in_features, out_features, activation, initialization_scaling, dtype, **kwargs),
            biases=create_biases(out_features, dtype) if bias else None,
        )

    def forward(self, inputs: torch.Tensor):
        return autograd_functions.wx_plus_b_with_kwarg(inputs, self.weights, self.biases)

    @property
    def has_bias(self) -> bool:
        return self.biases is not None

class DoubleLinearApplied(torch.nn.Module):
    def __init__(
            self,
            in_columns: int,
            intermediate_columns: int,
            out_columns: int = None,
            activation_func: torch.autograd.Function = autograd_functions.gelu,
            initialization_scaling: float = None,
            dtype=None
    ):
        """
        A macro for a double linear layer with an activation function. The idea is to blow up the hidden space to let the
        model make better decisions about the results of attention then shrink it back down to the original size. In
        practice, it would look something like:

        (Batch, Sequence Len, Embedding) -> (Batch, Sequence Len, Embedding * 4) -> (Batch, Sequence Len, Embedding)
        """

        # <editor-fold desc="Attribution">
        super().__init__()
        if out_columns is None:
            out_columns = in_columns
        self.activation_func: torch.autograd.Function = activation_func.apply \
            if isinstance(activation_func, torch.autograd.Function) else activation_func
        self.initialization_scaling = initialization_scaling
        # </editor-fold>

        # Using kaiming-he scaling because activation function will likely be relu or gelu
        self.up = LinearLayer.from_feature_counts(
            in_columns,
            intermediate_columns,
            activation=activation_func,
            initialization_scaling=initialization_scaling,
            dtype=dtype
        )

        # Regular init scaling here because no activation function after
        self.down = LinearLayer.from_feature_counts(
            intermediate_columns,
            out_columns,
            dtype=dtype
        )

    def forward(self, inputs: torch.Tensor):
        """
        1st: Upcast the inputs
        2nd: Apply the activation function on the result
        3rd: Downcast the activated result and return that
        """
        return self.down(self.activation_func(self.up(inputs)))

class LayerNorm(torch.nn.Module):
    def __init__(self, trailing_dim_of_input: int, dtype=None):
        super().__init__()
        # At the beginning it should normalize without scaling so setting the weights to one and the biases to 0 will
        # allow it to at first normalize and then learn to reapply the scale
        self.weights = torch.nn.Parameter(torch.ones(trailing_dim_of_input, dtype=dtype))
        self.biases = torch.nn.Parameter(torch.zeros(trailing_dim_of_input, dtype=dtype))

    def forward(self, inputs):
        return autograd_functions.layer_normalization.apply(inputs, self.weights, self.biases)

class InvertedDropout(torch.nn.Module):
    """
    Dropout takes a tensor and zeros-out a given probability of values therein, but only during training.

    Notwithstanding why the issue is that will reduce the total scale of the tensor so say you have [1, 1, 1] -> [1, 1, 0]
    the latter is smaller on average by 1 - the zero-out probability.
    There are two ways to fix this:
    Traditionally at inference/validation when there is no dropout people would apply tenser * (1 - zero-out probability)
    on whatever tensors the dropout was applied on in training. This however leads to more work at inference time when the
    model is deployed.
    Nowadays people scale up the output of the dropout by / (1 - zero-out probability). That way the inference model is
    faster and the dropout is self-contained in training. Hence the name "Inverted" Dropout which is what this implementation does.

    The theory behind dropout is that networks can overtrain/overfit to specific values in a tensor which can have cascading
    effects on every layer thereafter. By zeroing out random values you make it so the model is not guaranteed to see any
    specific value or a collection of any specific values in any meaningful way. As such, you are "forcing" the model to
    distribute what it uses across a meaningful number of values in the Tensor. Essentially you are forcing it to infer
    what something is, by forcing it to fill in the blanks when not all the information is provided in training. Kind of
    like someone listening to a conversation in a very noisy area. They understand what is being said not because they hear
    all the phonemes, but they hear some of them and fill in the blanks with context. There is no context here necessarily,
    it just has to fill in the blanks based on what it did "hear" but that still works well to reduce overfitting in training.

    Because of the above dropout is usually applied in big hidden spaces or pivotal points of the model where the model can
    use specific values to overfit. It is not usually applied when that is not the case. For instance in a transformer it
    may be applied after, embedding, attention and feed forward, since these layers fit those criteria. But it will probabily
    not be applied to layer-norms or just any random linear layer.
    """
    def __init__(self, probability: float = 0.1):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("Dropout probability must be in [0, 1).")
        self.probability = probability

    @property
    def keep_probability(self):
        return 1 - self.probability

    def forward(self, inputs):
        # Only apply dropout in training
        if not self.training or self.probability == 0:
            return inputs
        # Create mask of which values to randomly zero-out
        mask = torch.rand_like(inputs) > self.probability
        # Dividing by `1 - probability` re-applies the scale that was lost from randomly zeroing out a bunch of the values of the input.
        # The scale matters because it changes the output when it is larger at inference as well as several functions in the model,
        # namely activation functions.

        # This implementation scales the outputs from the training to be larger. But it used to be that instead people
        # scaled the outputs from validation/inference to be smaller by doing `inference_output = input * (1 - probability)`
        # (at least wherever the dropout was applied).
        # This is why this implementation is called inverted dropout. Since dropout used to be scaled down at inference,
        # but now is scaled up at training. Hence, `InvertedDropout` vs how it used to be.

        # However scaling up the training output is slightly more efficient because it means there is less work for the
        # model when deployed doing inference. We are essentially shifting that work to the training for the production
        # inference model to be slightly faster and smaller.

        # Apply mask and scale up by what was lost on average through the dropout operation to keep scale.
        return inputs * mask / (1 - self.probability)

class EmbeddingLayer(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dimensions: int, initializer=.02, dtype=None):
        super().__init__()
        # Scale the matrix by an arbitrary scalar for the randomized weights to be lower for more stable gradients (gpt recommends .02)
        # Maybe parametrize it
        self.embedding_matrix = torch.nn.Parameter(torch.randn(vocab_size, embedding_dimensions) * initializer).to(dtype=dtype)

    def forward(self, tokens):
        """
        For the tokens, each element is a token ID and each row is one sequence.
        Creates a new tensor where each corresponding token id is replaced with its embedding vector.

        :param tokens: (batch_size, sequence_length)
        !TOKENS MUST BE AN INT TENSOR! usually dtype=torch.long (otherwise the indexing won't work)
        :return: (batch_size, sequence_length, embedding_dimensions)
        """
        return autograd_functions.embedding_function.apply(tokens, self.embedding_matrix)

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
