import typing

import torch
import functools
import math

def upscale(
        dtype_to_cast: torch.dtype = torch.float32,
        original_dtype_arg_index: int = 0,
        args_indices_to_upcast: list[int] | None = None,
        kwarg_keys_to_upcast: list[str] | None = None,
        result_indices_to_downcast: list[int] | None = None,
):
    """
    :param dtype_to_cast: What dtype should be upscaled
    :param original_dtype_arg_index: index of argument to use as a basis for the original dtype (default 0)
    :param args_indices_to_upcast: iterable of indices to upcast (default is all float type tensors)
    :param kwarg_keys_to_upcast: iterable of keys to upcast their corresponding values (default is all float type tensors in kwargs)
    :param result_indices_to_downcast: index of results to downcast back to the original dtype
    leave a blank iterable (like tuple()) in order to return everything as upcasted.
    """
    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            original_dtype = args[original_dtype_arg_index].dtype
            args = list(args)

            # Default: upscale the tensor whose dtype determines the output dtype.
            if args_indices_to_upcast is None:
                indices = [
                    idx for idx, arg in enumerate(args)
                    if isinstance(arg, torch.Tensor) and arg.is_floating_point()
                ]
            else:
                indices = args_indices_to_upcast
            for index in indices:
                if isinstance(args[index], torch.Tensor) and args[index].is_floating_point():
                    args[index] = args[index].to(dtype_to_cast)

            if kwarg_keys_to_upcast is not None:
                for key in kwarg_keys_to_upcast:
                    value = kwargs[key]
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        kwargs[key] = value.to(dtype_to_cast)
            result = func(*args, **kwargs)

            def downcast(element, index=0):
                should_downcast = result_indices_to_downcast is None or index in result_indices_to_downcast
                if (
                    should_downcast
                    and isinstance(element, torch.Tensor)
                    and element.is_floating_point()
                ):
                    return element.to(original_dtype)
                return element

            if result_indices_to_downcast is not None and len(result_indices_to_downcast) == 0:
                return result
            if isinstance(result, tuple):
                return tuple(downcast(element, index) for index, element in enumerate(result))
            return downcast(result)

        return inner_wrapper
    return outer_wrapper

def upscale_autograd(
        compute_dtype: torch.dtype = torch.float32,
        args_to_upcast: tuple[int, ...] | None = None,
        outputs_to_downcast: tuple[int, ...] | None = None,
        output_dtype_from_arg: int = 0,
):
    """
    Wraps a torch.autograd.Function so selected forward inputs and backward
    gradients are computed at higher precision.

    args_to_upcast:
        Forward argument indices to upcast.
        None -> all floating-point tensor arguments.

    outputs_to_downcast:
        Output indices to restore to the original dtype.
        None -> all floating-point outputs.

    output_dtype_from_arg:
        Which forward argument determines the public output dtype.

    Does not include ctx in the argument index as it is posted separately and then the arguments are loaded in.
    """
    def decorator(cls):
        original_forward = cls.forward
        original_backward = cls.backward

        @staticmethod
        @functools.wraps(original_forward)
        def forward(ctx, *args):
            # Remember each input's original dtype for backward.
            ctx._original_dtypes = tuple(
                arg.dtype
                if isinstance(arg, torch.Tensor) and arg.is_floating_point()
                else None
                for arg in args
            )
            args = list(args)
            indices = range(len(args)) if args_to_upcast is None else args_to_upcast
            for index in indices:
                if isinstance(args[index], torch.Tensor) and args[index].is_floating_point():
                    args[index] = args[index].to(compute_dtype)
            result = original_forward(ctx, *args)
            original_output_dtype = ctx._original_dtypes[output_dtype_from_arg]

            def downcast(value, index):
                should_downcast = outputs_to_downcast is None or index in outputs_to_downcast
                if (
                    should_downcast
                    and original_output_dtype is not None
                    and isinstance(value, torch.Tensor)
                    and value.is_floating_point()
                ):
                    return value.to(original_output_dtype)
                return value

            if outputs_to_downcast is not None and len(outputs_to_downcast) == 0:
                return result

            if isinstance(result, tuple):
                return tuple(downcast(value, idx) for idx, value in enumerate(result))

            return downcast(result, 0)

        @staticmethod
        @functools.wraps(original_backward)
        def backward(ctx, *grad_outputs):
            # Run backward itself in the higher precision.
            grad_outputs = tuple(
                grad.to(compute_dtype)
                if isinstance(grad, torch.Tensor) and grad.is_floating_point()
                else grad
                for grad in grad_outputs
            )
            gradients = original_backward(ctx, *grad_outputs)
            if not isinstance(gradients, tuple):
                gradients = (gradients,)
            if len(gradients) != len(ctx._original_dtypes):
                raise RuntimeError(f"Backward returned {len(gradients)} gradients for {len(ctx._original_dtypes)} forward arguments.")

            # Each backward output corresponds to its forward input.
            result = tuple(
                grad.to(original_dtype)
                if (
                    original_dtype is not None
                    and isinstance(grad, torch.Tensor)
                    and grad.is_floating_point()
                )
                else grad
                for grad, original_dtype in zip(gradients, ctx._original_dtypes)
            )
            if len(result) == 1:
                return result[0]
            return result

        cls.forward = forward
        cls.backward = backward
        return cls
    return decorator

def is_autograd_function(obj) -> bool:
    return isinstance(obj, type) and issubclass(obj, torch.autograd.Function)


def callable_name(obj) -> str:
    if obj is None:
        return ''
    if isinstance(obj, str):
        return obj.lower().strip()

    # Handles SomeAutogradFunction.apply
    owner = getattr(obj, '__self__', None)
    if is_autograd_function(owner):
        return owner.__name__.lower()

    # Normal functions / callable methods
    name = getattr(obj, '__name__', None)
    if name is not None:
        return name.lower().strip()

    # Callable objects/modules
    return obj.__class__.__name__.lower()

class InitWeightScaling:
    """
    Weight Initialization Scaling:

        Generic/Unactivated Layer
            1/sqrt(in_features)
            • A generic linear layer at init should scaled by the 1/sqrt(in_features),
             though this could change from the activation function for example:
        Xavier/Glorot style, often for tanh/sigmoid-ish balanced layers
            W = torch.randn(in_features, out_features) * math.sqrt(2 / (in_features + out_features))
        Kaiming/He style, often for ReLU networks
            W = torch.randn(in_features, out_features) * math.sqrt(2 / in_features)
        etc...

    The reason you have to do this is that the activation function gates a lot of the output so the initial scale should be different

    Parameter Warning:
        Perform tensor transformations before wrapping the result in nn.Parameter,
        or modify an existing Parameter in-place under torch.no_grad().
    """

    def __init__(self, *, in_features: int, out_features: int, activation: str = 'generic', static_multiplier: float = None, **kwargs):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = callable_name(activation)
        self.static_multiplier = static_multiplier
        self.kwargs = kwargs

    def generic(self) -> float:
        # Fan-in
        return 1/math.sqrt(self.in_features)

    def tanh_style(self) -> float:
        # Xavier
        return math.sqrt(2 / (self.in_features + self.out_features))

    def relu_style(self) -> float:
        # Kaiming-He
        return math.sqrt(2 / self.in_features)

    def static_style(self) -> float:
        # Use any of these keys for a static multiplier
        if self.static_multiplier is None:
            raise ValueError('Must define `static_multiplier` in order to use static_style')
        return self.static_multiplier

    def static_style_on_residuals(self):
        if self.static_multiplier is None:
            raise ValueError('Must define `static_multiplier` in order to use static_style_on_residuals')
        blocks = self.kwargs['num_transformer_blocks']
        adds = self.kwargs['residual_additions']  # _per_block
        return self.static_multiplier / math.sqrt(blocks * adds)

    def get_init_scaling_const(self) -> float:
        if self.activation in ('', 'generic', 'default'):
            return self.generic()
        elif self.activation.endswith('lu'):
            return self.relu_style()
        elif self.activation in ('tanh', 'sigmoid'):
            return self.tanh_style()
        elif self.activation == 'static':
            return self.static_style()
        elif self.activation == 'static_residual':
            return self.static_style_on_residuals()
        else:
            raise ValueError(f'No init scaling const found for scaling-type {self.activation}')

    @staticmethod
    def get_const(in_features: int, out_features: int, activation: str = 'generic', static_multiplier: float = None, **kwargs):
        cls = InitWeightScaling(in_features=in_features, out_features=out_features, activation=activation,
                                static_multiplier=static_multiplier, **kwargs)
        return cls.get_init_scaling_const()

def create_weights(
        in_features: int,
        out_features: int,
        activation: str | typing.Callable = 'generic',
        static_multiplier: float = None,
        dtype=None,
        device=None,
        **kwargs
) -> torch.nn.Parameter:
    # Defining `static_multiplier` works as defining const scaling override where it will use that as the scaling thing!
    if static_multiplier is not None:
        activation = 'static'
    # Should be formatted out then in
    return torch.nn.Parameter(
        torch.randn(out_features, in_features, dtype=dtype, device=device)
        *
        InitWeightScaling.get_const(in_features, out_features, activation, static_multiplier, **kwargs)
    )

def create_biases(out_features, dtype=None, device=None):
    return torch.nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
