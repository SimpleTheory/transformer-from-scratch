import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            parameters,
            learning_rate=1e-3,
            momentum_decay=0.9,
            variance_decay=.999,
            weight_decay=1e-2,
            prevent_division_by_0=1e-8
    ):
        """
        Keep in mind Optimizer can receive multiple sets of these arguments that correspond to different parameters

        :param parameters: The parameters to adjust for this specific group
        :param learning_rate: Step size
        :param momentum_decay: How much past gradients influence the current gradient. .9 ~= to last 10 gradients whereas .99 would be last 100.
        :param variance_decay: How much past squared gradients influence the scaling .999 ~= 1000 most recent steps
        :param weight_decay: How strongly weights are pulled towards 0
        """
        # These are the defaults passed in this dict in case multiple parameter sets come in with only some arguments filled in
        defaults = {
            # lr is used instead of `learning_rate` because it is a pytorch convention, so that this works with other pytorch objects
            "lr": learning_rate,
            "momentum_decay": momentum_decay,
            "variance_decay": variance_decay,
            "prevent_division_by_0": prevent_division_by_0,
            "weight_decay": weight_decay,
        }
        super().__init__(parameters, defaults)

    @torch.no_grad()
    def get_current_param_state(self, parameter, update=True):
        current_batch_gradient = parameter.grad
        if current_batch_gradient.is_sparse:
            # Might implement this in the future
            raise RuntimeError("This AdamW implementation does not support sparse gradients.")
        parameter_state = self.state[parameter]

        if len(parameter_state) == 0:
            parameter_state["step_number"] = 0
            parameter_state["gradient_momentum"] = torch.zeros_like(parameter)
            parameter_state["gradient_squared_average"] = torch.zeros_like(parameter)

        gradient_momentum = parameter_state["gradient_momentum"]
        gradient_squared_average = parameter_state["gradient_squared_average"]

        if update:
            parameter_state["step_number"] += 1
        step_number = parameter_state["step_number"]
        return current_batch_gradient, gradient_momentum, gradient_squared_average, step_number

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for parameter_group in self.param_groups:
            # Get current parameter group's arguments
            learning_rate = parameter_group["lr"]
            momentum_decay = parameter_group["momentum_decay"]
            variance_decay = parameter_group["variance_decay"]
            prevent_division_by_0 = parameter_group["prevent_division_by_0"]
            weight_decay = parameter_group["weight_decay"]

            # Iterate over all the parameters
            for parameter in parameter_group['params']:
                if parameter.grad is None:
                    continue
                # Get the current state of the parameter
                current_batch_gradient, gradient_momentum, gradient_squared_average, step_number = self.get_current_param_state(parameter)

                # 1. Decoupled weight decay
                # AdamW shrinks the parameter directly. With the formula:
                    # parameter = parameter - learning_rate * weight_decay * parameter
                # Equivalent to:
                    # parameter = parameter * (1 - learning_rate * weight_decay)
                # Because these are referenced in memory and you want to modify the original copies, in place operations are best.
                parameter *= 1 - learning_rate * weight_decay

                # 2. Update average of gradients
                # This tracks the recent direction gradients have pointed with the formula:
                # gradient_momentum = momentum_decay * old_gradient_momentum + (1 - momentum_decay) * current_batch_gradient
                # AKA: m = m * .9 + current * .1

                # Because these are referenced in memory and you want to modify the original copies, in place operations are best.
                gradient_momentum *= momentum_decay
                gradient_momentum += (1 - momentum_decay) * current_batch_gradient

                # 3. Update average magnitude of gradient
                # This tracks how large the gradients have been recently, with the formula:
                # gradient_squared_average = variance_decay * old_gradient_squared_average + (1 - variance_decay) * current_batch_gradient**2
                # AKA mag = .999 * mag + .001 * grad**2
                # Because these are referenced in memory and you want to modify the original copies, in place operations are best.
                gradient_squared_average *= variance_decay
                gradient_squared_average += (1 - variance_decay) * current_batch_gradient**2

                # 4. Bias correction
                # Since the gradient and magnitude averages start at zero, they are too small during the first few steps.
                # Bias correction fixes that. Basically blows each up relative their decay to compensate for that 0 start.
                gradient_momentum_bias_corrected = gradient_momentum / (1 - momentum_decay ** step_number)
                gradient_squared_average_bias_corrected = gradient_squared_average / (1 - variance_decay ** step_number)

                # 5. Compute adaptive denominator
                # Get the true magnitude from the average bias corrected magnitude by taking its square root and add a
                # small number to avoid division by 0 in the adam formula
                true_magnitude_denominator = (
                    # Getting the true magnitude (taking out **2)
                    gradient_squared_average_bias_corrected.sqrt()
                    + prevent_division_by_0
                )

                # 6. Apply the Adam Formula to the parameter:
                # parameter = parameter - learning_rate * (grad_mom_corrected/grad_magnitude_corrected)
                # Because these are referenced in memory and you want to modify the original copies, in place operations are best.
                parameter -= learning_rate * (gradient_momentum_bias_corrected/true_magnitude_denominator)
        return loss

