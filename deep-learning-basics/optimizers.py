import torch
from typing import Iterator, Dict, Any, DefaultDict
import torch.optim


class SGD(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self, params: Iterator[torch.nn.Parameter], lr=1e-3, weight_decay: float = 0.0
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # define defaults
        defaults: Dict[Any, Any] = dict(lr=lr, weight_decay=weight_decay)

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # Optional closure argument is ignored in this implementation
        # closure is a mechanism to re-evaluate the model and return the loss

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            lr = group["lr"]

            for p in group["params"]:
                d_p = p.grad.data
                d_p += p.data * weight_decay
                p.data -= d_p * lr


class SGDMomentum(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

    Attr:
        param_groups: list with the dict of the parameters.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # define defaults
        defaults: Dict[Any, Any] = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Attr:
            param_groups: list with the dict of the parameters.
            state: dict with the state for each parameter.
        """

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            momentum = group.get("momentum", 0)

            for p in group["params"]:
                d_p = p.grad.data
                d_p += p.data * weight_decay

                param_state = self.state[p]

                # If the momentum buffer is not in the state, create it (initialize it)
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                else:
                    buf = param_state["momentum_buffer"]
                    buf = buf * momentum + d_p

                self.state[p]["momentum_buffer"] = buf

                p.data -= buf * lr


class SGDNesterov(torch.optim.Optimizer):
    """
    This class is a custom implementation of the SGD algorithm with
    momentum.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # define defaults
        defaults: Dict[Any, Any] = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            momentum = group.get("momentum", 0)

            for p in group["params"]:
                d_p = p.grad.data
                d_p += p.data * weight_decay

                param_state = self.state[p]

                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                else:
                    buf = param_state["momentum_buffer"]
                    buf = buf * momentum + d_p

                self.state[p]["momentum_buffer"] = buf

                d_p += buf * momentum

                p.data -= d_p * lr


class Adam(torch.optim.Optimizer):
    """
    This class is a custom implementation of the Adam algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # define defaults
        defaults: Dict[Any, Any] = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                d_p = p.grad.data
                d_p += p.data * weight_decay

                param_state = self.state[p]

                # If parameters are not in the state, create them (initialize them)
                if "step" not in param_state:
                    param_state["step"] = 1
                    param_state["exp_avg"] = torch.zeros_like(p.data)
                    param_state["exp_avg_sq"] = torch.zeros_like(p.data)
                else:
                    param_state["step"] += 1

                exp_avg, exp_avg_sq = param_state["exp_avg"], param_state["exp_avg_sq"]

                exp_avg = beta1 * exp_avg + (1 - beta1) * d_p
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * d_p**2

                # Update the state
                self.state[p]["exp_avg"] = exp_avg
                self.state[p]["exp_avg_sq"] = exp_avg_sq

                # Correct the values of exp_avg and exp_avg_sq
                step = param_state["step"]
                exp_avg_corr = exp_avg / (1 - beta1**step)
                exp_avg_sq_corr = exp_avg_sq / (1 - beta2**step)

                # Update using the corrected values (exp_avg_corr and exp_avg_sq_corr)
                p.data -= lr * exp_avg_corr / (torch.sqrt(exp_avg_sq_corr) + eps)


class Adagrad(torch.optim.Optimizer):
    """
    This class is a custom implementation of the Adam algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        lr_decay: float = 0.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for Adagrad.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
            lr_decay: decay of learning rate. Defaults to 0.
            eps: epsilon value for avoiding overflow. Defaults to 0.
            weight_decay: weight decay rate. Defaults to 0.
        """

        # TODO

        # define defaults
        defaults: Dict[Any, Any] = dict(
            lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay
        )

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        # TODO

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            lr_decay = group["lr_decay"]
            eps = group["eps"]

            for p in group["params"]:
                d_p = p.grad.data
                d_p += p.data * weight_decay

                param_state = self.state[p]

                # If parameters are not in the state, create them
                if "step" not in param_state:
                    param_state["step"] = 1
                    param_state["sum"] = 0
                else:
                    param_state["step"] += 1

                lr_aprox = lr / (1 + (param_state["step"] - 1) * lr_decay)
                param_state["sum"] += d_p**2

                suma = param_state["sum"]

                # Update the state
                self.state[p]["sum"] = suma

                # Update using the values
                p.data -= lr_aprox * d_p / (eps + torch.sqrt(suma))


class RMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8):
        defaults = dict(lr=lr, alpha=alpha, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["square_avg"] = torch.zeros_like(p.data)

                # Update square average
                square_avg = state["square_avg"]
                square_avg = (
                    square_avg * group["alpha"] + (1 - group["alpha"]) * grad * grad
                )
                state["square_avg"] = square_avg
                std = (square_avg + group["eps"]).sqrt()
                p.data = p.data - group["lr"] * grad / std

        return loss


class AdamW(torch.optim.Optimizer):
    """
    This class is a custom implementation of the AdamW algorithm.

    Attr:
        param_groups: list with the dict of the parameters.
        state: dict with the state for each parameter.
    """

    # define attributes
    param_groups: list[Dict[str, torch.Tensor]]
    state: DefaultDict[torch.Tensor, Any]

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr=1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """
        This is the constructor for SGD.

        Args:
            params: parameters of the model.
            lr: learning rate. Defaults to 1e-3.
        """

        # define defaults
        defaults: Dict[Any, Any] = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

        # call super class constructor
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure: None = None) -> None:  # type: ignore
        """
        This method is the step of the optimization algorithm.

        Args:
            closure: Ignore this parameter. Defaults to None.
        """

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                d_p = p.grad.data
                d_p -= p.data * weight_decay * lr

                param_state = self.state[p]

                # If parameters are not in the state, create them (initialize them)
                if "step" not in param_state:
                    param_state["step"] = 1
                    param_state["exp_avg"] = torch.zeros_like(p.data)
                    param_state["exp_avg_sq"] = torch.zeros_like(p.data)
                else:
                    param_state["step"] += 1

                exp_avg, exp_avg_sq = param_state["exp_avg"], param_state["exp_avg_sq"]

                exp_avg = beta1 * exp_avg + (1 - beta1) * d_p
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * d_p**2

                # Update the state
                self.state[p]["exp_avg"] = exp_avg
                self.state[p]["exp_avg_sq"] = exp_avg_sq

                # Correct the values of exp_avg and exp_avg_sq
                step = param_state["step"]
                exp_avg_corr = exp_avg / (1 - beta1**step)
                exp_avg_sq_corr = exp_avg_sq / (1 - beta2**step)

                # Update using the corrected values (exp_avg_corr and exp_avg_sq_corr)
                p.data -= lr * exp_avg_corr / (torch.sqrt(exp_avg_sq_corr) + eps)
