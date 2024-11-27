import torch
from typing import Optional
import torch.optim


class MultiStepLR(torch.optim.lr_scheduler.LRScheduler):
    """
    This class represents a learning rate scheduler that multiplies the learning rate by gamma at each milestone.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: list[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        """
        This method initializes the learning rate scheduler.
        """
        self.milestones = milestones
        self.gamma = gamma
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        # Si no llamais al super no funciona
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def step(self, epoch=None) -> None:
        """
        This method updates the learning rate at each epoch.
        """
        self.last_epoch += 1
        if self.last_epoch in self.milestones:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.gamma
        return None


class WarmupMultiStepLR(torch.optim.lr_scheduler.LRScheduler):
    """
    This class represents a learning rate scheduler that multiplies the learning rate by gamma at each milestone.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: list[int],
        warmup_start_value: float,
        warmup_duration: int,
        warmup_end_value: float,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        """
        This method initializes the learning rate scheduler.
        """
        self.milestones = milestones
        self.gamma = gamma
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        self.warmup_start_value = warmup_start_value
        self.warmup_duration = warmup_duration
        self.warmup_end_value = warmup_end_value

        # Variable to count the number of warmup steps
        self.warmup_steps = 0

        # Si no llamais al super no funciona
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def step(self, epoch=None) -> None:
        """
        This method updates the learning rate at each epoch.
        """
        if self.warmup_steps < self.warmup_duration:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = (
                    self.warmup_start_value
                    + (self.warmup_end_value - self.warmup_start_value)
                    * self.warmup_steps
                    / self.warmup_duration
                )
            self.warmup_steps += 1
        else:
            self.last_epoch += 1
            if self.last_epoch in self.milestones:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] *= self.gamma
        return None


class Warmup(torch.optim.lr_scheduler._LRScheduler):
    """
    This class is a custom implementation of the Warmup algorithm.

    Attr:
        n: number of epochs.
        warmup_epochs: number of epochs to warmup.
        model: model to train.
        loss_fn: loss function.
        optimizer: optimizer.
    """

    def __init__(self, n, warmup_epochs, model, loss_fn, optimizer):
        self.n = n
        self.warmup_epochs = warmup_epochs
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        super(Warmup, self).__init__()

    def step(self, epoch):
        if epoch <= self.warmup_epochs:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = (epoch / self.warmup_epochs) * param_group["lr"]
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.1


class ExponentialLR(torch.optim.lr_scheduler.LRScheduler):
    """
    Sets the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch.
    """

    optimizer: torch.optim.Optimizer
    gamma: float
    last_epoch: int

    def __init__(
        self, optimizer: torch.optim.Optimizer, gamma: float = 0.1, last_epoch: int = -1
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            gamma (float): Multiplicative factor of learning rate decay.
                Default: 0.1.
            last_epoch (int): The index of last epoch. Default: -1.
        """

        self.optimizer = optimizer
        self.gamma = gamma
        self.last_epoch = last_epoch

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, epoch: Optional[int] = None) -> None:
        """
        Args:
            epoch (int): The epoch at which to adjust the learning rate.
                Default: None.
        """

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * self.gamma


class StepLR(torch.optim.lr_scheduler.LRScheduler):
    """
    This

    Attr:
        optimizer: optimizer that the scheduler is using.
        step_size: number of steps to decrease learning rate.
        gamma: factor to decrease learning rate.
        count: count of steps.
    """

    optimizer: torch.optim.Optimizer
    step_size: int
    gamma: float
    last_epoch: int

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float = 0.1,
    ) -> None:
        """
        This method is the constructor of StepLR class.

        Args:
            optimizer: optimizer.
            step_size: size of the step.
            gamma: factor to change the lr. Defaults to 0.1.
        """

        # set attributes
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = -1

        # call super class constructor
        super(StepLR, self).__init__(optimizer)

    def step(self, epoch: Optional[int] = None) -> None:
        """
        This function is the step of the scheduler.

        Args:
            epoch: ignore this argument. Defaults to None.
        """

        # TODO

        # update count
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # check if it is time to update lr
        if self.last_epoch % self.step_size == 0 and self.last_epoch != 0:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.gamma

        return None


class ReduceLROnPlateau(torch.optim.lr_scheduler._LRScheduler):
    """
    This class is a custom implementation of the ReduceLROnPlateau algorithm.

    Attr:
        patience: number of epochs to wait before reducing the learning rate.
        factor: factor to reduce the learning rate.
        min_lr: minimum learning rate.
        best_loss: best loss found.
        counter: counter for the number of epochs.
    """

    # define attributes
    patience: int
    factor: float
    min_lr: float
    best_loss: float
    counter: int

    def __init__(
        self,
        patience: int = 5,
        factor: float = 0.1,
        min_lr: float = 1e-6,
        optimizer: torch.optim.Optimizer = None,
    ) -> None:
        """
        This is the constructor for ReduceLROnPlateau.

        Args:
            patience: number of epochs to wait before reducing the learning rate. Defaults to 5.
            factor: factor to reduce the learning rate. Defaults to 0.1.
            min_lr: minimum learning rate. Defaults to 1e-6.
        """

        # define attributes
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float("inf")
        self.counter = 0

        super(ReduceLROnPlateau, self).__init__(optimizer)

    def step(self, loss: float) -> None:
        """
        This method is the step of the ReduceLROnPlateau algorithm.

        Args:
            loss: loss value.
        """

        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = max(
                        param_group["lr"] * self.factor, self.min_lr
                    )
                self.counter = 0

        return None
