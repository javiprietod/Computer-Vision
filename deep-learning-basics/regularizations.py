import torch
import torch.optim


class L1Regularization(torch.nn.Module):
    """
    This class is a custom
    implementation of the L1 regularization.
    """

    def __init__(self, l1_lambda: float = 0.0) -> None:
        """
        This is the constructor for the L1 regularization.

        Args:
            l1_lambda: L1 regularization lambda. Defaults to 0.0.
        """

        # call super constructor
        super().__init__()

        # define attributes
        self.l1_lambda = l1_lambda

    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        """
        This method is the forward of the L1 regularization.

        Args:
            model: model to apply the L1 regularization.

        Returns:
            loss with the L1 regularization.
        """

        # initialize loss
        loss = 0

        # iterate over the model parameters
        for param in model.parameters():
            loss += torch.norm(param, p=1)

        # return loss
        return self.l1_lambda * loss


class L2Regularization(torch.nn.Module):
    """
    This class is a custom
    implementation of the L2 regularization.
    """

    def __init__(self, l2_lambda: float = 0.0) -> None:
        """
        This is the constructor for the L2 regularization.

        Args:
            l2_lambda: L2 regularization lambda. Defaults to 0.0.
        """

        # call super constructor
        super().__init__()

        # define attributes
        self.l2_lambda = l2_lambda

    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        """
        This method is the forward of the L2 regularization.

        Args:
            model: model to apply the L2 regularization.

        Returns:
            loss with the L2 regularization.
        """

        # initialize loss
        loss = 0

        # iterate over the model parameters
        for param in model.parameters():
            loss += torch.norm(param, p=2)

        # return loss
        return self.l2_lambda * loss
