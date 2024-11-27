import torch
import torch.nn.functional as F
import torch.optim


class L1LossFunction(torch.autograd.Function):
    """
    This is the class that represents the L1 loss function.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the L1 loss function.

        Args:
            ctx: the context.
            inputs: the input tensor. Dimension [batch_size, *]
            targets: the target tensor. Dimension [batch_size, *]

        Returns:
            the output tensor. Dimension [batch_size], a tensor with the loss for each sample in the batch.
        """
        # TODO
        diff = inputs - targets
        ctx.save_for_backward(diff)
        return torch.abs(diff).mean()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method computes the backward pass of the L1 loss function.

        Args:
            ctx: the context.
            grad_output: the gradient of the loss with respect to the output tensor. Dimension [batch_size]

        tuple[torch.Tensor, torch.Tensor]:
                the gradient of the loss with respect to the input tensor
                and the target tensor. Dimension [batch_size, *], same as the input tensor.
        """
        # TODO
        (diff,) = ctx.saved_tensors
        grad_input = torch.sign(diff) / diff.numel()
        return grad_input * grad_output, None


class L1Loss(torch.nn.Module):
    """
    This is the class that represents the L1 loss function.
    """

    def __init__(self):
        """
        This method is the constructor of the class.
        """
        super(L1Loss, self).__init__()

        self.fn = L1LossFunction.apply

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the L1 loss function.

        Args:
            inputs: the input tensor. Dimension [batch_size, *]
            targets: the target tensor. Dimension [batch_size, *]

        Returns:
            the output tensor. Dimension [batch_size], a tensor with the loss for each sample in the batch.
        """
        return self.fn(inputs, targets)


class MSELossFunction(torch.autograd.Function):
    """
    This is the class that represents the Mean Squared Error (MSE) loss function.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the Mean Squared Error (MSE) loss function.

        Args:
            ctx: the context.
            inputs: the input tensor. Dimension [batch_size, *]
            targets: the target tensor. Dimension [batch_size, *]

        Returns:
            the output tensor. Dimension [batch_size], a tensor with the loss for each sample in the batch.
        """
        # TODO
        diff = inputs - targets
        ctx.save_for_backward(diff)
        return torch.square(diff).mean()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method computes the backward pass of the Mean Squared Error (MSE) loss function.

        Args:
            ctx: the context.
            grad_output: the gradient of the loss with respect to the output tensor. Dimension [batch_size]

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                the gradient of the loss with respect to the input tensor
                and the target tensor. Dimension [batch_size, *], same as the input tensor.
        """
        # TODO
        (diff,) = ctx.saved_tensors
        grad_input = 2 * diff / diff.numel()
        return grad_input * grad_output, None


class MSELoss(torch.nn.Module):
    """
    This is the class that represents the Mean Squared Error (MSE) loss function.
    """

    def __init__(self):
        """
        This method is the constructor of the class.
        """
        super(MSELoss, self).__init__()

        self.fn = MSELossFunction.apply

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the Mean Squared Error (MSE) loss function.

        Args:
            inputs: the input tensor. Dimension [batch_size, *]
            targets: the target tensor. Dimension [batch_size, *]

        Returns:
            the output tensor. Dimension [batch_size], a tensor with the loss for each sample in the batch.
        """
        return self.fn(inputs, targets)


class HuberLossFunction(torch.autograd.Function):
    """
    This is the class that represents the Huber loss function.
    """

    @staticmethod
    def forward(
        ctx, inputs: torch.Tensor, targets: torch.Tensor, delta: float = 1.0
    ) -> torch.Tensor:
        """
        This method computes the forward pass of the Huber loss function.

        Args:
            ctx: the context.
            inputs: the input tensor. Dimension [batch_size, *]
            targets: the target tensor. Dimension [batch_size, *]
            delta: the threshold for the loss. Dimension []

        Returns:
            the output tensor. Dimension [batch_size], a tensor with the loss for each sample in the batch.
        """
        # TODO
        ctx.delta = delta
        diff = inputs - targets
        ctx.save_for_backward(diff)
        abs_diff = torch.abs(diff)
        loss = torch.where(
            abs_diff < delta, 0.5 * torch.square(diff), delta * (abs_diff - 0.5 * delta)
        )
        return loss.mean()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method computes the backward pass of the Huber loss function.

        Args:
            ctx: the context.
            grad_output: the gradient of the loss with respect to the output tensor. Dimension [batch_size]

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                the gradient of the loss with respect to the input tensor
                and the target tensor. Dimension [batch_size, *], same as the input tensor.
        """
        # TODO
        (diff,) = ctx.saved_tensors
        delta = ctx.delta
        grad_input = (
            torch.where(torch.abs(diff) < delta, diff, delta * torch.sign(diff))
            / diff.numel()
        )
        return grad_input * grad_output, None, None


class HuberLoss(torch.nn.Module):
    """
    This is the class that represents the Huber loss function.
    """

    def __init__(self, delta: float = 1.0):
        """
        This method is the constructor of the class.

        Args:
            delta: the threshold for the loss. Dimension []
        """
        super(HuberLoss, self).__init__()

        self.delta = delta
        self.fn = HuberLossFunction.apply

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the Huber loss function.

        Args:
            inputs: the input tensor. Dimension [batch_size, *]
            targets: the target tensor. Dimension [batch_size, *]

        Returns:
            the output tensor. Dimension [batch_size], a tensor with the loss for each sample in the batch.
        """
        return self.fn(inputs, targets, self.delta)


class BinaryCrossEntropyLossFunction(torch.autograd.Function):
    """
    This is the class that represents the Binary Cross Entropy loss function.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the Binary Cross Entropy loss function.

        Args:
            ctx: the context.
            inputs: the input tensor. Dimension [batch_size, num_classes]
            targets: the target tensor. Dimension [batch_size, num_classes]

        Returns:
            the output tensor. Dimension [batch_size], a tensor with the loss for each sample in the batch.
        """
        # TODO
        log_sigmoid = inputs - torch.log1p(torch.exp(inputs))
        log_1msigmoid = -torch.log1p(torch.exp(inputs))
        bcel = targets * log_sigmoid + (1 - targets) * log_1msigmoid
        ctx.save_for_backward(log_sigmoid, targets)
        return -bcel.mean()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method computes the backward pass of the Binary Cross Entropy loss function.

        Args:
            ctx: the context.
            grad_output: the gradient of the loss with respect to the output tensor. Dimension [1]

        Returns:
            the gradient of the loss with respect to the input tensor. Dimension [batch_size, num_classes], same as the input tensor.
        """
        # TODO
        log_sigmoid, targets = ctx.saved_tensors
        grad_input = (log_sigmoid.exp() - targets) / targets.numel()
        return grad_input * grad_output, None


class BinaryCrossEntropyLoss(torch.nn.Module):
    """
    This is the class that represents the Binary Cross Entropy loss function.
    """

    def __init__(self):
        """
        This method is the constructor of the class.
        """
        super(BinaryCrossEntropyLoss, self).__init__()

        self.fn = BinaryCrossEntropyLossFunction.apply

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the Binary Cross Entropy loss function.

        Args:
            inputs: the input tensor. Dimension [batch_size]
            targets: the target tensor. Dimension [batch_size]

        Returns:
            the output tensor. Dimension [batch_size], a tensor with the loss for each sample in the batch.
        """
        return self.fn(inputs, targets)


class CrossEntropyLossFunction(torch.autograd.Function):
    """
    This is the class that represents the Cross Entropy loss function.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the Cross Entropy loss function.

        Args:
            ctx: the context.
            inputs: the input tensor. Dimension [batch_size, num_classes]
            targets: the target tensor. Dimension [batch_size]

        Returns:
            the output tensor. Dimension [batch_size], a tensor with the loss for each sample in the batch.
        """
        # TODO
        ctx.save_for_backward(inputs, targets)
        log_softmax = inputs[range(targets.shape[0]), targets] - torch.logsumexp(
            inputs, dim=1
        )
        return -log_softmax.mean()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method computes the backward pass of the Cross Entropy loss function.

        Args:
            ctx: the context.
            grad_output: the gradient of the loss with respect to the output tensor. Dimension [batch_size]

        Returns:
            the gradient of the loss with respect to the input tensor. Dimension [batch_size, num_classes], same as the input tensor.
        """
        # TODO
        inputs, targets = ctx.saved_tensors
        log_softmax = inputs - torch.logsumexp(inputs, dim=1, keepdim=True)
        grad_input = log_softmax.exp()
        grad_input[range(targets.shape[0]), targets] -= 1
        grad_input /= targets.shape[0]
        return grad_input * grad_output, None


class CrossEntropyLoss(torch.nn.Module):
    """
    This is the class that represents the Cross Entropy loss function.
    """

    def __init__(self):
        """
        This method is the constructor of the class.
        """
        super(CrossEntropyLoss, self).__init__()

        self.fn = CrossEntropyLossFunction.apply

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the Cross Entropy loss function.

        Args:
            inputs: the input tensor. Dimension [batch_size, num_classes]
            targets: the target tensor. Dimension [batch_size]

        Returns:
            the output tensor. Dimension [batch_size], a tensor with the loss for each sample in the batch.
        """
        return self.fn(inputs, targets)
