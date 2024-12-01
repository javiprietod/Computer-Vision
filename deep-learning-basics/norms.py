import torch


class BatchNorm1d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = torch.nn.Parameter(torch.ones(num_features))
        self.beta = torch.nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    @staticmethod
    def forward(ctx, self, input):
        """
        input: (batch_size, num_features)

        returns: (batch_size, num_features)
        """

        # Calculate mean and variance
        mean = input.mean(dim=0)
        var = input.var(dim=0, unbiased=False)

        # Update running mean and variance
        self.running_mean = (
            1 - self.momentum
        ) * self.running_mean + self.momentum * mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        # Normalize
        output = (input - mean) / (var + self.eps).sqrt()
        ctx.desv = (var + self.eps).sqrt()
        ctx.mean = mean

        # Scale and shift
        output = self.gamma * output + self.beta

        return output

    @staticmethod
    def backward(ctx, self, grad_output):
        """
        This method computes the backward pass of the Batch Normalization layer.

        Args:
            grad_output: the gradient of the loss with respect to the output of the layer.

        Returns:
            the gradient of the loss with respect to the input of the layer.
            grad_scale: the gradient of the loss with respect to the scale parameter.
            grad_bias: the gradient of the loss with respect to the bias parameter.
        """
        # Compute the gradient of the loss with respect to the input of the layer
        grad_input = grad_output * self.gamma / ctx.desv

        # Compute the gradient of the loss with respect to the scale parameter
        grad_scale = (grad_output * (ctx.x - ctx.mean) / ctx.desv).sum(dim=0)

        # Compute the gradient of the loss with respect to the bias parameter
        grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_scale, grad_bias


class BatchNorm2d(torch.nn.Module):
    """
    This is the class that represents the Batch Normalization layer.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        """
        This method is the constructor of the class.

        Args:
            num_features: the number of channels in the input tensor.
            eps: a value added to the denominator for numerical stability.
            momentum: the value used for the running_mean and running_var computation.
            affine: use learnable scale and bias if True.
        """
        super(BatchNorm2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if self.affine:
            self.scale = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_features))

        # These are the estimated mean and variance of the dataset
        # We do this because we want torch to automatically recognize these
        # tensors and change their device when we call model.to(device)
        # If we create these tensors like: self.running_mean = torch.zeros(num_features)
        # then model.to(device) will not change their device automatically and will raise an error
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.running_mean: torch.Tensor
        self.running_var: torch.Tensor

    def forward(ctx, self, x: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the Batch Normalization layer.
        It also keeps track of the running mean and variance of the dataset using
        exponential moving average if the layer is in training mode.

        Args:
            x: the input tensor. Dimension [N, C, H, W]

        Returns:
            the output tensor. Dimension [N, C, H, W], same as the input tensor.
        """
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            n = x.numel() / x.shape[1]
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var.squeeze() * (n / (n - 1))
        else:
            mean = self.running_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            var = self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = (x - mean) / (var + self.eps).sqrt()
        ctx.desv = (var + self.eps).sqrt()
        ctx.mean = mean

        if self.affine:
            x = x * self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(
                3
            ) + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return x

    def backward(ctx, self, grad_output):
        """
        This method computes the backward pass of the Batch Normalization layer.

        Args:
            grad_output: the gradient of the loss with respect to the output of the layer.

        Returns:
            the gradient of the loss with respect to the input of the layer.
            grad_scale: the gradient of the loss with respect to the scale parameter.
            grad_bias: the gradient of the loss with respect to the bias parameter.
        """
        # Compute the gradient of the loss with respect to the input of the layer
        grad_input = grad_output * self.scale / ctx.desv

        # Compute the gradient of the loss with respect to the scale parameter
        grad_scale = (grad_output * (ctx.x - ctx.mean) / ctx.desv).sum(dim=(0, 2, 3))

        # Compute the gradient of the loss with respect to the bias parameter
        grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input


class GroupNorm(torch.nn.Module):
    """
    This is the class that represents the Group Normalization layer.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """
        This method is the constructor of the class.

        Args:
            num_groups: the number of groups to separate the channels into.
            num_channels: the number of channels in the input tensor.
            eps: a value added to the denominator for numerical stability.
            affine: use learnable scale and bias if True.
        """
        super(GroupNorm, self).__init__()

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        assert (
            num_channels % num_groups == 0
        ), "Number of channels must be divisible by the number of groups."

        if self.affine:
            self.scale = torch.nn.Parameter(torch.ones(num_channels))
            self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(ctx, self, x: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the Group Normalization layer.

        Args:
            x: the input tensor. Dimension [N, C, H, W]

        Returns:
            the output tensor. Dimension [N, C, H, W], same as the input tensor.
        """
        N, C, H, W = x.shape

        x = x.view(N, self.num_groups, -1)

        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, unbiased=False, keepdim=True)
        ctx.desv = (var + self.eps).sqrt()
        ctx.mean = mean

        x = (x - mean) / (var + self.eps).sqrt()

        x = x.view(N, C, H, W)

        if self.affine:
            x = x * self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(
                3
            ) + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return x

    def backward(ctx, self, grad_output):
        """
        This method computes the backward pass of the Group Normalization layer.

        Args:
            grad_output: the gradient of the loss with respect to the output of the layer.

        Returns:
            the gradient of the loss with respect to the input of the layer.
            grad_scale: the gradient of the loss with respect to the scale parameter.
            grad_bias: the gradient of the loss with respect to the bias parameter.
        """
        # Compute the gradient of the loss with respect to the input of the layer
        grad_input = grad_output * self.scale / ctx.desv

        # Compute the gradient of the loss with respect to the scale parameter
        grad_scale = (grad_output * (ctx.x - ctx.mean) / ctx.desv).sum(dim=(0, 2, 3))

        # Compute the gradient of the loss with respect to the bias parameter
        grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_scale, grad_bias


class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.sc = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bc = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(ctx, self, input):
        """
        input: (batch_size, normalized_shape)

        returns: (batch_size, normalized_shape)
        """

        # Calculate mean and variance
        mean = input.mean(dim=1)
        var = input.var(dim=1, unbiased=False)
        ctx.mean = mean
        ctx.desv = var

        # Normalize
        output = (input - mean.unsqueeze(1)) / (var + self.eps).sqrt()

        # Scale and shift
        output = self.sc * output + self.bc

        return output

    def backward(ctx, self, grad_output):
        """
        This method computes the backward pass of the Layer Normalization layer.

        Args:
            grad_output: the gradient of the loss with respect to the output of the layer.

        Returns:
            the gradient of the loss with respect to the input of the layer.
            grad_scale: the gradient of the loss with respect to the scale parameter.
            grad_bias: the gradient of the loss with respect to the bias parameter.
        """
        # Compute the gradient of the loss with respect to the input of the layer
        grad_input = grad_output * self.sc / ctx.desv

        # Compute the gradient of the loss with respect to the scale parameter
        grad_scale = (grad_output * (ctx.x - ctx.mean) / ctx.desv).sum(dim=1)

        # Compute the gradient of the loss with respect to the bias parameter
        grad_bias = grad_output.sum(dim=1)

        return grad_input, grad_scale, grad_bias
