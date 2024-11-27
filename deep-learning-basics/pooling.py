import torch
import torch.nn.functional as F
import torch.optim


class MaxPooling(torch.nn.Module):
    def __init__(self, kernel_size):
        super(MaxPooling, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return MaxPoolingFunction.apply(x, self.kernel_size)


class MaxPoolingFunction(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, inputs, kernel_size):
        batch_size, channels, height, width = inputs.shape
        kh, kw = kernel_size

        # Calculate output dimensions
        out_height = height // kh
        out_width = width // kw

        # Unfold the input to get sliding windows
        unfolded_input = F.unfold(inputs, kernel_size=kernel_size, stride=kernel_size)
        unfolded_input = unfolded_input.view(
            batch_size, channels, kh * kw, out_height * out_width
        )

        # Find the max and indices of the max
        output, indices = torch.max(unfolded_input, dim=2, keepdim=False)
        output = output.view(batch_size, channels, out_height, out_width)

        # Save variables for backward pass
        ctx.save_for_backward(
            inputs, indices, torch.tensor(kernel_size), torch.tensor([height, width])
        )
        ctx.mark_non_differentiable(indices)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, indices, kernel_size, input_size = ctx.saved_tensors
        kh, kw = kernel_size.tolist()
        height, width = input_size.tolist()

        batch_size, channels, out_height, out_width = grad_output.shape

        # Prepare the output gradient tensor
        grad_input = torch.zeros(
            (batch_size, channels, kh * kw, out_height * out_width),
            device=grad_output.device,
        )

        # Reshape grad_output to match the unfolded input shape
        grad_output_reshaped = grad_output.view(
            batch_size, channels, 1, out_height * out_width
        ).expand(-1, -1, kh * kw, -1)

        # Correct indices shape to match the last dimension
        indices = indices.view(batch_size, channels, out_height * out_width)

        # Use indices to place gradients
        grad_input.scatter_(2, indices.unsqueeze(2), grad_output_reshaped)

        # Fold the unfolded gradients back to the input dimensions
        grad_input = grad_input.view(
            batch_size, channels * kh * kw, out_height * out_width
        )
        grad_input = F.fold(
            grad_input,
            output_size=(height, width),
            kernel_size=kernel_size,
            stride=kernel_size,
        )

        return grad_input, None


class AvgPooling(torch.nn.Module):
    def __init__(self, kernel_size):
        super(AvgPooling, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return AvgPoolingFunction.apply(x, self.kernel_size)


class AvgPoolingFunction(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, inputs, kernel_size):
        batch_size, channels, height, width = inputs.shape
        kh, kw = kernel_size

        # Calculate output dimensions
        out_height = (height - kh) // kh + 1
        out_width = (width - kw) // kw + 1

        # Unfold the input to get sliding windows
        unfolded_input = F.unfold(inputs, kernel_size=kernel_size, stride=kernel_size)
        # Calculate the average of the patches
        output = unfolded_input.view(batch_size, channels, kh * kw, -1).mean(dim=2)
        output = output.view(batch_size, channels, out_height, out_width)

        # Save variables for backward pass
        ctx.save_for_backward(inputs, torch.tensor(kernel_size))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, kernel_size_tensor = ctx.saved_tensors
        kh, kw = tuple(kernel_size_tensor.tolist())
        batch_size, channels, height, width = inputs.shape

        # Calculate the dimensions of the output of the forward pass
        out_height = (height - kh) // kh + 1
        out_width = (width - kw) // kw + 1

        # Reshape grad_output correctly
        grad_output_reshaped = grad_output.view(
            batch_size, channels, out_height, out_width
        )

        # Create a tensor of ones size (kh, kw) and multiply each grad_output by this to simulate average pooling
        grad_expanded = grad_output_reshaped.unsqueeze(2).repeat(1, 1, kh * kw, 1, 1)
        grad_expanded = grad_expanded.reshape(
            batch_size, channels, kh, kw, out_height, out_width
        )
        grad_expanded = grad_expanded.permute(0, 1, 4, 2, 5, 3).reshape(
            batch_size, channels, out_height * kh, out_width * kw
        )

        # Calculate the gradient per element by dividing by the kernel area
        grad_input = grad_expanded / (kh * kw)

        return grad_input, None


class MaxUnpooling(torch.nn.Module):
    def __init__(self, kernel_size):
        super(MaxUnpooling, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x, indices, output_size):
        return MaxUnpoolingFunction.apply(x, indices, self.kernel_size, output_size)


class MaxUnpoolingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indices, kernel_size, output_size):
        ctx.save_for_backward(
            indices, torch.tensor(kernel_size), torch.tensor(output_size)
        )
        batch_size, channels, height, width = inputs.shape
        kh, kw = kernel_size

        # Initialize output tensor with zeros in the shape of the original input size
        output = torch.zeros(
            (batch_size, channels) + tuple(output_size), device=inputs.device
        )

        # Unfold the indices to match the output size
        unfolded_indices = indices.view(batch_size, channels, -1)

        # Unfold the inputs to match the indices size
        unfolded_inputs = inputs.view(batch_size, channels, -1)

        # Use the indices to scatter inputs back to the original positions
        output = output.view(batch_size, channels, -1)
        output.scatter_(2, unfolded_indices, unfolded_inputs)
        output = output.view(batch_size, channels, output_size[0], output_size[1])

        return output


class MaxPool2d_malo(torch.nn.Module):
    """
    This class is the MaxPool2d class.
    """

    def __init__(self, kernel_size: int, stride: int) -> None:
        """
        This method is the constructor of the MaxPool2d class.

        Args:
            kernel_size: size of the kernel.
            stride: stride of the kernel.
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the MaxPool2d class.

        Args:
            inputs: batch of tensors.
                Dimensions: [batch, input_channels, height, width]

        Returns:
            batch of tensors. Dimensions: [batch, input_channels,
                (height - kernel_size)/stride + 1, (width - kernel_size)/stride + 1].
        """

        # Do it manually
        batch, input_channels, height, width = inputs.shape
        new_height = (height - self.kernel_size) // self.stride + 1
        new_width = (width - self.kernel_size) // self.stride + 1
        output = torch.zeros(batch, input_channels, new_height, new_width)
        for i in range(new_height):
            for j in range(new_width):
                output[:, :, i, j] = torch.max(
                    inputs[:, :, i : i + self.kernel_size, j : j + self.kernel_size],
                    dim=(2, 3),
                ).values
        return output


class AvgPool2d_malo(torch.nn.Module):
    """
    This class is the AvgPool2d class.
    """

    def __init__(self, kernel_size: int, stride: int) -> None:
        """
        This method is the constructor of the AvgPool2d class.

        Args:
            kernel_size: size of the kernel.
            stride: stride of the kernel.
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the AvgPool2d class.

        Args:
            inputs: batch of tensors.
                Dimensions: [batch, input_channels, height, width]

        Returns:
            batch of tensors. Dimensions: [batch, input_channels,
                (height - kernel_size)/stride + 1, (width - kernel_size)/stride + 1].
        """

        # Do it manually
        batch, input_channels, height, width = inputs.shape
        new_height = (height - self.kernel_size) // self.stride + 1
        new_width = (width - self.kernel_size) // self.stride + 1
        output = torch.zeros(batch, input_channels, new_height, new_width)
        for i in range(new_height):
            for j in range(new_width):
                output[:, :, i, j] = torch.mean(
                    inputs[:, :, i : i + self.kernel_size, j : j + self.kernel_size],
                    dim=(2, 3),
                )
        return output
