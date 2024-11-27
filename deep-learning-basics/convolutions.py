import torch
import torch.nn.functional as F
import torch.optim
import math


class Conv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0):
        # Guardamos para el backward
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        batch_size, in_channels, width = input.shape
        out_channels, _, kernel_width = weight.shape

        # Aplicando padding
        padded_input = F.pad(input, (padding, padding), "constant", 0)

        # Calculamos la nueva anchura después del padding y el stride
        output_width = (width + 2 * padding - kernel_width) // stride + 1

        # Salida inicializada
        output = torch.zeros((batch_size, out_channels, output_width))

        # Realizamos la convolución
        for i in range(output_width):
            start = i * stride
            end = start + kernel_width

            input_slice = padded_input[:, :, start:end]
            for j in range(out_channels):
                output[:, j, i] = (input_slice * weight[j, :, :]).sum(
                    dim=(1, 2)
                ) + bias[j]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride, padding = ctx.stride, ctx.padding

        batch_size, in_channels, width = input.shape
        out_channels, _, kernel_width = weight.shape

        # Aplicando padding
        padded_input = F.pad(input, (padding, padding), "constant", 0)

        # Calculamos la nueva anchura después del padding y el stride
        output_width = (width + 2 * padding - kernel_width) // stride + 1

        # Inicializamos gradientes
        grad_input = torch.zeros_like(padded_input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)

        # Realizamos el backward
        for i in range(output_width):
            start = i * stride
            end = start + kernel_width

            # Para cada posición en la salida
            for j in range(out_channels):
                # Calculamos gradiente para el peso (grad_weight)
                input_slice = padded_input[:, :, start:end]
                grad_weight[j, :, :] += (
                    grad_output[:, j, i].unsqueeze(1).unsqueeze(2) * input_slice
                ).sum(0)

                # Calculamos gradiente para la entrada (grad_input)
                grad_input[:, :, start:end] += (
                    grad_output[:, j, i].unsqueeze(1).unsqueeze(2) * weight[j, :, :]
                )

                # Calculamos gradiente para el sesgo (grad_bias)
                grad_bias[j] += grad_output[:, j, i].sum()

        # Ajuste final para grad_input para eliminar el padding aplicado al inicio
        if padding > 0:
            grad_input = grad_input[:, :, padding:-padding]
        else:
            grad_input = grad_input

        return grad_input, grad_weight, grad_bias, None, None


class Conv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size)
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, input):
        return Conv1dFunction.apply(
            input, self.weight, self.bias, self.stride, self.padding
        )


class Conv2dFunction(torch.autograd.Function):
    """
    Class to implement the forward and backward methods of the Conv2d
    layer.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        padding: int,
        stride: int,
    ) -> torch.Tensor:
        """
        This function is the forward method of the class.

        Args:
            ctx: context for saving elements for the backward.
            inputs: inputs for the model. Dimensions: [batch,
                input channels, height, width].
            weight: weight of the layer.
                Dimensions: [output channels, input channels,
                kernel size, kernel size].
            bias: bias of the layer. Dimensions: [output channels].

        Returns:
            output of the layer. Dimensions:
                [batch, output channels,
                (height + 2*padding - kernel size) / stride + 1,
                (width + 2*padding - kernel size) / stride + 1]
        """

        # Compute the output dimensions
        _, _, in_height, in_width = inputs.shape
        out_channels, _, kernel_size, _ = weight.shape
        out_height = (in_height + 2 * padding - kernel_size) // stride + 1
        out_width = (in_width + 2 * padding - kernel_size) // stride + 1

        # Save tensors for backward pass
        ctx.save_for_backward(
            inputs, weight, bias, torch.tensor(stride), torch.tensor(padding)
        )

        # Unfold the input, each column is a receptive field
        inputs_unfold = torch.nn.functional.unfold(
            inputs, kernel_size, padding=padding, stride=stride
        )

        # Reshape the weight, each row is a filter
        weight_reshaped = weight.view(out_channels, -1)

        # Compute and fold the output
        output = weight_reshaped @ inputs_unfold + bias.view(-1, 1)

        return torch.nn.functional.fold(output, (out_height, out_width), (1, 1))

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        # Load tensors from the forward pass
        inputs, weight, bias, stride, padding = ctx.saved_tensors
        stride, padding = stride.item(), padding.item()
        _, _, in_height, in_width = inputs.shape
        out_channels, _, kernel_size, _ = weight.shape

        # Compute gradients
        grad_inputs = grad_weight = grad_bias = None

        # Compute gradients if needed
        if ctx.needs_input_grad[0]:
            # Unfold the grad_output
            grad_output_unfolded = torch.nn.functional.unfold(grad_output, (1, 1))
            weight_reshaped = weight.view(out_channels, -1)
            grad_inputs_unfolded = weight_reshaped.t() @ grad_output_unfolded
            grad_inputs = torch.nn.functional.fold(
                grad_inputs_unfolded,
                (in_height, in_width),
                kernel_size,
                padding=padding,
                stride=stride,
            )
        if ctx.needs_input_grad[1]:
            # Unfold the inputs
            inputs_unfolded = torch.nn.functional.unfold(
                inputs, kernel_size, padding=padding, stride=stride
            )
            grad_weight_unfolded = inputs_unfolded @ grad_output_unfolded.transpose(
                1, 2
            )
            grad_weight_unfolded = grad_weight_unfolded.transpose(1, 2).sum(dim=0)
            grad_weight = grad_weight_unfolded.view(
                out_channels, -1, kernel_size, kernel_size
            )
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_inputs, grad_weight, grad_bias, None, None


class Conv2d(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        """
        This method is the constructor of the Linear layer. Follow the
        pytorch convention.

        Args:
            input_channels: input dimension.
            output_channels: output dimension.
            kernel_size: kernel size to use in the convolution.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_channels, input_channels, kernel_size, kernel_size)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(output_channels))
        self.padding = padding
        self.stride = stride

        # init parameters corectly
        self.reset_parameters()

        # define layer function
        self.fn = Conv2dFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: inputs tensor. Dimensions: [batch, input channels,
                output channels, height, width].

        Returns:
            outputs tensor. Dimensions: [batch, output channels,
                height - kernel size + 1, width - kernel size + 1].
        """

        return self.fn(inputs, self.weight, self.bias, self.padding, self.stride)

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return None


def unfold_2d(input, kernel_size, padding, stride, dilation):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    batch, in_channels, height, width = input.size()

    input = torch.nn.functional.pad(
        input, (padding[1], padding[1], padding[0], padding[0])
    )

    unfolded_height = (
        height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    unfolded_width = (
        width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1

    unfolded = torch.zeros(
        batch,
        in_channels * kernel_size[0] * kernel_size[1],
        unfolded_height * unfolded_width,
        dtype=input.dtype,
        device=input.device,
    )

    idx = 0
    for y in range(unfolded_height):
        for x in range(unfolded_width):
            block = input[
                :,
                :,
                y : y + (kernel_size[0] - 1) * dilation[0] + 1 : dilation[0],
                x : x + (kernel_size[1] - 1) * dilation[1] + 1 : dilation[1],
            ].reshape(batch, -1)
            unfolded[:, :, idx] = block
            idx += 1

    return unfolded


def fold_2d(unfolded, output_size, kernel_size, padding, stride, dilation):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # Tamaño esperado de entrada y cálculo de dimensiones de salida
    batch_size, channels_X_kernel, L = unfolded.size()
    in_channels = channels_X_kernel // (kernel_size[0] * kernel_size[1])
    output_height, output_width = output_size

    unfolded_height = (
        output_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    unfolded_width = (
        output_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1

    assert (
        L == unfolded_height * unfolded_width
    ), "L must be equal to unfolded_height * unfolded_width"

    output = torch.zeros(
        (
            batch_size,
            in_channels,
            output_height + 2 * padding[0],
            output_width + 2 * padding[1],
        ),
        dtype=unfolded.dtype,
        device=unfolded.device,
    )

    idx = 0
    for y in range(unfolded_height):
        for x in range(unfolded_width):
            patch = unfolded[:, :, idx].view(
                batch_size, in_channels, kernel_size[0], kernel_size[1]
            )
            output[
                :,
                :,
                y : y + (kernel_size[0] - 1) * dilation[0] + 1 : dilation[0],
                x : x + (kernel_size[1] - 1) * dilation[1] + 1 : dilation[1],
            ] = patch
            idx += 1

    if padding[0] > 0 or padding[1] > 0:
        output = output[
            :,
            :,
            padding[0] : output_height + padding[0],
            padding[1] : output_width + padding[1],
        ]

    return output


class Conv3dFunction(torch.autograd.Function):
    """
    Class to implement the forward and backward methods of the Conv2d
    layer.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        padding: int,
        stride: int,
    ) -> torch.Tensor:
        """
        This function is the forward method of the class.

        Args:
            ctx: context for saving elements for the backward.
            inputs: inputs for the model. Dimensions: [batch,
                input channels, depth, height, width].
            weight: weight of the layer.
                Dimensions: [output channels, input channels,
                kernel size, kernel size, kernel size].
            bias: bias of the layer. Dimensions: [output channels].

        Returns:
            output of the layer. Dimensions:
                [batch, output channels,
                depth_out,
                height_out,
                width_out]
        """
        # Save for backward
        ctx.save_for_backward(inputs, weight, bias, torch.Tensor([padding, stride]))

        # Get all dimension values
        _, _, depth, height, width = inputs.shape
        output_channels, _, kernel_size, _, _ = weight.shape

        # Unfold input. New dimension [B, Ci·K^3, L]
        inputs_unfold = unfold_3d(
            inputs,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        # Reshape weight. New dimension [1, Ci·K^3, Co]
        weight_unfold = unfold_3d(weight, kernel_size=kernel_size).transpose(0, 2)

        # Outputs unfold. Dimension [B, Co, L]
        output_unfold = torch.matmul(weight_unfold.transpose(1, 2), inputs_unfold)

        output_unfold = output_unfold + bias.unsqueeze(0).unsqueeze(2)

        # Fold output
        output_size = (
            (depth + 2 * padding - kernel_size) // stride + 1,
            (height + 2 * padding - kernel_size) // stride + 1,
            (width + 2 * padding - kernel_size) // stride + 1,
        )
        output: torch.Tensor = fold_3d(
            output_unfold,
            output_size,
            kernel_size=1,
            padding=padding,
            stride=stride,
        )

        return output

    @staticmethod
    def backward(  # type: ignore
        ctx, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None, None
    ]:
        # Obtain context items
        inputs, weight, _, padding_stride = ctx.saved_tensors
        padding, stride = int(padding_stride[0].item()), int(padding_stride[1].item())

        # Get all dimension values
        _, _, depth, height, width = inputs.shape
        output_channels, input_channels, kernel_size, _, _ = weight.shape

        # Unfold and reshape inputs and weights as in forward
        # Unfold input. New dimension [B, Ci·K^3, L]
        inputs_unfold = unfold_3d(
            inputs,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        # Reshape weight. New dimension [1, Ci·K^3, Co]
        weight_unfold = unfold_3d(weight, kernel_size=kernel_size).transpose(0, 2)

        # Unfold grad_output. New dimension [B, Co, L]
        grad_output_unfold = unfold_3d(
            grad_output,
            kernel_size=1,
            padding=padding,
            stride=stride,
        )

        # Grad_inputs unfold. Dimension [B, Ci·K^3, L]
        grad_inputs_unfold = torch.matmul(weight_unfold, grad_output_unfold)

        # Grad_weight unfold. Dimension [Co, Ci·K^3]
        grad_weight_unfold = (
            torch.matmul(inputs_unfold, grad_output_unfold.transpose(1, 2)).sum(0).T
        )

        # Grad_inputs. Dimension same as input.
        grad_inputs: torch.Tensor = fold_3d(
            grad_inputs_unfold,
            output_size=(depth, height, width),
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )

        # Grad_weight. Dimension same as weight.
        grad_weight: torch.Tensor = grad_weight_unfold.view(
            output_channels, input_channels, kernel_size, kernel_size, kernel_size
        )

        # Grad_bias. Dimension same as bias.
        grad_bias: torch.Tensor = grad_output.sum(dim=(0, 2, 3, 4))

        return grad_inputs, grad_weight, grad_bias, None, None


class Conv3d(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
    ) -> None:
        """
        This method is the constructor of the Linear layer. Follow the
        pytorch convention.

        Args:
            input_channels: input dimension.
            output_channels: output dimension.
            kernel_size: kernel size to use in the convolution.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(
                output_channels, input_channels, kernel_size, kernel_size, kernel_size
            )
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(output_channels))
        self.padding = padding
        self.stride = stride

        # init parameters corectly
        self.reset_parameters()

        # define layer function
        self.fn = Conv3dFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: inputs tensor. Dimensions: [batch, input channels,
                output channels, height, width].

        Returns:
            outputs tensor. Dimensions: [batch, output channels,
                height - kernel size + 1, width - kernel size + 1].
        """

        return self.fn(inputs, self.weight, self.bias, self.padding, self.stride)

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        # init parameters the correct way
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        return None


def unfold_3d(input, kernel_size, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    batch, in_channels, depth, height, width = input.size()

    input = torch.nn.functional.pad(
        input, (padding[2], padding[2], padding[1], padding[1], padding[0], padding[0])
    )

    unfolded_depth = (
        depth + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    unfolded_height = (
        height + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1
    unfolded_width = (
        width + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1
    ) // stride[2] + 1

    # Inicialización del tensor desplegado
    unfolded = torch.zeros(
        batch,
        in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2],
        unfolded_depth * unfolded_height * unfolded_width,
        dtype=input.dtype,
        device=input.device,
    )

    # Índice para el tensor desplegado
    idx = 0
    for z in range(unfolded_depth):
        for y in range(unfolded_height):
            for x in range(unfolded_width):
                block = input[
                    :,
                    :,
                    z : z + (kernel_size[0] - 1) * dilation[0] + 1 : dilation[0],
                    y : y + (kernel_size[1] - 1) * dilation[1] + 1 : dilation[1],
                    x : x + (kernel_size[2] - 1) * dilation[2] + 1 : dilation[2],
                ].reshape(batch, -1)
                unfolded[:, :, idx] = block
                idx += 1

    return unfolded


def fold_3d(unfolded, output_size, kernel_size, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    batch_size, channels_X_kernel, L = unfolded.size()
    in_channels = channels_X_kernel // (
        kernel_size[0] * kernel_size[1] * kernel_size[2]
    )
    output_depth, output_height, output_width = output_size

    unfolded_depth = (
        output_depth + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    unfolded_height = (
        output_height + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1
    unfolded_width = (
        output_width + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1
    ) // stride[2] + 1

    assert (
        L == unfolded_depth * unfolded_height * unfolded_width
    ), "L must be equal to unfolded_depth * unfolded_height * unfolded_width"

    output = torch.zeros(
        (
            batch_size,
            in_channels,
            output_depth + 2 * padding[0],
            output_height + 2 * padding[1],
            output_width + 2 * padding[2],
        ),
        dtype=unfolded.dtype,
        device=unfolded.device,
    )

    idx = 0
    for z in range(unfolded_depth):
        for y in range(unfolded_height):
            for x in range(unfolded_width):
                patch = unfolded[:, :, idx].view(
                    batch_size,
                    in_channels,
                    kernel_size[0],
                    kernel_size[1],
                    kernel_size[2],
                )
                output[
                    :,
                    :,
                    z : z + (kernel_size[0] - 1) * dilation[0] + 1 : dilation[0],
                    y : y + (kernel_size[1] - 1) * dilation[1] + 1 : dilation[1],
                    x : x + (kernel_size[2] - 1) * dilation[2] + 1 : dilation[2],
                ] = patch
                idx += 1

    if padding[0] > 0 or padding[1] > 0 or padding[2] > 0:
        output = output[
            :,
            :,
            padding[0] : output_depth + padding[0],
            padding[1] : output_height + padding[1],
            padding[2] : output_width + padding[2],
        ]

    return output
