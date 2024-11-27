import torch
import torch.nn.functional as F
from typing import Any
import torch.optim
import math
from torch.autograd import Variable, Function


class ReLUFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the ReLU.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward method of the relu.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO

        ctx.save_for_backward(inputs)

        inputs_Relu = inputs.clone()
        inputs_Relu[inputs_Relu <= 0] = 0

        return inputs_Relu

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        This method is the backward of the relu.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [*], same as the grad_output.
        """

        # TODO

        input = ctx.saved_tensors[0]
        grad_i = grad_output.clone()
        grad_i[input <= 0] = 0

        return grad_i


class ReLU(torch.nn.Module):
    """
    This is the class that represents the ReLU Layer.
    """

    def __init__(self):
        """
        This method is the constructor of the ReLU layer.
        """

        # call super class constructor
        super().__init__()

        self.fn = ReLUFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*] (same as the input).
        """

        return self.fn(inputs)


class LinearFunction(torch.autograd.Function):
    """
    This class implements the forward and backward of the Linear layer.
    """

    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """
        This method is the forward pass of the Linear layer.

        Args:
            ctx: contex for saving elements for the backward.
            inputs: inputs tensor. Dimensions:
                [batch, input dimension].
            weight: weights tensor.
                Dimensions: [output dimension, input dimension].
            bias: bias tensor. Dimensions: [output dimension].

        Returns:
            outputs tensor. Dimensions: [batch, output dimension].
        """

        # TODO
        ctx.save_for_backward(inputs, weight, bias)

        return inputs @ weight.t() + bias

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method is the backward for the Linear layer.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients.
                Dimensions: [batch, output dimension].

        Returns:
            tuple of gradients, gradients of the inputs, the weights
                and the bias, in respective order.
                Inputs gradients dimension: [batch, input dimension].
                Weights gradients dimension:
                [output dimension, input dimension].
                Bias gradients dimension: [output dimension].
        """

        # TODO

        inputs, weight, _ = ctx.saved_tensors
        grad_input = grad_output @ weight
        grad_weight = grad_output.t() @ inputs
        grad_bias = grad_output.sum(0)  # this is the same as grad_output.t() @ ones

        return grad_input, grad_weight, grad_bias


class Linear(torch.nn.Module):
    """
    This is the class that represents the Linear Layer.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        This method is the constructor of the Linear layer.
        The attributes must be named the same as the parameters of the
        linear layer in pytorch. The parameters should be initialized

        Args:
            input_dim: input dimension.
            output_dim: output dimension.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(output_dim, input_dim)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(output_dim))

        # init parameters corectly
        self.reset_parameters()

        # define layer function
        self.fn = LinearFunction.apply

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method if the forward pass of the layer.

        Args:
            inputs: inputs tensor. Dimenions: [batch, input dim].

        Returns:
            outputs tensor. Dimensions: [batch, output dim].
        """

        return self.fn(inputs, self.weight, self.bias)

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


def relu(ctx, x):
    ctx.save_for_backward(x)
    return torch.max(0.0, x)


def backwardRelu(ctx, grad_output):
    (x,) = ctx.saved_tensors
    return (x > 0).float() * grad_output


class LeakyReLUFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the ReLU.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        This is the forward method of the relu.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [*].

        Returns:
            outputs tensor. Dimensions: [*], same as inputs.
        """

        # TODO
        outputs: torch.Tensor = torch.where(inputs > 0, inputs, inputs * 0.01)
        ctx.save_for_backward(inputs)
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        This method is the backward of the relu.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [*], same as the grad_output.
        """

        # TODO
        (inputs,) = ctx.saved_tensors
        return torch.where(inputs > 0, grad_output, grad_output * 0.01)


def leakyReluFoward(ctx, x):
    ctx.save_for_backward(x)
    return torch.max(0.01 * x, x)


def leakyReluBackward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    return (x > 0).float() * grad_output + 0.01 * (x < 0).float() * grad_output


class ELUFunction(torch.autograd.Function):
    """
    This class implements the forward and backward passes of the Logistic layer.
    """

    @staticmethod
    def forward(ctx, inputs, alpha=1.0):
        """
        This method is the forward pass of the Logistic layer.

        Args:
            ctx: context for saving elements for the backward pass.
            inputs: input tensor with dimensions [batch, input dimension].
            weight: weights tensor with dimensions [output dimension, input dimension].
            bias: bias tensor with dimensions [output dimension].

        Returns:
            output tensor with dimensions [batch, output dimension].
        """
        # Compute the linear part
        # Apply the sigmoid activation

        elu_output = torch.where(inputs > 0, inputs, alpha * (torch.exp(inputs) - 1))

        # Save tensors for backward computation
        ctx.save_for_backward(inputs, torch.tensor(alpha))

        return elu_output

    @staticmethod
    def backward(ctx, grad_output):
        """
        This method is the backward pass for the Logistic layer.

        Args:
            ctx: context for loading elements from the forward pass.
            grad_output: gradient of the loss with respect to the output of this layer,
                         with dimensions [batch, output dimension].

        Returns:
            tuple of gradients for the inputs, the weights, and the bias, respectively:
            - Gradient of the inputs with dimensions [batch, input dimension].
            - Gradient of the weights with dimensions [output dimension, input dimension].
            - Gradient of the bias with dimensions [output dimension].
        """
        inputs, alpha = ctx.saved_tensors
        grad_input = grad_output * torch.where(
            inputs > 0, torch.ones_like(inputs), alpha * torch.exp(inputs)
        )
        return grad_input


def elu(ctx, x, alpha):
    ctx.save_for_backward(x, alpha)
    return (x > 0) * x + (x <= 0) * (alpha * (torch.exp(x) - 1))


def eluBackward(ctx, grad_output):
    x, alpha = ctx.saved_tensors
    return (x > 0) * grad_output + (x <= 0) * alpha * grad_output


def absolute_value(ctx, x):
    ctx.save_for_backward(x)
    return torch.abs(x)


def absolute_value_backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    return torch.sign(x) * grad_output


def maxout(ctx, x, num_pieces):
    ctx.save_for_backward(x, num_pieces)
    # spliteamos en la dimension 1 por el numero de piezas
    # dims = [batch, s.shape[1]//num_pieces, num_pieces, x.shape[2], x.shape[3]]
    # cogemos la pieza con el maximo valor (maximo de la dimension 2)
    #
    return x.view(
        x.shape[0], x.shape[1] // num_pieces, num_pieces, x.shape[2], x.shape[3]
    ).max(dim=2)[0]


class Maxout(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        x = input
        max_out = 4  # Maxout Parameter
        kernels = x.shape[1]  # to get how many kernels/output
        feature_maps = int(kernels / max_out)
        out_shape = (x.shape[0], feature_maps, max_out, x.shape[2], x.shape[3])
        x = x.view(out_shape)
        y, indices = torch.max(x[:, :, :], 2)
        ctx.save_for_backward(input)
        ctx.indices = indices
        ctx.max_out = max_out
        return y

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input1, indices, max_out = (
            ctx.saved_variables[0],
            Variable(ctx.indices),
            ctx.max_out,
        )
        input = input1.clone()
        for i in range(max_out):
            a0 = indices == i
            input[:, i : input.data.shape[1] : max_out] = a0.float() * grad_output

        return input


def forward_prelu(inputs: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    This is the forward method of the PReLU.

    Args:
        inputs: input tensor. Dimensions: [*].
        a: parameter of PReLU. Dimensions: [0].

    Returns:
        outputs tensor. Dimensions: [*], same as inputs.
    """

    # TODO

    return torch.where(inputs < 0.0, 0.0, inputs) + a * torch.where(
        0.0 < inputs, 0.0, inputs
    )


def backward_prelu(
    grad_output: torch.Tensor, inputs: torch.Tensor, a: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This method is the backward of the PReLU.

    Args:
        grad_output: outputs gradients. Dimensions: [*].
        inputs: input tensor. Dimensions: [*].
        a: parameter of PReLU. Dimensions: [0].

    Returns:
        inputs gradients. Dimensions: [*], same as the grad_output.
        a gradients. Dimensions: [0], same as the a parameter.
    """

    # TODO

    return grad_output * torch.where(inputs <= 0, 0, 1) + a * grad_output * torch.where(
        inputs <= 0, 1, 0
    ), grad_output * torch.where(0.0 < inputs, 0.0, inputs)


class PReLUFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the ReLU.
    """

    @staticmethod
    @torch.no_grad()
    def forward(ctx: Any, inputs: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        This is the forward method of the PReLU.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [*].
            a: parameter of PReLU. Dimensions: [0].

        Returns:
            outputs tensor. Dimensions: [*], same as inputs.
        """

        # save tensors for the backward
        ctx.save_for_backward(inputs, a)

        # compute forward
        outputs = forward_prelu(inputs, a)

        return outputs

    @staticmethod
    @torch.no_grad()
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        """
        This method is the backward of the PReLU.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [*], same as the grad_output.
            a gradients. Dimension: [0].
        """

        # load tensors from the forward
        inputs, a = ctx.saved_tensors

        grad_input: torch.Tensor
        grad_a: torch.Tensor
        grad_input, grad_a = backward_prelu(grad_output, inputs, a)

        return grad_input, grad_a


class PReLU(torch.nn.Module):
    """
    This is the class that represents the PReLU Layer.
    """

    def __init__(self, init: float = 0.25) -> None:
        """
        This method is the constructor of the PReLU layer.
        """

        # call super class constructor
        super().__init__()

        self.a: torch.nn.Parameter = torch.nn.Parameter(torch.tensor(init))

        self.fn = PReLUFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, *].

        Returns:
            outputs tensor. Dimensions: [*] (same as the input).
        """

        return self.fn(inputs, self.a)


class SigmoidFunction(torch.autograd.Function):
    """
    This is the class that represents the Sigmoid activation function.

    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the Sigmoid activation function.

        Args:
            ctx: the context.
            inputs: the input tensor. Dimension [*]

        Returns:
            the output tensor. Dimension [*], same as the input tensor.
        """
        # TODO. Hint: use ctx.save_for_backward() to save information for the backward pass
        output = 1 / (1 + torch.exp(-inputs))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method computes the backward pass of the Sigmoid activation function.

        Args:
            ctx: the context.
            grad_output: the gradient of the loss with respect to the output tensor. Dimension [*]

        Returns:
            the gradient of the loss with respect to the input tensor. Dimension [*], same as the input tensor.
        """
        # TODO. Hint: use ctx.saved_tensors to retrieve information saved in the forward pass
        (output,) = ctx.saved_tensors
        grad_input = output * (1 - output) * grad_output
        return grad_input


class Sigmoid(torch.nn.Module):
    """
    This is the class that represents the Sigmoid activation function.
    """

    def __init__(self):
        """
        This method is the constructor of the class.
        """
        super(Sigmoid, self).__init__()

        self.fn = SigmoidFunction.apply

    def forward(self, x):
        """
        This method computes the forward pass of the Sigmoid activation function.

        Args:
            x: the input tensor. Dimension [*]

        Returns:
            the output tensor. Dimension [*], same as the input tensor.
        """
        return self.fn(x)


class SoftmaxFunction(torch.autograd.Function):
    """
    This is the class that represents the Softmax activation function.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the Softmax activation function.

        Args:
            ctx: the context.
            inputs: the input tensor. Dimension [*]

        Returns:
            the output tensor. Dimension [*], same as the input tensor.
        """
        # TODO. Hint: use ctx.save_for_backward() to save information for the backward pass
        output = torch.exp(inputs) / torch.exp(inputs).sum(dim=-1, keepdim=True)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method computes the backward pass of the Softmax activation function.

        Args:
            ctx: the context.
            grad_output: the gradient of the loss with respect to the output tensor. Dimension [*]

        Returns:
            the gradient of the loss with respect to the input tensor. Dimension [*], same as the input tensor.
        """
        # TODO. Hint: use ctx.saved_tensors to retrieve information saved in the forward pass
        (output,) = ctx.saved_tensors
        grad_input = output * (
            grad_output - (output * grad_output).sum(dim=-1, keepdim=True)
        )
        return grad_input


class Softmax(torch.nn.Module):
    """
    This is the class that represents the Softmax activation function.
    """

    def __init__(self):
        """
        This method is the constructor of the class.
        """
        super(Softmax, self).__init__()

        self.fn = SoftmaxFunction.apply

    def forward(self, x):
        """
        This method computes the forward pass of the Softmax activation function.

        Args:
            x: the input tensor. Dimension [*]

        Returns:
            the output tensor. Dimension [*], same as the input tensor.
        """
        return self.fn(x)


class TanhFunction(torch.autograd.Function):
    """
    This is the class that represents the Tanh activation function.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forward pass of the Tanh activation function.

        Args:
            ctx: the context.
            inputs: the input tensor. Dimension [*]

        Returns:
            the output tensor. Dimension [*], same as the input tensor.
        """
        # TODO. Hint: use ctx.save_for_backward() to save information for the backward pass
        output = (torch.exp(inputs) - torch.exp(-inputs)) / (
            torch.exp(inputs) + torch.exp(-inputs)
        )
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        This method computes the backward pass of the Tanh activation function.

        Args:
            ctx: the context.
            grad_output: the gradient of the loss with respect to the output tensor. Dimension [*]

        Returns:
            the gradient of the loss with respect to the input tensor. Dimension [*], same as the input tensor.
        """
        # TODO. Hint: use ctx.saved_tensors to retrieve information saved in the forward pass

        (output,) = ctx.saved_tensors
        grad_input = (1 - output**2) * grad_output
        return grad_input


class Tanh(torch.nn.Module):
    """
    This is the clas that represents the Tanh activation function.
    """

    def __init__(self):
        """
        This method is the constructor of the class.
        """
        super(Tanh, self).__init__()

        self.fn = TanhFunction.apply

    def forward(self, x):
        """
        This method computes the forward pass of the Tanh activation function.

        Args:
            x: the input tensor. Dimension [*]

        Returns:
            the output tensor. Dimension [*], same as the input tensor.
        """
        return self.fn(x)


class SoftplusFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.exp().log1p()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        sigmoid = 1 / (1 + (-input).exp())
        return grad_output * sigmoid


class Softplus(torch.nn.Module):
    def __init__(self):
        super(Softplus, self).__init__()

    def forward(self, input):
        return SoftplusFunction.apply(input)


def get_dropout_random_indexes(shape: torch.Size, p: float) -> torch.Tensor:
    """
    This function get the indexes to put elements at zero for the
    dropout layer. It ensures the elements are selected following the
    same implementation than the pytorch layer.

    Args:
        shape: shape of the inputs to put it at zero. Dimensions: [*].
        p: probability of the dropout.

    Returns:
        indexes to put elements at zero in dropout layer.
            Dimensions: shape.
    """

    # TODO

    # get inputs indexes
    inputs: torch.Tensor = torch.ones(shape)

    # get indexes
    indexes: torch.Tensor = F.dropout(inputs, p)
    indexes = (indexes == 0).int()

    return indexes


class Dropout(torch.nn.Module):
    """
    This the Dropout class.

    Attr:
        p: probability of the dropout.
        inplace: indicates if the operation is done in-place.
            Defaults to False.
    """

    def __init__(self, p: float, inplace: bool = False) -> None:
        """
        This function is the constructor of the Dropout class.

        Args:
            p: probability of the dropout.
            inplace: if the operation is done in place.
                Defaults to False.
        """

        # TODO

        # call super class constructor
        super(Dropout, self).__init__()

        # set attributes
        self.p = p
        self.inplace = inplace

    def forward(self, ctx, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 1:
            # En modo de evaluación, devolver los inputs directamente
            return inputs

        # En modo de entrenamiento, aplicar dropout usando get_dropout_random_indexes
        # how to do drop_out_indexes:
        # FORMA 1
        vector_aleatorio = torch.rand(inputs.shape)
        dropout_mask = (vector_aleatorio < self.p).int()  # mascara de 0 y 1
        # FORMA 2
        # dropout_mask = (get_dropout_random_indexes(inputs.shape, self.p)==0)

        ctx.save_for_backward(dropout_mask / (1 - self.p))
        outputs = dropout_mask * inputs
        outputs = outputs / (1 - self.p)

        if self.inplace:
            inputs.data = outputs

        return outputs

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        # Recuperar la máscara de dropout guardada en el forward
        dropout_mask = self.saved_tensors[0]
        # Aplicar la máscara de dropout al grad_output
        grad_input = dropout_mask * grad_output
        return grad_input
