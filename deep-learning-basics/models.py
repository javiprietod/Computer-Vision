import torch
import torch.nn.functional as F
from typing import Optional, Any
import torch.optim
import math
import torch.nn as nn


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


# RESIDUAL -----------------------------------------------------------------------------------------------
class ResidualFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the Residual.
    """

    @staticmethod
    @torch.no_grad()
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """
        This is the forward method of the residual.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, input dimension].
            weight: weights tensor.
                Dimensions: [input dimension, input dimension].
            bias: bias tensor. Dimensions: [input dimension].
            a: parameter of PReLU. Dimensions: [0].

        Returns:
            outputs tensor. Dimensions: [batch, input dimension].
        """

        # save elements for backward
        ctx.save_for_backward(inputs, weight, bias, a)

        # TODO
        batch, _ = inputs.shape
        output_dim, _ = weight.shape
        outputs = torch.zeros(batch, output_dim)
        outputs = inputs @ weight.t() + bias

        outputs = forward_prelu(outputs, a)

        outputs += inputs

        return outputs

    @staticmethod
    @torch.no_grad()
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore
        """
        This method is the backward of the residual.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients.
                Dimensions: [batch, input dimension].

        Returns:
            inputs gradients. Dimension: [batch, input dimension].
            weights gradients.
                Dimension: [input dimension, input dimension].
            bias gradients. Dimension: [input dimension].
            a gradients. Dimension: [0].
        """

        # load elements from forward
        inputs, weight, bias, a = ctx.saved_tensors
        # weight: [Co, Ci]
        # TODO
        batch, _ = inputs.shape
        output_dim, _ = weight.shape
        outputs_L = torch.zeros(batch, output_dim)
        outputs_L = inputs @ weight.t() + bias

        grad_output_prelu, grad_a = backward_prelu(grad_output, outputs_L, a)

        grad_inputs = grad_output_prelu @ weight
        grad_inputs += grad_output

        # g_o_p [B, Co]; inputs [B, Ci] -> [Co, Ci]
        grad_weight = grad_output_prelu.t() @ inputs

        batch, _ = grad_output_prelu.shape
        aux_bias = torch.ones(1, batch, dtype=torch.float)
        grad_bias = (aux_bias @ grad_output_prelu).view(3)

        return grad_inputs, grad_weight, grad_bias, grad_a


class Residual(torch.nn.Module):
    """
    This is the class that represents the Residual Layer.
    """

    def __init__(self, input_dim: int, init: float = 0.25) -> None:
        """
        This method is the constructor of the Residual layer.

        Args:
            input_dim: input dimension.
            a: parameter of prelu.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(input_dim, input_dim)
        )
        self.bias: torch.nn.Parameter = torch.nn.Parameter(torch.empty(input_dim))
        self.a: torch.nn.Parameter = torch.nn.Parameter(torch.tensor(init))

        # init parameters corectly
        self.reset_parameters()

        self.fn = ResidualFunction.apply

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, input dimension].

        Returns:
            outputs tensor. Dimensions: [batch, input dimension].
        """

        return self.fn(inputs, self.weight, self.bias, self.a)

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


# RNN ---------------------------------------------------------------------------------------------------
class RNNFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the RNN.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is the forward method of the RNN.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, sequence,
                input size].
            h0: first hidden state. Dimensions: [1, batch,
                hidden size].
            weight_ih: weight for the inputs.
                Dimensions: [hidden size, input size].
            weight_hh: weight for the inputs.
                Dimensions: [hidden size, hidden size].
            bias_ih: bias for the inputs.
                Dimensions: [hidden size].
            bias_hh: bias for the inputs.
                Dimensions: [hidden size].


        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        # Get dimensions
        batch, sequence, _ = inputs.size()

        # Initialize outputs
        outputs = torch.empty(
            batch, sequence, h0.size(-1), device=inputs.device, dtype=inputs.dtype
        )

        # Initialize hidden state
        hn = h0.squeeze(0)

        # Loop over the sequence
        for t in range(sequence):
            # Get input
            x = inputs[:, t, :]

            # Calculate the hidden state
            hn = x @ weight_ih.t() + hn @ weight_hh.t() + bias_ih + bias_hh

            # Apply the activation function
            hn = hn * (hn > 0).float()

            # Save the hidden state
            outputs[:, t, :] = hn

        # Save elements for the backward
        ctx.save_for_backward(
            inputs, h0, weight_ih, weight_hh, bias_ih, bias_hh, outputs, hn.unsqueeze(0)
        )

        return outputs, hn.unsqueeze(0)

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor, grad_hn: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        This method is the backward of the RNN.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [batch, sequence,
                input size].
            h0 gradients state. Dimensions: [1, batch,
                hidden size].
            weight_ih gradient. Dimensions: [hidden size,
                input size].
            weight_hh gradients. Dimensions: [hidden size,
                hidden size].
            bias_ih gradients. Dimensions: [hidden size].
            bias_hh gradients. Dimensions: [hidden size].
        """

        # Load elements from the forward
        (
            inputs,
            h0,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            outputs,
            hn,
        ) = ctx.saved_tensors

        # Get dimensions
        batch, sequence, _ = inputs.size()

        # Initialize gradients
        grad_inputs = torch.zeros_like(inputs)
        grad_h0 = torch.zeros_like(h0)
        grad_weight_ih = torch.zeros_like(weight_ih)
        grad_weight_hh = torch.zeros_like(weight_hh)
        grad_bias_ih = torch.zeros_like(bias_ih)
        grad_bias_hh = torch.zeros_like(bias_hh)

        # Initialize gradients for the hidden state
        grad_hn = grad_hn.squeeze(0)

        # Loop over the sequence
        for t in range(sequence - 1, -1, -1):
            # Get input
            x = inputs[:, t, :]

            # Get hidden state
            hn = outputs[:, t, :]

            hant = h0.squeeze(0) if t == 0 else outputs[:, t - 1, :]

            # Get gradients
            grad_hn = grad_hn + grad_output[:, t, :]

            # Calculate the gradients
            grad_hn = grad_hn * (hn > 0).float()

            # Calculate the gradients for the input
            grad_inputs[:, t, :] = grad_hn @ weight_ih

            # Calculate the gradients for the weights
            grad_weight_ih = grad_weight_ih + grad_hn.t() @ x
            grad_weight_hh = grad_weight_hh + grad_hn.t() @ hant

            # Calculate the gradients for the biases
            grad_bias_ih = grad_bias_ih + grad_hn
            grad_bias_hh = grad_bias_hh + grad_hn

            # Calculate the gradients for the hidden state
            grad_hn = grad_hn @ weight_hh

        # Calculate the gradients for the initial hidden state
        grad_h0 = grad_hn.unsqueeze(0)

        return (
            grad_inputs,
            grad_h0,
            grad_weight_ih,
            grad_weight_hh,
            grad_bias_ih,
            grad_bias_hh,
        )


class RNN(torch.nn.Module):
    """
    This is the class that represents the RNN Layer.
    """

    def __init__(self, input_dim: int, hidden_size: int):
        """
        This method is the constructor of the RNN layer.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.hidden_size = hidden_size
        self.weight_ih: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, input_dim)
        )
        self.weight_hh: torch.Tensor = torch.nn.Parameter(
            torch.empty(hidden_size, hidden_size)
        )
        self.bias_ih: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))
        self.bias_hh: torch.Tensor = torch.nn.Parameter(torch.empty(hidden_size))

        # init parameters corectly
        self.reset_parameters()

        self.fn = RNNFunction.apply

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the class.

        Args:
            inputs: inputs tensor. Dimensions: [batch, sequence,
                input size].
            h0: initial hidden state.

        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        return self.fn(
            inputs, h0, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh
        )

    def reset_parameters(self) -> None:
        """
        This method initializes the parameters in the correct way.
        """

        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

        return None


# DeepRNN ---------------------------------------------------------------------------------------------------
class DeepRNN(torch.nn.Module):
    """
    Deep Recurrent Neural Network (DeepRNN) class that supports multiple layers.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        """
        Initializes the DeepRNN model with the specified number of layers and configuration.

        Args:
            input_size (int): The number of expected features in the input `x`.
            hidden_size (int): The number of features in the hidden state `h`.
            num_layers (int): Number of recurrent layers.
        """
        super(DeepRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialize parameters for input-hidden and hidden-hidden connections
        self.weights_ih_0 = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weights_ih_n = torch.nn.Parameter(
            torch.Tensor(num_layers - 1, hidden_size, hidden_size)
        )

        self.weights_hh = torch.nn.Parameter(
            torch.Tensor(num_layers, hidden_size, hidden_size)
        )

        self.biases_ih = torch.nn.Parameter(torch.Tensor(num_layers, hidden_size))

        self.biases_hh = torch.nn.Parameter(torch.Tensor(num_layers, hidden_size))

        # Initialize parameters
        self.reset_parameters()

        self.fn = DeepRNNFunction.apply

    def reset_parameters(self):
        """
        Initialize parameters using the Xavier uniform method.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            torch.nn.init.uniform_(w, -stdv, stdv)

    def forward(self, input: torch.Tensor, h0: Optional[torch.Tensor] = None):
        """
        Defines the forward pass of the DeepRNN.

        Args:
            input (torch.Tensor): Tensor containing the features of the input sequence.
            h0 (Optional[torch.Tensor]): Tensor containing the initial hidden state for each element in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple (output, h_n), where:
            output (torch.Tensor) is the output features (h_t) from the last layer of the RNN, for each t.
            h_n (torch.Tensor) is the hidden state for t = seq_len.
        """
        if h0 is None:
            h0 = torch.zeros(
                self.num_layers,
                input.size(0),
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )

        return self.fn(
            input,
            h0,
            self.weights_ih_0,
            self.weights_ih_n,
            self.weights_hh,
            self.biases_ih,
            self.biases_hh,
        )


class DeepRNNFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the RNN.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        weight_lih_0: torch.Tensor,
        weight_lih_n: torch.Tensor,
        weight_lhh: torch.Tensor,
        bias_lih: torch.Tensor,
        bias_lhh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is the forward method of the RNN.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, sequence,
                input size].
            h0: first hidden state. Dimensions: [hidden_layers, batch,
                hidden size].
            weight_ih: weight for the inputs.
                Dimensions: [hidden size, input size].
            weight_lih: weight for the previous hidden.
                Dimensions: [hidden_layers-1, hidden size, input size].
            weight_lhh: weight for the inputs.
                Dimensions: [hidden_layer, hidden size, hidden size].
            bias_lih: bias for the inputs.
                Dimensions: [hidden_layer, hidden size].
            bias_lhh: bias for the inputs.
                Dimensions: [hidden_layer, hidden size].


        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        # TODO
        batch, sequence, _ = inputs.shape
        hidden_layers = len(weight_lhh)
        _, batch, hidden_size = h0.shape

        # Initialize outputs (for all layers, at the end we only return last layer)
        # Dimensions: [num_layers, B, S, hidden_size]
        outputs = torch.zeros(
            hidden_layers, batch, sequence, hidden_size, device=inputs.device
        ).double()

        # Initialize h_n
        h_n = torch.zeros(
            hidden_layers, batch, hidden_size, device=inputs.device
        ).double()

        # Initialize mask tensor for the relu (for backward)
        # Dimensions: same as outputs
        mask_tensor = torch.zeros(
            hidden_layers, batch, sequence, hidden_size, device=inputs.device
        )

        # Iterate over layers
        for i in range(hidden_layers):
            # Dimensions: [B, hidden_size]
            hidden_new = h0[i]

            # Iterate over sequence
            for h in range(sequence):
                # If first layer, compute with inputs, else with previous layer
                if i == 0:
                    hidden_new = (
                        torch.matmul(hidden_new, weight_lhh[i].t())
                        + bias_lhh[i]
                        + torch.matmul(inputs[:, h, :], weight_lih_0.t())
                        + bias_lih[i]
                    )
                else:
                    hidden_new = (
                        torch.matmul(hidden_new, weight_lhh[i].t())
                        + bias_lhh[i]
                        + torch.matmul(outputs[i - 1, :, h, :], weight_lih_n[i - 1].t())
                        + bias_lih[i]
                    )

                # Mask tensor for the relu (for backward)
                # Dimensions: [B, hidden_size]
                mask = hidden_new > 0
                mask_tensor[i, :, h, :] = mask

                # Apply the relu
                hidden_new = hidden_new * mask

                # Save outputs for later use
                outputs[i, :, h, :] = hidden_new

            # Save h_n
            h_n[i] = hidden_new

        ctx.save_for_backward(inputs, h0, outputs)
        ctx.mask_tensor = mask_tensor
        ctx.weight_lih_0 = weight_lih_0
        ctx.weight_lih_n = weight_lih_n
        ctx.weight_lhh = weight_lhh

        return outputs[-1, :, :, :], h_n

    @staticmethod
    def backward(  # type: ignore
        ctx: Any, grad_output: torch.Tensor, grad_hn: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        This method is the backward of the RNN.

        Args:
            ctx: context for loading elements from the forward.
            grad_output: outputs gradients. Dimensions: [*].

        Returns:
            inputs gradients. Dimensions: [batch, sequence,
                input size].
            h0 gradients state. Dimensions: [hidden_layers, 1, batch,
                hidden size].
            weight_lih gradient. Dimensions: [hidden_layers, hidden size,
                x] with x = input_size if i == 0 else hidden_size.
            weight_lhh gradients. Dimensions: [hidden_layers, hidden size,
                hidden size].
            bias_lih gradients. Dimensions: [hidden_layers, hidden size].
            bias_lhh gradients. Dimensions: [hidden_layers, hidden size].
        """

        # TODO
        # Get all saved tensors
        inputs, h0, outputs = ctx.saved_tensors
        weight_lih_0 = ctx.weight_lih_0
        weight_lih_n = ctx.weight_lih_n
        weight_lhh = ctx.weight_lhh
        mask_tensor = ctx.mask_tensor

        # Get all sizes
        hidden_layers = len(weight_lhh)
        hidden_size, _ = weight_lhh[0].shape
        batch, sequence, inputs_size = inputs.shape

        # Define gradients
        grad_inputs = torch.zeros_like(
            inputs, dtype=torch.float64, device=inputs.device
        )
        grad_h_inputs = torch.zeros(
            batch, sequence, hidden_size, dtype=torch.float64, device=inputs.device
        )
        grad_h0 = torch.zeros_like(h0, dtype=torch.float64, device=inputs.device)
        grad_weight_lih_0 = torch.zeros(hidden_size, inputs_size)
        grad_weight_lih_n = torch.zeros(hidden_layers - 1, hidden_size, hidden_size)
        grad_weight_lhh = torch.zeros(hidden_layers, hidden_size, hidden_size)
        grad_bias_lih = torch.zeros(hidden_layers, hidden_size)
        grad_bias_lhh = torch.zeros(hidden_layers, hidden_size)

        grad_output_l = grad_output.clone()

        # Iterate through the layers
        for i in range(hidden_layers - 1, -1, -1):
            # Iterate through the sequence
            grad_h_it = grad_hn[i]
            if i == 0:
                curr_inputs = inputs
            else:
                curr_inputs = outputs[i - 1, :, :, :]

            for h in range(sequence - 1, -1, -1):
                # Mask tensor for the relu (for backward)
                mask_t = mask_tensor[i, :, h, :]

                # Compute the gradient of current hidden before ReLu
                if h == sequence - 1:
                    grad_h_it = grad_output_l[:, h, :] * mask_t
                else:
                    grad_h_it = grad_h_it + grad_output_l[:, h, :] * mask_t

                # Grad input
                if i == 0:
                    grad_inputs[:, h, :] = torch.matmul(grad_h_it, weight_lih_0)
                else:
                    grad_h_inputs[:, h, :] = torch.matmul(
                        grad_h_it, weight_lih_n[i - 1]
                    )

                # Grad weights_lih
                if i == 0:
                    grad_weight_lih_0 = grad_weight_lih_0 + torch.matmul(
                        grad_h_it.t(), curr_inputs[:, h, :]
                    )
                else:
                    grad_weight_lih_n[i - 1] = grad_weight_lih_n[i - 1] + torch.matmul(
                        grad_h_it.t(), curr_inputs[:, h, :]
                    )

                # Grad weights_lhh
                h_ant = outputs[i, :, h - 1, :] if h > 0 else h0[i]
                grad_weight_lhh[i] = grad_weight_lhh[i] + torch.matmul(
                    grad_h_it.t(), h_ant
                )

                # Grad bias_lih
                grad_bias_lih[i] = grad_bias_lih[i] + grad_h_it.sum(0)

                # Grad bias_lhh
                grad_bias_lhh[i] = grad_bias_lhh[i] + grad_h_it.sum(0)

                # Grad h_h-1
                grad_h_it = torch.matmul(grad_h_it, weight_lhh[i])

                # Grad output for next layer
                grad_output_l[:, h, :] = grad_h_inputs[:, h, :].clone()

            grad_h0[i] = grad_h_it

        return (
            grad_inputs,
            grad_h0,
            grad_weight_lih_0,
            grad_weight_lih_n,
            grad_weight_lhh,
            grad_bias_lih,
            grad_bias_lhh,
        )


class DeepRNN_LOOP(torch.nn.Module):
    """
    This is the class that represents the Deep RNN.
    """

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int):
        """
        This method is the constructor of the Deep RNN.
        """

        # call super class constructor
        super().__init__()

        # define attributes
        self.num_layers = num_layers

        # create a list to hold the layers
        self.rnn_layers = torch.nn.ModuleList()

        # add the first layer
        self.rnn_layers.append(RNN(input_dim, hidden_size))

        # add the rest of the layers
        for _ in range(1, num_layers):
            self.rnn_layers.append(RNN(hidden_size, hidden_size))

    def forward(self, inputs: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        This is the forward pass for the Deep RNN.

        Args:
            inputs: inputs tensor. Dimensions: [batch, sequence,
                input size].
            h0: initial hidden state.

        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        # loop through each layer
        for layer in self.rnn_layers:
            outputs, h0 = layer(inputs, h0)
            inputs = outputs  # pass the output of the current layer to the next layer

        return outputs, h0


# Bidirectional RNN ---------------------------------------------------------------------------------------------------
class BidirectionalRNNFunction(torch.autograd.Function):
    """
    Class for the implementation of the forward and backward pass of
    the RNN.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        inputs: torch.Tensor,
        h0: torch.Tensor,
        weight_ih_f: torch.Tensor,
        weight_hh_f: torch.Tensor,
        bias_ih_f: torch.Tensor,
        bias_hh_f: torch.Tensor,
        weight_ih_b: torch.Tensor,
        weight_hh_b: torch.Tensor,
        bias_ih_b: torch.Tensor,
        bias_hh_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This is the forward method of the RNN.

        Args:
            ctx: context for saving elements for the backward.
            inputs: input tensor. Dimensions: [batch, sequence,
                input size].
            h0: first hidden state. Dimensions: [1, batch, hidden size].
            weight_ih_f: weight for the inputs for forward pass.
                Dimensions: [hidden size, input size].
            weight_hh_f: weight for the inputs for forward pass.
                Dimensions: [hidden size, hidden size].
            bias_ih_f: bias for the inputs for forward pass.
                Dimensions: [hidden size].
            bias_hh_f: bias for the inputs for forward pass.
                Dimensions: [hidden size].
            weight_ih_b: weight for the inputs for backward pass.
                Dimensions: [hidden size, input size].
            weight_hh_b: weight for the inputs for backward pass.
                Dimensions: [hidden size, hidden size].
            bias_ih_b: bias for the inputs for backward pass.
                Dimensions: [hidden size].
            bias_hh_b: bias for the inputs for backward pass.
                Dimensions: [hidden size].

        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """

        batch, sequence, _ = inputs.shape
        hidden_size = h0.shape[-1]

        outputs_f = torch.zeros(
            batch, sequence, hidden_size, device=inputs.device, dtype=inputs.dtype
        )

        outputs_b = torch.zeros(
            batch, sequence, hidden_size, device=inputs.device, dtype=inputs.dtype
        )
        h0_f = h0[0]
        h0_b = h0[1]
        hn_f = h0_f.squeeze(0)
        hn_b = h0_b.squeeze(0)

        for i in range(sequence):
            x_f = inputs[:, i, :]
            hn_f = (
                x_f @ weight_ih_f.t() + hn_f @ weight_hh_f.t() + bias_ih_f + bias_hh_f
            )
            hn_f = hn_f * (hn_f > 0).float()
            outputs_f[:, i, :] = hn_f

        for i in range(sequence - 1, -1, -1):
            x_b = inputs[:, i, :]
            hn_b = (
                x_b @ weight_ih_b.t() + hn_b @ weight_hh_b.t() + bias_ih_b + bias_hh_b
            )
            hn_b = hn_b * (hn_b > 0).float()
            outputs_b[:, i, :] = hn_b

        outputs = torch.cat((outputs_f, outputs_b), dim=-1)

        ctx.save_for_backward(
            inputs,
            h0,
            weight_ih_f,
            weight_hh_f,
            bias_ih_f,
            bias_hh_f,
            weight_ih_b,
            weight_hh_b,
            bias_ih_b,
            bias_hh_b,
            outputs,
            hn_f.unsqueeze(0),
            hn_b.unsqueeze(0),
        )

        return outputs, torch.cat((hn_f.unsqueeze(0), hn_b.unsqueeze(0)), dim=0)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor, grad_hn: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        This is the backward method of the RNN.

        Args:
            ctx: context for saving elements for the backward.
            grad_output: gradient of the output.
                Dimensions: [batch, sequence, hidden size].
            grad_hn: gradient of the hidden state.
                Dimensions: [1, batch, hidden size].

        Returns:
            gradients for the inputs.
                Dimensions: [batch, sequence, input size].
            gradients for the first hidden state.
                Dimensions: [1, batch, hidden size].
            gradients for the weight_ih for the forward pass.
                Dimensions: [hidden size, input size].
            gradients for the weight_hh for the forward pass.
                Dimensions: [hidden size, hidden size].
            gradients for the bias_ih for the forward pass.
                Dimensions: [hidden size].
            gradients for the bias_hh for the forward pass.
                Dimensions: [hidden size].
            gradients for the weight_ih for the backward pass.
                Dimensions: [hidden size, input size].
            gradients for the weight_hh for the backward pass.
                Dimensions: [hidden size, hidden size].
            gradients for the bias_ih for the backward pass.
                Dimensions: [hidden size].
            gradients for the bias_hh for the backward pass.
                Dimensions: [hidden size].
        """
        (
            inputs,
            h0,
            weight_ih_f,
            weight_hh_f,
            bias_ih_f,
            bias_hh_f,
            weight_ih_b,
            weight_hh_b,
            bias_ih_b,
            bias_hh_b,
            outputs,
            hn_f,
            hn_b,
        ) = ctx.saved_tensors

        batch, sequence, input_size = inputs.size()

        # deconcatenate the outputs
        outputs_f = outputs[:, :, : hn_f.size(-1)]
        outputs_b = outputs[:, :, hn_f.size(-1) :]

        # initialize the gradients
        grad_output_f = grad_output[:, :, : hn_f.size(-1)]
        grad_output_b = grad_output[:, :, hn_f.size(-1) :]

        grad_inputs = torch.zeros(
            batch, sequence, input_size, device=inputs.device, dtype=inputs.dtype
        )
        grad_h0_f = torch.zeros_like(h0[0])
        grad_h0_b = torch.zeros_like(h0[1])
        grad_weight_ih_f = torch.zeros_like(weight_ih_f)
        grad_weight_hh_f = torch.zeros_like(weight_hh_f)
        grad_bias_ih_f = torch.zeros_like(bias_ih_f)
        grad_bias_hh_f = torch.zeros_like(bias_hh_f)
        grad_weight_ih_b = torch.zeros_like(weight_ih_b)
        grad_weight_hh_b = torch.zeros_like(weight_hh_b)
        grad_bias_ih_b = torch.zeros_like(bias_ih_b)
        grad_bias_hh_b = torch.zeros_like(bias_hh_b)

        grad_hn_f = grad_hn[0].squeeze(0)
        grad_hn_b = grad_hn[1].squeeze(0)

        for i in range(sequence - 1, -1, -1):
            x_f = inputs[:, i, :]
            hn_f = outputs_f[:, i, :]
            h_ant_f = h0[0].squeeze(0) if i == 0 else outputs_f[:, i - 1, :]

            grad_hn_f = grad_hn_f + grad_output_f[:, i, :]
            grad_hn_f = grad_hn_f * (hn_f > 0).float()
            grad_x_f = grad_hn_f @ weight_ih_f

            grad_inputs[:, i, :] = grad_x_f

            grad_weight_ih_f = grad_weight_ih_f + grad_hn_f.t() @ x_f
            grad_weight_hh_f = grad_weight_hh_f + grad_hn_f.t() @ h_ant_f
            grad_bias_ih_f = grad_bias_ih_f + grad_hn_f
            grad_bias_hh_f = grad_bias_hh_f + grad_hn_f

            grad_hn_f = grad_hn_f @ weight_hh_f

        for i in range(sequence):
            x_b = inputs[:, i, :]
            hn_b = outputs_b[:, i, :]
            h_ant_b = h0[1].squeeze(0) if i == sequence - 1 else outputs_b[:, i + 1, :]

            grad_hn_b = grad_hn_b + grad_output_b[:, i, :]
            grad_hn_b = grad_hn_b * (hn_b > 0).float()
            grad_x_b = grad_hn_b @ weight_ih_b

            grad_inputs[:, i, :] += grad_x_b
            grad_weight_ih_b = grad_weight_ih_b + grad_hn_b.t() @ x_b
            grad_weight_hh_b = grad_weight_hh_b + grad_hn_b.t() @ h_ant_b
            grad_bias_ih_b = grad_bias_ih_b + grad_hn_b
            grad_bias_hh_b = grad_bias_hh_b + grad_hn_b

            grad_hn_b = grad_hn_b @ weight_hh_b

        grad_h0_f = grad_hn_f.unsqueeze(0)
        grad_h0_b = grad_hn_b.unsqueeze(0)

        return (
            grad_inputs,
            torch.cat((grad_h0_f, grad_h0_b), dim=0),
            grad_weight_ih_f,
            grad_weight_hh_f,
            grad_bias_ih_f,
            grad_bias_hh_f,
            grad_weight_ih_b,
            grad_weight_hh_b,
            grad_bias_ih_b,
            grad_bias_hh_b,
        )


class BidirectionalRNN(torch.nn.Module):
    """
    Class for the implementation of the RNN.
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        Constructor method for the RNN.

        Args:
            input_size: size of the input.
            hidden_size: size of the hidden state.
        """
        super(BidirectionalRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # forward pass
        self.weight_ih_f = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh_f = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_ih_f = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hh_f = torch.nn.Parameter(torch.Tensor(hidden_size))

        # backward pass
        self.weight_ih_b = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh_b = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_ih_b = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hh_b = torch.nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        This method initializes the weights and biases.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(
        self, inputs: torch.Tensor, h0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method is the forward of the RNN.

        Args:
            inputs: input tensor. Dimensions: [batch, sequence,
                input size].
            h0: initial hidden state.

        Returns:
            outputs tensor. Dimensions: [batch, sequence,
                hidden size].
            final hidden state for each element in the batch.
                Dimensions: [1, batch, hidden size].
        """
        return BidirectionalRNNFunction.apply(
            inputs,
            h0,
            self.weight_ih_f,
            self.weight_hh_f,
            self.bias_ih_f,
            self.bias_hh_f,
            self.weight_ih_b,
            self.weight_hh_b,
            self.bias_ih_b,
            self.bias_hh_b,
        )


# LSTM ---------------------------------------------------------------------------------------------------
class LSTMFunction(torch.autograd.Function):
    """
    Class to implement the forward and backward methods of the Conv2d
    layer.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        w_ih: torch.Tensor,
        w_hh: torch.Tensor,
        b_ih: torch.Tensor,
        b_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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

        gates = torch.mm(inputs, w_ih.t()) + b_ih + torch.mm(hidden, w_hh.t()) + b_hh
        i, f, g, o = gates.chunk(4, 1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        next_cell = f * cell + i * g
        next_hidden = o * torch.tanh(next_cell)

        # ctx.save_for_backward(inputs, hidden, cell, w_ih, w_hh, b_ih, b_hh, i, f, g, o, next_cell)

        return next_hidden, (next_hidden, next_cell)


# GRU ---------------------------------------------------------------------------------------------------
class GRUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, hidden, w_ih, w_hh, b_ih, b_hh):
        """
        Forward pass for the GRU.

        Args:
            ctx: context for saving elements for backward computation.
            inputs: input tensor at current timestep.
            hidden: hidden state from the previous timestep.
            w_ih: weights for input-hidden connections (3 gates).
            w_hh: weights for hidden-hidden connections (3 gates).
            b_ih: biases for input-hidden connections (3 gates).
            b_hh: biases for hidden-hidden connections (3 gates).

        Returns:
            next_hidden: next hidden state.
        """
        # Gates calculations
        input_gates = torch.mm(inputs, w_ih.t()) + b_ih
        hidden_gates = torch.mm(hidden, w_hh.t()) + b_hh
        # Split gates into update, reset, and new candidate gates
        i_z, i_r, i_n = input_gates.chunk(3, 1)
        h_z, h_r, h_n = hidden_gates.chunk(3, 1)

        z = torch.sigmoid(i_z + h_z)  # Update gate
        r = torch.sigmoid(i_r + h_r)  # Reset gate
        n = torch.tanh(i_n + r * h_n)  # New candidate

        # Compute the next hidden state
        next_hidden = (1 - z) * n + z * hidden

        ctx.save_for_backward(inputs, hidden, w_ih, w_hh, b_ih, b_hh, z, r, n)

        return next_hidden, next_hidden

    @staticmethod
    def backward(ctx, grad_hidden, grad_next_hidden):
        # To be implemented: Calculate the gradients for backward pass
        raise NotImplementedError("Backward pass not implemented.")


# Self-Attention ---------------------------------------------------------------------------------------------------
class SelfAttention(torch.nn.Module):
    """
    Self attention module.
    """

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        """
        Constructor of the class SelfAttention.

        Args:
            embedding_dim: embedding dimension of the model.
            num_heads: number of heads in the multi-head attention.
        """

        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.q = torch.nn.Linear(embedding_dim, embedding_dim)
        self.k = torch.nn.Linear(embedding_dim, embedding_dim)
        self.v = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method returns the output of the self attention module.

        Args:
            x: input tensor.
                Dimensions: [batch, sequence, embedding_dim].

        Returns:
            output of the self attention module.
        """

        q: torch.Tensor = self.q(x)
        k: torch.Tensor = self.k(x)
        v: torch.Tensor = self.v(x)

        # [B, H, N, E/H]
        q = q.view(
            x.size(0), x.size(1), self.num_heads, self.embedding_dim // self.num_heads
        )
        k = k.view(
            x.size(0), x.size(1), self.num_heads, self.embedding_dim // self.num_heads
        )
        v = v.view(
            x.size(0), x.size(1), self.num_heads, self.embedding_dim // self.num_heads
        )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # [B, H, N, E/H] x [B, H, E/H, N] -> [B, H, N, N]
        attention = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(
            self.embedding_dim // self.num_heads
        )

        attention = F.softmax(attention, dim=-1)

        # [B, H, N, N] x [B, H, N, E/H] -> [B, H, N, E/H]
        output = torch.matmul(attention, v)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(x.size(0), x.size(1), self.embedding_dim)
        )

        return output


# Bahdanau Attention ---------------------------------------------------------------------------------------------------
class BahdanauAttention(torch.nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(BahdanauAttention, self).__init__()

        # Define linear layers
        self.encoder_linear = torch.nn.Linear(
            encoder_hidden_dim, decoder_hidden_dim, bias=False
        )
        self.decoder_linear = torch.nn.Linear(
            decoder_hidden_dim, decoder_hidden_dim, bias=False
        )
        self.v = torch.nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        hidden: [batch_size, decoder_hidden_dim] - hidden state of the decoder
        encoder_outputs: [src_len, batch_size, encoder_hidden_dim] - output features from the encoder
        """
        # Repeat decoder hidden state to match the shape of encoder outputs
        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(0), 1)

        # Calculate the energy between the decoder hidden state and encoder outputs
        # Applying linear transformations and summing to get the energy scores
        energy = torch.tanh(
            self.encoder_linear(encoder_outputs) + self.decoder_linear(hidden)
        )

        # Project the energy scores to a single value and remove the last dimension
        attention = self.v(energy).squeeze(2)

        # Compute attention weights
        attention_weights = F.softmax(attention, dim=1)

        # Compute context vector as the weighted sum of encoder outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs.transpose(0, 1)
        )

        # Remove the time dimension from the context
        context = context.squeeze(1)

        return context, attention_weights


# SEARCH =================================================================================
def generate_caption_beam_search(
    self,
    image: torch.Tensor,
    vocab: Vocabulary,
    beam_size: int = 20,
    max_len: int = 50,
) -> str:
    """
    Generate a caption for a single image using beam search.

    Args:
        image (torch.Tensor): A single image tensor.
        vocab (Vocabulary): The Vocabulary object.
        beam_size (int): The size of the beam.
        max_len (int): Maximum length for the generated caption.

    Returns:
        str: The generated caption.
    """

    self.eval()
    with torch.no_grad():
        # Get first word (start token)
        hidden, states = self.decoder.lstm(self.encoder(image).unsqueeze(0), None)
        output = self.decoder.linear(hidden.squeeze(0))
        probs = torch.nn.functional.softmax(output, dim=1)
        predicted = probs.argmax(1)

        # Get next words
        features = self.decoder.embedding(predicted).unsqueeze(0)
        hidden, states = self.decoder.lstm(features, states)
        output = self.decoder.linear(hidden.squeeze(0))
        probs = torch.nn.functional.softmax(output, dim=1)
        scores, words = probs.topk(beam_size)

        # normalize the scores
        scores = scores / scores.sum()

        # initialize the lists
        captions = []
        probabilities = []
        features_list = []
        states_list = []

        # get the k best captions
        for i in range(beam_size):
            predicted = torch.tensor([words[0, i].item()], device=image.device)
            probability = scores[0, i].item()
            captions.append([predicted.item()])
            probabilities.append(probability)
            features_list.append(self.decoder.embedding(predicted).unsqueeze(0))
            states_list.append(states)

        for j in range(2, max_len):
            # initialize the lists
            new_captions = []
            new_probabilities = []
            new_features_list = []
            new_states_list = []

            # for each caption in the beam, get the k best captions
            for i in range(beam_size):
                # if the last word is the end token, stop
                if captions[i][-1] == vocab.word2idx["</s>"]:
                    new_captions.append(captions[i])
                    new_probabilities.append(probabilities[i])
                    new_features_list.append(features_list[i])
                    new_states_list.append(states_list[i])

                else:
                    # pass the features and the states through the lstm
                    hidden, states = self.decoder.lstm(features_list[i], states_list[i])
                    output = self.decoder.linear(hidden.squeeze(0))
                    probs = torch.nn.functional.softmax(output, dim=1)
                    scores, words = probs.topk(beam_size)

                    # normalize the scores
                    scores = scores / scores.sum()

                    # get the k best captions and its probabilities
                    for k in range(beam_size):
                        predicted = torch.tensor(
                            [words[0, k].item()], device=image.device
                        )
                        new_probabilities.append(
                            probabilities[i]
                            * scores[0, k].item()
                            * (max_len - j)
                            / max_len
                        )
                        new_captions.append(captions[i] + [predicted.item()])
                        new_features_list.append(
                            self.decoder.embedding(predicted).unsqueeze(0)
                        )
                        new_states_list.append(states)

            # rank the captions by probability and get the k best
            best_captions = sorted(
                list(
                    zip(
                        new_captions,
                        new_probabilities,
                        new_features_list,
                        new_states_list,
                    )
                ),
                key=lambda x: x[1],
                reverse=True,
            )[:beam_size]
            captions = [x[0] for x in best_captions]
            probabilities = [x[1] for x in best_captions]
            features_list = [x[2] for x in best_captions]
            states_list = [x[3] for x in best_captions]

            # normalize the probabilities
            probabilities = [p / sum(probabilities) for p in probabilities]

    return vocab.indices_to_caption(captions[0])


def generate_caption_greedy(
    self, image: torch.Tensor, vocab: Vocabulary, max_len: int = 50
) -> str:
    """
    Generate a caption for a single image.

    Args:
        image (torch.Tensor): A single image tensor.
        vocab (Vocabulary): The Vocabulary object.
        max_len (int): Maximum length for the generated caption.

    Returns:
        str: The generated caption.
    """

    # Model to evaluation mode
    self.eval()

    # Initialize the caption list
    caption = []

    with torch.no_grad():
        # Extract features from the image and stablish
        # the initial state for the lstm as None
        features = self.encoder(image).unsqueeze(0)
        states = None

        for _ in range(max_len):
            # Pass the features and the states through the lstm
            # to get the outputs and the new hidden state
            # and pass the outputs through the linear layer
            # to get the predicted word scores

            hidden, states = self.decoder.lstm(features, states)
            output = self.decoder.linear(hidden.squeeze(0))
            predicted = output.argmax(1)

            # Append the predicted word to the caption
            caption.append(predicted.item())

            # If the predicted word is the end token, stop
            if predicted == vocab.word2idx["</s>"]:
                break

            # Get the embedding of the predicted word
            # to use it in the next iteration as the input
            features = self.decoder.embedding(predicted).unsqueeze(0)

    # Convert the predicted word indices to words
    return vocab.indices_to_caption(caption)


def beam_search(model, src_input, max_len, start_symbol, beam_width=5):
    """
    Realiza la búsqueda de haz (Beam Search) para generar una secuencia de salida.

    Args:
        model (torch.nn.Module): El modelo entrenado.
        src_input (torch.Tensor): Secuencia de entrada.
        max_len (int): Longitud máxima de la secuencia de salida.
        start_symbol (int): Símbolo de inicio de la secuencia de salida.
        beam_width (int): Ancho del haz.

    Returns:
        torch.Tensor: Secuencia de salida generada.
    """
    device = next(model.parameters()).device
    src_input = src_input.unsqueeze(0).to(device)  # Añade la dimensión del lote

    with torch.no_grad():
        # Generar el símbolo de inicio
        trg_input = torch.tensor([start_symbol], device=device).unsqueeze(0)
        trg_scores = torch.zeros(1, 1, device=device)

        # Lista de hipótesis (secuencias y puntajes)
        hypotheses = [([], trg_scores)]

        # Bucle para generar la secuencia de salida
        for _ in range(max_len):
            new_hypotheses = []

            # Generar hipótesis para cada hipótesis actual
            for prefix, prefix_score in hypotheses:
                if len(prefix) > 0:
                    trg_input = torch.tensor(prefix, device=device).unsqueeze(0)

                output = model(src_input, trg_input)  # Generar salida del modelo
                log_probs = torch.log_softmax(output, dim=2)

                # Obtener los tokens y sus puntajes para el siguiente paso
                topk_scores, topk_indices = torch.topk(
                    log_probs[:, -1, :], beam_width, dim=1
                )

                # Expandir hipótesis
                for k in range(beam_width):
                    token = topk_indices[0][k].item()
                    score = topk_scores[0][k].item()
                    new_prefix = prefix + [token]
                    new_score = prefix_score + score
                    new_hypotheses.append((new_prefix, new_score))

            # Seleccionar las mejores hipótesis
            new_hypotheses = sorted(new_hypotheses, key=lambda x: x[1], reverse=True)[
                :beam_width
            ]
            hypotheses = new_hypotheses

            # Detenerse si se ha generado el símbolo de final de secuencia
            if (
                hypotheses[0][0][-1]
                == model.trg_field.vocab.stoi[model.trg_field.eos_token]
            ):
                break

    return torch.tensor(hypotheses[0][0], device=device)


def greedy_search(model, src_input, max_len, start_symbol):
    """
    Realiza la búsqueda greedy para generar una secuencia de salida.

    Args:
        model (torch.nn.Module): El modelo entrenado.
        src_input (torch.Tensor): Secuencia de entrada.
        max_len (int): Longitud máxima de la secuencia de salida.
        start_symbol (int): Símbolo de inicio de la secuencia de salida.

    Returns:
        torch.Tensor: Secuencia de salida generada.
    """
    device = next(model.parameters()).device
    src_input = src_input.unsqueeze(0).to(device)  # Añade la dimensión del lote

    with torch.no_grad():
        # Generar el símbolo de inicio
        trg_input = torch.tensor([start_symbol], device=device).unsqueeze(0)

        # Bucle para generar la secuencia de salida
        for _ in range(max_len):
            output = model(src_input, trg_input)  # Generar salida del modelo
            pred_token = output.argmax(2)[:, -1].unsqueeze(
                1
            )  # Obtener el token predicho
            trg_input = torch.cat(
                (trg_input, pred_token), dim=1
            )  # Agregar el token predicho a la secuencia de salida

            # Detenerse si se ha generado el símbolo de final de secuencia
            if (
                pred_token.item()
                == model.trg_field.vocab.stoi[model.trg_field.eos_token]
            ):
                break

    return trg_input.squeeze(0)


# POSITIONAL ENCODING =================================================================================
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class PositionalEncoding2(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding2, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ENCODER MODEL =================================================================================
class EncoderModel(torch.nn.Module):
    """
    Model constructed using complete Self Attention.
    """

    def __init__(
        self,
        sequence_length: int,
        vocab_to_int: dict[str, int],
        num_classes: int = 6,
        hidden_size: int = 1024,
        encoders: int = 6,
        embedding_dim: int = 100,
        num_heads: int = 4,
        **kwargs,
    ) -> None:
        """
        Constructor of the class EncoderModel.

        Args:
            sequence_length: input channels of the model.
            vocab_to_int: dictionary of vocabulary to integers.
            num_classes: output channels of the model.
            hidden_size: hidden size of the model.
            encoders: number of encoders in the model.
            embedding_dim: embedding dimension of the model.
            num_heads: number of heads in the multi-head attention.
        """

        super().__init__()
        self.vocab_to_int: dict[str, int] = vocab_to_int

        self.encoders: int = encoders

        # Embeddings
        self.embeddings = torch.nn.Embedding(
            len(vocab_to_int), embedding_dim, len(vocab_to_int) - 1
        )

        self.positional_encodings = PositionalEncoding2(embedding_dim)

        # Normalization
        self.normalization = torch.nn.LayerNorm(embedding_dim)

        # self-attention
        self.self_attention = SelfAttention(embedding_dim, num_heads)

        # mlp
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, embedding_dim),
        )

        # classification
        self.model = torch.nn.Linear(embedding_dim * sequence_length, num_classes)
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim * sequence_length),
            torch.nn.Linear(embedding_dim * sequence_length, hidden_size),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method returns a batch of predictions.
        Args:
            inputs: batch of texts.
                Dimensions: [batch, sequence]

        Returns:
            batch of predictions.
                Dimensions: [batch, num_classes].
        """

        x = self.embeddings(inputs)
        x = self.positional_encodings(x)

        for _ in range(self.encoders):
            attention_x = self.self_attention(x)

            x = self.normalization(attention_x) + x

            fc_x = self.fc(x)

            x = self.normalization(fc_x) + x

        x = x.view(x.size(0), -1)

        return self.model(x)


# DECODER MODEL =================================================================================
class DecoderModel(torch.nn.Module):
    """
    Model constructed using Self Attention and Cross Attention for decoding.
    """

    def __init__(
        self,
        sequence_length: int,
        vocab_to_int: dict[str, int],
        embedding_dim: int = 100,
        hidden_size: int = 1024,
        decoders: int = 6,
        num_heads: int = 4,
    ) -> None:
        """
        Constructor of the class DecoderModel.

        Args:
            sequence_length: Length of the sequence.
            vocab_to_int: dictionary of vocabulary to integers.
            embedding_dim: Embedding dimension of the model.
            hidden_size: Hidden size of the model.
            decoders: Number of decoder blocks in the model.
            num_heads: Number of heads in the multi-head attention.
        """
        super().__init__()
        self.vocab_to_int = vocab_to_int
        self.decoders = decoders

        # Embeddings
        self.embeddings = torch.nn.Embedding(
            len(vocab_to_int), embedding_dim, len(vocab_to_int) - 1
        )

        self.positional_encodings = PositionalEncoding2(embedding_dim)

        # Normalization
        self.normalization = torch.nn.LayerNorm(embedding_dim)

        # self-attention and cross-attention
        self.self_attention_blocks = torch.nn.ModuleList(
            [
                torch.nn.MultiheadAttention(embedding_dim, num_heads)
                for _ in range(decoders)
            ]
        )
        self.cross_attention_blocks = torch.nn.ModuleList(
            [
                torch.nn.MultiheadAttention(embedding_dim, num_heads)
                for _ in range(decoders)
            ]
        )

        # MLP
        self.fc_blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(embedding_dim, hidden_size),
                    torch.nn.Dropout(0.2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, embedding_dim),
                )
                for _ in range(decoders)
            ]
        )

        # Output to vocabulary size
        self.output_projection = torch.nn.Linear(embedding_dim, len(vocab_to_int))

    def forward(
        self, inputs: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        This method returns a batch of output sequences.
        Args:
            inputs: Batch of input tokens for the decoder.
                Dimensions: [batch, sequence]
            encoder_outputs: Outputs from the encoder to attend over.
                Dimensions: [batch, sequence, embedding_dim]

        Returns:
            Batch of output logits.
                Dimensions: [batch, sequence, vocab_size].
        """
        x = self.embeddings(inputs)
        x = self.positional_encodings(x)

        for idx, self_att, cross_att, fc in enumerate(
            zip(self.self_attention_blocks, self.cross_attention_blocks, self.fc_blocks)
        ):
            attention_x = self_att(x, x, x)
            x = self.normalization(attention_x) + x

            cross_attention_x = cross_att(
                x, encoder_outputs, encoder_outputs
            )  # queries, keys, values
            x = self.normalization(cross_attention_x) + x

            fc_x = fc(x)
            x = self.normalization(fc_x) + x

        logits = self.output_projection(x)

        return logits


# TRANSFORMER MODEL =================================================================================
# https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding2(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
