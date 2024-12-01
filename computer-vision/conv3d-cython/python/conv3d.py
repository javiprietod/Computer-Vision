import torch


# The functions are the same as in cython


def _unfold_one(
    x: torch.Tensor,
    kernel_size: tuple[int, int, int],
    stride: int,
    padding: int,
) -> torch.Tensor:
    """
    Performs the unfold operation to one 3D tensor using for loops.

    Parameters
    ----------
    x           : Input tensor.
    kernel_size : Size of the kernel. It does not require all dimensions to be equal.
    stride      : Stride.
    padding     : Padding

    Returns
    -------
    Unfolded tensor.
    """

    if x.dim() != 4:
        raise ValueError("Input tensor x must be a 4D tensor with shape (C, D, H, W)")

    x = x.unsqueeze(0)  # Shape: (1, C, D, H, W)

    # Pad the input tensor
    x_padded = torch.nn.functional.pad(
        x,
        pad=(
            padding,
            padding,  # pad W dimension
            padding,
            padding,  # pad H dimension
            padding,
            padding,  # pad D dimension
        ),
        mode="constant",
        value=0,
    )

    # Get sizes
    _, channels, D_padded, H_padded, W_padded = x_padded.shape

    # Compute output sizes
    output_D = ((D_padded - kernel_size[0]) // stride) + 1
    output_H = ((H_padded - kernel_size[1]) // stride) + 1
    output_W = ((W_padded - kernel_size[2]) // stride) + 1

    # Number of patches
    num_patches = output_D * output_H * output_W

    # Size of each patch
    patch_size = channels * kernel_size[0] * kernel_size[1] * kernel_size[2]

    # Initialize the output tensor
    unfolded = torch.empty((patch_size, num_patches), dtype=x.dtype, device=x.device)

    # Index to keep track of the position in the unfolded tensor
    patch_idx = 0

    # Loop over the output positions
    for d_out in range(output_D):
        d_start = d_out * stride
        for h_out in range(output_H):
            h_start = h_out * stride
            for w_out in range(output_W):
                w_start = w_out * stride

                # Extract the block
                block = x_padded[
                    :,
                    :,
                    d_start : d_start + kernel_size[0],
                    h_start : h_start + kernel_size[1],
                    w_start : w_start + kernel_size[2],
                ]  # Shape: (1, C, kD, kH, kW)

                # Flatten the block and store it in the unfolded tensor
                unfolded[:, patch_idx] = block.reshape(-1)
                patch_idx += 1

    return unfolded


def _unfold(
    x: torch.Tensor,
    kernel_size: tuple[int, int, int],
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """
    Performs the unfold operation to a batch of tensors.

    Parameters
    ----------
    x           : Input tensors.
    kernel_size : Size of the kernel. It does not require all dimensions to be equal.
    stride      : Stride.
    padding     : Padding

    Returns
    -------
    Unfolded tensors.
    """

    # TODO
    if x.dim() != 5:
        raise ValueError(
            "Input tensor x must be a 5D tensor with shape (N, C, D, H, W)"
        )

    # Unfold each tensor in the batch
    x_unfolded = torch.stack(
        [_unfold_one(x[i], kernel_size, stride, padding) for i in range(x.shape[0])]
    )

    return x_unfolded


def _convolve_one(x_unfold: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Performs the convolution operation to an unfolded tensor.

    Parameters
    ----------
    x_unfold : Unfolded tensor.
    w        : Weights of the kernel.

    Returns
    -------
    Convolved tensor.
    """

    # TODO
    out_channels = w.shape[0]
    in_channels = w.shape[1]
    kD = w.shape[2]
    kH = w.shape[3]
    kW = w.shape[4]
    kernel_size = in_channels * kD * kH * kW
    w_flat = w.view(out_channels, kernel_size)

    # Perform matrix multiplication
    out = w_flat @ x_unfold

    return out


def _convolve(x_unfold: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Performs the convolution operation to a batch of unfolded tensors.

    Parameters
    ----------
    x_unfold : Unfolded tensors.
    w        : Weights of the kernel.

    Returns
    -------
    Convolved tensors.
    """

    # TODO
    N = x_unfold.shape[0]
    C_in = w.shape[1]
    kD = w.shape[2]
    kH = w.shape[3]
    kW = w.shape[4]
    L = x_unfold.shape[2]
    C_out = w.shape[0]
    kernel_size_product = C_in * kD * kH * kW

    # Reshape weights to (C_out, C_in * kD * kH * kW)
    w_reshaped = w.reshape(C_out, kernel_size_product)

    # Perform batch matrix multiplication using einsum
    # x_unfold shape: (N, C_in * kD * kH * kW, L)
    # w_reshaped shape: (C_out, C_in * kD * kH * kW)
    # The operation computes y[n, o, l] = sum_k x_unfold[n, k, l] * w_reshaped[o, k]
    y = torch.einsum("nkl,ok->nol", x_unfold, w_reshaped)

    return y


def _fold(conv_output: torch.Tensor, output_size: tuple[int, int, int]) -> torch.Tensor:
    """
    Performs the fold operation to a batch of convolved tensors.

    Parameters
    ----------
    conv_output : Convolved tensors.
    output_size : Size of the expected output.

    Returns
    -------
    Folded tensors.
    """

    # TODO
    N = conv_output.shape[0]
    C_out = conv_output.shape[1]
    D_out = output_size[0]
    H_out = output_size[1]
    W_out = output_size[2]
    # C_in = conv_output.shape[2]
    # kD = conv_output.shape[3]
    # kH = conv_output.shape[4]
    # kW = conv_output.shape[5]

    # Reshape conv_output to (N, C_out, C_in * kD * kH * kW, D_out, H_out, W_out)
    conv_output_reshaped = conv_output.view(N, C_out, D_out, H_out, W_out)

    return conv_output_reshaped


def conv3d(
    x: torch.Tensor, w: torch.Tensor, stride: int = 1, padding: int = 0
) -> torch.Tensor:
    """
    Performs the convolution operation to a batch of 3D tensors using the above functions.

    Parameters
    ----------
    x       : Batch of 3D tensors.
    w       : Weights of the kernel.
    stride  : Stride.
    padding : Padding.

    Returns
    -------
    Convolved tensors.
    """

    # TODO
    kernel_size = w.shape[2:]
    x_unfold = _unfold(
        x, (kernel_size[0], kernel_size[1], kernel_size[2]), stride, padding
    )
    y = _convolve(x_unfold, w)
    output_shape = (
        (x.shape[2] + 2 * padding - (kernel_size[0] - 1) - 1) // stride + 1,
        (x.shape[3] + 2 * padding - (kernel_size[1] - 1) - 1) // stride + 1,
        (x.shape[4] + 2 * padding - (kernel_size[2] - 1) - 1) // stride + 1,
    )
    return _fold(y, output_size=output_shape)
