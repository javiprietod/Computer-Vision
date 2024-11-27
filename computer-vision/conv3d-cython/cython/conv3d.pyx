# cython: language_level=3

import numpy as np
cimport numpy as cnp
from numpy.lib.stride_tricks import sliding_window_view


cdef cnp.ndarray _unfold_one(
    cnp.ndarray[cnp.float32_t, ndim=4] x,
    tuple kernel_size,
    int stride,
    int padding,
):
    """
    Performs the unfold operation to one 3D tensor.

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

    # TODO
    x_padded = np.pad(
        x,
        pad_width=(
            (padding, padding),
            (padding, padding),
            (padding, padding)
        ),
        mode='constant',
        constant_values=0,
    )

    # Generate sliding windows
    x_unfolded = sliding_window_view(x_padded, kernel_size)

    # Apply stride
    x_unfolded = x_unfolded[::stride, ::stride, ::stride]

    # Reshape to combine the kernel dimensions
    x_unfolded = x_unfolded.reshape(-1, kernel_size[0] * kernel_size[1] * kernel_size[2])

    return x_unfolded


cpdef cnp.ndarray unfold(
    cnp.ndarray[cnp.float32_t, ndim=5] x,
    tuple kernel_size,
    int stride=1,
    int padding=0,
):
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
    x_padded = np.pad(
        x,
        pad_width=(
            (0, 0),                # Batch dimension
            (0, 0),                # Channel dimension
            (padding, padding),    # Depth dimension
            (padding, padding),    # Height dimension
            (padding, padding),    # Width dimension
        ),
        mode='constant',
        constant_values=0,
    )

    # Generate sliding windows over spatial dimensions (D, H, W)
    x_unfolded = sliding_window_view(
        x_padded,
        window_shape=kernel_size,
        axis=(2, 3, 4)
    )

    # Apply stride along spatial dimensions
    x_unfolded = x_unfolded[:, :, ::stride, ::stride, ::stride, :, :, :]

    # x_unfolded has shape (N, C, D_out, H_out, W_out, kD, kH, kW)
    N, C, D_out, H_out, W_out, kD, kH, kW = x_unfolded.shape

    # Reshape to combine the kernel dimensions
    x_unfolded = x_unfolded.reshape(N, C, D_out * H_out * W_out, kD * kH * kW)

    # Rearrange axes to get the final unfolded shape
    x_unfolded = x_unfolded.transpose(0, 1, 3, 2)  # (N, C, kernel_size_product, L)
    x_unfolded = x_unfolded.reshape(N, C * kD * kH * kW, D_out * H_out * W_out)

    return x_unfolded


cdef cnp.ndarray _convolve_one(cnp.ndarray[cnp.float32_t, ndim=2] x_unfold, cnp.ndarray[cnp.float32_t, ndim=5] w):
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
    cdef int out_channels = w.shape[0]
    cdef int in_channels = w.shape[1]
    cdef int kD = w.shape[2]
    cdef int kH = w.shape[3]
    cdef int kW = w.shape[4]
    cdef int kernel_size = in_channels * kD * kH * kW

    # Reshape the weights to match the unfolded input
    cdef cnp.ndarray[cnp.float32_t, ndim=2] w_flat = w.reshape(out_channels, kernel_size)

    # Perform matrix multiplication
    cdef cnp.ndarray[cnp.float32_t, ndim=2] out = np.dot(w_flat, x_unfold)

    return out


cpdef cnp.ndarray convolve(cnp.ndarray[cnp.float32_t, ndim=3] x_unfold, cnp.ndarray[cnp.float32_t, ndim=5] w):
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
    cdef int N = x_unfold.shape[0]
    cdef int C_in = w.shape[1]
    cdef int kD = w.shape[2]
    cdef int kH = w.shape[3]
    cdef int kW = w.shape[4]
    cdef int L = x_unfold.shape[2]
    cdef int C_out = w.shape[0]
    cdef int kernel_size_product = C_in * kD * kH * kW

    # Reshape weights to (C_out, C_in * kD * kH * kW)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] w_reshaped
    w_reshaped = w.reshape(C_out, kernel_size_product)

    # Perform batch matrix multiplication using einsum
    # x_unfold shape: (N, C_in * kD * kH * kW, L)
    # w_reshaped shape: (C_out, C_in * kD * kH * kW)
    # The operation computes y[n, o, l] = sum_k x_unfold[n, k, l] * w_reshaped[o, k]
    cdef cnp.ndarray[cnp.float32_t, ndim=3] y
    y = np.einsum('nkl,ok->nol', x_unfold, w_reshaped)

    return y


cpdef cnp.ndarray fold(cnp.ndarray[cnp.float32_t, ndim=3] conv_output, tuple output_size):
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
    cdef int N = conv_output.shape[0]
    cdef int C_out = conv_output.shape[1]
    cdef int L = conv_output.shape[2]
    cdef int D_out = output_size[0]
    cdef int H_out = output_size[1]
    cdef int W_out = output_size[2]
    cdef int expected_L = D_out * H_out * W_out

    # Validate that the number of positions matches the expected output size
    if L != expected_L:
        raise ValueError(f"Expected L (number of positions) to be {expected_L}, but got {L}.")

    # Reshape conv_output to (N, C_out, D_out, H_out, W_out)
    cdef cnp.ndarray[cnp.float32_t, ndim=5] folded
    folded = conv_output.reshape(N, C_out, D_out, H_out, W_out)

    return folded


cpdef cnp.ndarray conv3d(
    cnp.ndarray[cnp.float32_t, ndim=5] x,
    cnp.ndarray[cnp.float32_t, ndim=5] w,
    int stride=1,
    int padding=0,
):
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
    cdef Py_ssize_t N, C_in, D_in, H_in, W_in
    cdef Py_ssize_t C_out, kD, kH, kW
    cdef Py_ssize_t D_out, H_out, W_out
    cdef tuple output_size
    cdef cnp.ndarray x_unfold
    cdef cnp.ndarray conv_output
    cdef cnp.ndarray y

    # Extract dimensions
    N = x.shape[0]
    C_in = x.shape[1]
    D_in = x.shape[2]
    H_in = x.shape[3]
    W_in = x.shape[4]

    C_out = w.shape[0]
    # w.shape[1] is C_in, which we already have
    kD = w.shape[2]
    kH = w.shape[3]
    kW = w.shape[4]

    # Calculate output spatial dimensions
    D_out = (D_in + 2 * padding - kD) // stride + 1
    H_out = (H_in + 2 * padding - kH) // stride + 1
    W_out = (W_in + 2 * padding - kW) // stride + 1

    output_size = (D_out, H_out, W_out)

    # Unfold the input tensor
    x_unfold = unfold(x, (kD, kH, kW), stride, padding)  # Shape: (N, C_in * kD * kH * kW, L)

    # Convolve the unfolded tensor with the kernel weights
    conv_output = convolve(x_unfold, w)  # Shape: (N, C_out, L)

    # Fold the convolved output back to spatial dimensions
    y = fold(conv_output, output_size)  # Shape: (N, C_out, D_out, H_out, W_out)

    return y