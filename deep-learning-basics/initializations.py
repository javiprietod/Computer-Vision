import torch
import torch.optim
import torch.nn as nn


def xavier_uniform_(tensor, gain=1.0):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * torch.sqrt(torch.tensor(3.0 / (fan_in + fan_out)))
    a = std
    with torch.no_grad():
        return tensor.uniform_(-a, a)


def xavier_normal_(tensor, gain=1.0):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
    with torch.no_grad():
        return tensor.normal_(0, std)


def kaiming_uniform_(tensor, gain=1.0):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * torch.sqrt(torch.tensor(6.0 / fan_in))
    bound = std  # For uniform, we use the standard deviation to calculate the bound
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def kaiming_normal_(tensor, gain=1.0):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * torch.sqrt(torch.tensor(2.0 / fan_in))
    with torch.no_grad():
        return tensor.normal_(0, std)
