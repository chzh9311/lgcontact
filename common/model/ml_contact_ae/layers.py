
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Conv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        return self.conv(x)


class Deconv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Deconv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        return self.conv(x)

class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv3d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv3d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim) for _ in range(n_res_layers)])

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 64, 2, 2, 2))
    x = torch.tensor(x).float()
    # test Residual Layer
    res = ResidualLayer(64, 64, 128)
    res_out = res(x)
    print('Res Layer out shape:', res_out.shape)
    # test res stack
    res_stack = ResidualStack(64, 64, 128, 3)
    res_stack_out = res_stack(x)
    print('Res Stack out shape:', res_stack_out.shape)
