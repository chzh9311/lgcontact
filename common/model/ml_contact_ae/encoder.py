
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.model.ml_contact_ae.layers import ResidualStack, Conv3D, MaxPool3D


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dims, n_res_layers, res_h_dim, condition=True):
        super(Encoder, self).__init__()
        self.condition = condition
        self.conv0 = Conv3D(in_dim, h_dims[0], kernel_size=1, stride=1, padding=0)
        ## Double the channels if conditional input is provided
        self.conv1 = Conv3D(h_dims[0] * (1+condition), h_dims[1], kernel_size=3, stride=1, padding=1) 
        self.pool1 = MaxPool3D(kernel_size=2)
        self.conv2 = Conv3D(h_dims[1] * (1+condition), h_dims[2], kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool3D(kernel_size=2)
        self.conv3 = Conv3D(h_dims[2] * (1+condition), h_dims[3], kernel_size=3, stride=1, padding=1)
        self.res_stack = ResidualStack(h_dims[3], h_dims[3], res_h_dim, n_res_layers)

        # self.conv_stack = nn.Sequential(
        #     nn.Conv3d(in_dim, h_dim // 2, kernel_size=kernel,
        #               stride=stride, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool3d(2),
        #     nn.Conv3d(h_dim // 2, h_dim, kernel_size=kernel,
        #               stride=stride, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool3d(2),
        #     nn.Conv3d(h_dim, h_dim, kernel_size=kernel-1,
        #               stride=stride-1, padding=1),
        #     ResidualStack(
        #         h_dim, h_dim, res_h_dim, n_res_layers)

        # )

    def forward(self, x, cond=None):
        ## Concatenate the features once for each feature scale
        x0 = self.conv0(x)
        if not self.condition:
            x1 = self.conv1(x0)
            x1 = self.pool1(x1)
            x2 = self.conv2(x1)
            x2 = self.pool2(x2)
            x3 = self.conv3(x2)
        else:
            assert cond is not None, "Conditional features must be provided!"
            x1 = self.conv1(torch.cat([x0, cond[0]], dim=1))
            x1 = self.pool1(x1)
            x2 = self.conv2(torch.cat([x1, cond[1]], dim=1))
            x2 = self.pool2(x2)
            x3 = self.conv3(torch.cat([x2, cond[2]], dim=1))
        x3 = self.res_stack(x3)
        return x3, [x0, x1, x2]


class ObjectGridEncoder(nn.Module):
    """
    Encode the object local grid into multi-scale latent features
    """
    def __init__(self, in_dim, h_dims):
        super(ObjectGridEncoder, self).__init__()
        self.conv11 = Conv3D(in_dim, h_dims[0], kernel_size=1, stride=1, padding=0)
        self.conv12 = Conv3D(h_dims[0], h_dims[0], kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool3D(kernel_size=2)
        self.conv21 = Conv3D(h_dims[0], h_dims[1], kernel_size=1, stride=1, padding=0)
        self.conv22 = Conv3D(h_dims[1], h_dims[1], kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool3D(kernel_size=2)
        self.conv31 = Conv3D(h_dims[1], h_dims[2], kernel_size=1, stride=1, padding=0)
        self.conv32 = Conv3D(h_dims[2], h_dims[2], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv11(x)
        x0 = self.conv12(x)
        x1 = self.pool1(x0)
        x1 = self.conv21(x1)
        x1 = self.conv22(x1)
        x2 = self.pool2(x1)
        x2 = self.conv31(x2)
        x2 = self.conv32(x2)
        return [x0, x1, x2]


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 17, 8, 8, 8))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(17, 128, 3, 128)
    encoder_out, encoder_feats = encoder(x)
    print('Encoder out shape:', encoder_out.shape, [feat.shape for feat in encoder_feats])
