
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.model.ml_contact_ae.layers import ResidualStack, Conv3D


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

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 3
        stride = 1
        self.conv1 = Conv3D(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        self.conv2 = Conv3D(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1)
        self.conv3 = nn.Conv3d(h_dim, h_dim, kernel_size=kernel, stride=stride, padding=1)
        self.res_stack = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)

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
        if cond is None:
            cond = [0, 0, 0, 0]
        x1 = self.conv1(x) + cond[0]
        x2 = self.conv2(x1) + cond[1]
        x3 = self.conv3(x2) + cond[2]
        x4 = self.res_stack(x3) + cond[3]
        return x4, [x1, x2, x3]


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 17, 8, 8, 8))
    x = torch.tensor(x).float()

    # test encoder
    encoder = Encoder(17, 128, 3, 128)
    encoder_out, encoder_feats = encoder(x)
    print('Encoder out shape:', encoder_out.shape, [feat.shape for feat in encoder_feats])
