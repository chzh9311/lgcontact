
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.model.gridvqvae.layers import ResidualStack, Deconv3D, UpSample3D, Conv3D


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dims, out_dim, n_res_layers, res_h_dim,
                 condition=True):
        super(Decoder, self).__init__()
        self.deconv0 = nn.ConvTranspose3d(
                    in_dim, h_dims[0], kernel_size=3, stride=1, padding=1)
        self.residual_block = ResidualStack(h_dims[0], h_dims[0], res_h_dim, n_res_layers)
        self.conv0 = Conv3D(h_dims[0], h_dims[1], kernel_size=1, stride=1, padding=0)
        self.upsample1 = UpSample3D(scale_factor=2)
        self.deconv1 = Deconv3D(h_dims[1] * (1+condition), h_dims[2],
                               kernel_size=3, stride=1, padding=1)
        self.upsample2 = UpSample3D(scale_factor=2)
        self.deconv2 = Deconv3D(h_dims[2] * (1 + condition), h_dims[3],
                               kernel_size=3, stride=1, padding=1)
        # self.deconv3 = nn.Conv3d(h_dim // 2, out_dim,
        #                        kernel_size=1, stride=1, padding=0)
        self.deconv3 = Conv3D(h_dims[3] * (1 + condition), h_dims[3], kernel_size=1,
                                stride=1, padding=0)
        self.final_cse = nn.Conv3d(h_dims[3], out_dim-1, kernel_size=1,
                            stride=1, padding=0)
        self.final_contact = nn.Sequential(
                nn.Conv3d(h_dims[3], 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )

        # self.inverse_conv_stack = nn.Sequential(
        #     nn.ConvTranspose3d(
        #         in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
        #     ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
        #     nn.ConvTranspose3d(h_dim, h_dim // 2,
        #                        kernel_size=kernel, stride=stride, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose3d(h_dim//2, 16, kernel_size=kernel,
        #                        stride=stride, padding=1)
        # )

    def forward(self, x, cond=None):
        if cond is None:
            x = self.deconv0(x)
            x0 = self.residual_block(x)
            x0 = self.conv0(x0)
            x1 = self.upsample1(x0)
            x1 = self.deconv1(x0)
            x2 = self.upsample2(x1)
            x2 = self.deconv2(x2)
            x3 = self.deconv3(x2)
        else:
            x = self.deconv0(x)
            x0 = self.residual_block(x)
            x0 = self.conv0(x0)
            x1 = self.upsample1(torch.cat([x0, cond[0]], dim=1))
            x1 = self.deconv1(x1)
            x2 = self.upsample2(torch.cat([x1, cond[1]], dim=1))
            x2 = self.deconv2(x2)
            x3 = self.deconv3(torch.cat([x2, cond[2]], dim=1))
        cse = self.final_cse(x3)
        contact = self.final_contact(x3)
        return torch.cat([contact, cse], dim=1), [x0, x1, x2]
        # cse = self.final_cse(x) + cond[3]
        # contact = self.final_contact(x)
        # return torch.cat([contact, cse], dim=1)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 128, 2, 2, 2))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(128, 32, 17, 3, 64)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)
