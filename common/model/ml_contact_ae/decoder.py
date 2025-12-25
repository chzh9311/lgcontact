
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.model.ml_contact_ae.layers import ResidualStack, Deconv3D


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

    def __init__(self, in_dim, h_dim, out_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 3
        stride = 1
        self.deconv0 = nn.ConvTranspose3d(
                    in_dim, h_dim, kernel_size=kernel, stride=stride, padding=1)
        self.residual_block = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        self.deconv1 = Deconv3D(h_dim, h_dim,
                               kernel_size=kernel, stride=stride, padding=1)
        self.deconv2 = Deconv3D(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1)
        self.deconv3 = nn.Conv3d(h_dim // 2, out_dim,
                               kernel_size=1, stride=1, padding=0)
        # self.final_cse = nn.Conv3d(h_dim // 2, out_dim-1, kernel_size=1,
        #                        stride=1, padding=0)
        # self.final_contact = nn.Sequential(
        #         nn.Conv3d(h_dim // 2, 1, kernel_size=1, stride=1, padding=0),
        #         nn.Sigmoid()
        #     )

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
            cond = [0, 0, 0, 0]
        x = self.deconv0(x)
        x = self.residual_block(x) + cond[0]
        x = self.deconv1(x) + cond[1]
        x = self.deconv2(x) + cond[2]
        x = self.deconv3(x) + cond[3]
        return x
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
