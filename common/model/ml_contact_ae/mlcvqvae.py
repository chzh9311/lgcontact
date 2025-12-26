import torch
import torch.nn as nn
from .encoder import Encoder, ObjectGridEncoder
from .decoder import Decoder
from .quantizer import VectorQuantizer

class MLContactVQVAE(nn.Module):
    """
    MSDF-based 3D contact VQVAE of one local 3D patch
    The input contact representation is expected to be kernel_size^3 x (1 + 16); 1 refers to contact likelihood, 16 refers to Hand CSE.
    Also, the object local patch is represented as kernel_size^3 SDF values.
    TODO 1: how to integrate scale? Fix the scale to a constant value like 0.01 (1cm);
    Thus the input is k x k x k x (1 + 1 + 16) = k x k x k x 18
    Will be processed by 3D convolutions & deconvolutions.
    TODO 2: how to enable conditional input, i.e., predict the contact given the object local geometry? 
    We can try add this as part of the input to the encoder & decoder.
    """
    def __init__(self, in_dim, h_dims, res_h_dim, n_res_layers,
                 obj_in_dim, n_embeddings, embedding_dim, beta, out_dim=17, **kwargs):
        super(MLContactVQVAE, self).__init__()
        # encode image into continuous latent space
        # self.obj_encoder = Encoder(obj_in_dim, h_dims, obj_n_res_layers, obj_res_h_dim, condition=False)
        self.obj_encoder = ObjectGridEncoder(obj_in_dim, h_dims)
        self.encoder = Encoder(in_dim, h_dims, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv3d(
            h_dims[-1], embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        # self.obj_decoder = Decoder(h_dims[-1], h_dims[::-1], obj_in_dim, obj_n_res_layers, obj_res_h_dim, condition=True, final_layer=False)
        self.decoder = Decoder(embedding_dim, h_dims[::-1], out_dim, n_res_layers, res_h_dim, condition=True)

        # if save_img_embedding_map:
        #     self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        # else:
        #     self.img_to_embedding_map = None

    def forward(self, x, obj_msdf, verbose=False):

        obj_cond = self.obj_encoder(obj_msdf)
        z_e, _ = self.encoder(x, cond=obj_cond)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        ## Adding skip connections for object decoder
        # dec_obj_cond = self.obj_decoder(obj_feat, cond=enc_obj_cond[::-1])

        x_hat, _ = self.decoder(z_q, cond=obj_cond[::-1])

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity
