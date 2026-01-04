import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        # Add a dummy parameter to satisfy optimizer requirements
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        recon_x = x.clone()
        mean = torch.randn(batch_size, 16) * 0.1
        logvar = torch.randn(batch_size, 16) * 0.1

        return recon_x, mean, logvar