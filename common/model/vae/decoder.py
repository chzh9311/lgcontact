import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.utils.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

class PointNetDecoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=128, out_dim=3):
        super(PointNetDecoder, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_dim,
                                          mlp=[hidden_dim, hidden_dim * 2], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=hidden_dim * 2 + 3,
                                          mlp=[hidden_dim * 2, hidden_dim * 4], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=hidden_dim * 4 + 3,
                                          mlp=[hidden_dim * 4, hidden_dim * 8], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=hidden_dim * 8 + hidden_dim * 4,
                                              mlp=[hidden_dim * 8, hidden_dim * 4])
        self.fp2 = PointNetFeaturePropagation(in_channel=hidden_dim * 4 + hidden_dim * 2,
                                              mlp=[hidden_dim * 4, hidden_dim * 2])
        self.fp1 = PointNetFeaturePropagation(in_channel=hidden_dim * 2 + in_dim,
                                              mlp=[hidden_dim * 2, hidden_dim])
        self.conv1 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(out_dim)

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        B, C, N = xyz.shape
        l0_xyz = xyz[:, :3, :] # B x 3 x N
        l0_points = xyz[:, 3:, :] # B x (C-3) x N
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) # xyz: B x 3 x 512; points: B x 128 x 512
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # xyz: B x 3 x 128; points: B x 256 x 128
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # xyz: B x 3 x 1; points: B x 512 x 1
        # l3_points = torch.cat((l3_points, obj_mass.view(-1, 16, 1)), dim=1)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points) # B x 256 x 128
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points) # B x 128 x 512
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points) # B x 64 x N
        feat = F.relu(self.bn1(self.conv1(l0_points))) # B x 64 x N
        return feat.permute(0, 2, 1)