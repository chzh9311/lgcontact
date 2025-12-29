import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
from common.model.layers import ResnetBlockFC


class PointNetEncoder(nn.Module):
    """
    Modified from PointNet++
    """
    def __init__(self, in_channel, out_channel):
        super(PointNetEncoder, self).__init__()
        if in_channel == 3:
            self.normal_channel = False
        else:
            self.normal_channel = True
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, out_channel)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points


class LatentEncoder(nn.Module):
    def __init__(self, in_dim, dim, out_dim):
        super().__init__()
        self.block = ResnetBlockFC(size_in=in_dim, size_out=dim, size_h=dim)
        self.fc_mean = nn.Linear(dim, out_dim)
        self.fc_logstd = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = self.block(x, final_nl=True)
        return self.fc_mean(x), self.fc_logstd(x)