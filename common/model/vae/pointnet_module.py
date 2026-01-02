import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
from common.model.layers import ResnetBlockFC


class Pointnet(nn.Module):
    ''' PointNet-based encoder network
    Args:
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        out_dim (int): dimension of output
    '''
    def __init__(self, in_dim=3, hidden_dim=128, out_dim=3):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv2 = torch.nn.Conv1d(hidden_dim, 2 * hidden_dim, 1)
        self.conv3 = torch.nn.Conv1d(2 * hidden_dim, 4 * hidden_dim, 1)
        self.conv4 = torch.nn.Conv1d(4 * hidden_dim, out_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(2 * hidden_dim)
        self.bn3 = nn.BatchNorm1d(4 * hidden_dim)

        def maxpool(x, dim=-1, keepdim=False):
            out, _ = x.max(dim=dim, keepdim=keepdim)
            return out

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, x):
        ## global_vec: b x 3
        # x = x.permute(0, 2, 1) # b x c x 2048
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        # x = x.permute(0, 2, 1)

        return self.pool(x, dim=2), x


class PointNet2cls(nn.Module):
    """
    Modified from PointNet++
    """
    def __init__(self, in_channel, in_point, hidden_dim, out_channel):
        super(PointNet2cls, self).__init__()
        if in_channel == 3:
            self.normal_channel = False
        else:
            self.normal_channel = True
        self.sa1 = PointNetSetAbstraction(npoint=int(in_point//2), radius=0.2, nsample=32, in_channel=in_channel,
                                          mlp=[hidden_dim, hidden_dim, hidden_dim*2], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=int(in_point//4), radius=0.4, nsample=64, in_channel=hidden_dim*2 + 3,
                                          mlp=[hidden_dim*2, hidden_dim*2, hidden_dim*4], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=hidden_dim*4 + 3,
                                          mlp=[hidden_dim*4, hidden_dim*4, hidden_dim*8], group_all=True)
        self.fc1 = nn.Linear(hidden_dim*8, hidden_dim*4)
        self.bn1 = nn.BatchNorm1d(hidden_dim*4)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(hidden_dim*2, out_channel)

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
        x = l3_points.view(B, -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, -1)

        return x, l3_points


class PointNet2seg(nn.Module):
    def __init__(self, in_dim=6, in_point=256, hidden_dim=128, out_dim=3):
        super(PointNet2seg, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=int(in_point//2), radius=0.2, nsample=32, in_channel=in_dim,
                                          mlp=[hidden_dim, hidden_dim * 2], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=int(in_point//4), radius=0.4, nsample=64, in_channel=hidden_dim * 2 + 3,
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
        return None, feat

class LatentEncoder(nn.Module):
    def __init__(self, in_dim, dim, out_dim):
        super().__init__()
        self.block = ResnetBlockFC(size_in=in_dim, size_out=dim, size_h=dim)
        self.fc_mean = nn.Linear(dim, out_dim)
        self.fc_logstd = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = self.block(x, final_nl=True)
        return self.fc_mean(x), self.fc_logstd(x)