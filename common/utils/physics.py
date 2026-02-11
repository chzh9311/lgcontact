import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils.geometry import get_perpend_vecs_tensor

class StableLoss(nn.Module):
    def __init__(self, k=0.1, pene_th=0.001, mu=0.8, eps=1e-3):
        super(StableLoss, self).__init__()
        self.k = k
        self.mu = mu
        self.pene_th = pene_th
        self.eps = eps

    def forward(self, obj_msdf, all_grid_points, all_grid_contact, all_grid_sdf_grad, n_adj_pt, obj_mass, gravity_direction, J=1):
        """
        Docstring for grid_contact2force
        
        :param all_grid_points: <B x N * K^3 x 3> relative to the center of mass.
        :param all_grid_contact: <B x N * K^3 x 1 + cse_dim>
        :param all_grid_sdf: <B x N * K^3>
        :param all_grid_sdf_grad: <B x N * K^3 x 3>
        :param k: constant
        :param n_adj_pt: <B x N * K^3 x 1>
        :J inertia matrix for calculating the stable loss, can be a scalar or a matrix of shape (B, 3, 3)
        """
        pene_depth = - obj_msdf.clone() + self.pene_th
        ## Do not mask according to contact to allow gradient propagation.
        pene_depth[pene_depth < 0] = 0
        F = self.k * pene_depth

        ## Average over adjacent points
        F = F * all_grid_contact / (1 + n_adj_pt)
        ## reducing point numbers by taking mask
        normals = all_grid_sdf_grad / (torch.norm(all_grid_sdf_grad, dim=-1, keepdim=True) + 1e-8)
        stable_loss = self.calc_stable_loss(all_grid_points, normals, F, obj_mass=obj_mass, gravity_direction=gravity_direction, J=J, mu=self.mu, eps=self.eps)
        
        return stable_loss

    def calc_stable_loss(self, obj_verts: torch.Tensor, obj_normal: torch.Tensor, point_force: torch.Tensor, obj_mass: torch.Tensor, gravity_direction:torch.Tensor,
                        J=1, mu=0.8, eps=0.001):
        """
        Regularizes the output so that the predicted forces are capable of keeping the object stable.
        The center of mass is in (0, 0, 0), the mass of the object is 1kg
        obj_verts: (batch_size, num_sampled_pts, 3)
        obj_normal: (batch_size, num_sampled_pts, 3)
        point_force: (batch_size, num_sampled_pts)
        obj_mass: (batch_size,)
        gravity_direction: (batch_size, 3)
        eps: the allowed error range of each penetration point.
        """
        N = - obj_normal
        B, T = get_perpend_vecs_tensor(obj_normal, device=obj_verts.device)
        Fub, Flb = calc_ub_lb(N / obj_mass[:, None, None], B / obj_mass[:, None, None], T / obj_mass[:, None, None], mu=mu)

        ## Calculate the torque stability
        ls = obj_verts
        Np = torch.cross(ls, N, dim=-1)
        Bp = torch.cross(ls, B, dim=-1)
        Tp = torch.cross(ls, T, dim=-1)
        if hasattr(J, 'shape') and len(J.shape) == 3:
            Np = torch.linalg.solve(J.unsqueeze(1), Np.unsqueeze(-1)).squeeze(-1)
            Bp = torch.linalg.solve(J.unsqueeze(1), Bp.unsqueeze(-1)).squeeze(-1)
            Tp = torch.linalg.solve(J.unsqueeze(1), Tp.unsqueeze(-1)).squeeze(-1)
        else:
            Np = Np / J
            Bp = Bp / J
            Tp = Tp / J

        Mub, Mlb = calc_ub_lb(Np, Bp, Tp, mu=mu)

        Aub = torch.cat((Fub, Mub), dim=-1).transpose(-1, -2)
        Alb = torch.cat((Flb, Mlb), dim=-1).transpose(-1, -2)

        PF = point_force.unsqueeze(1).repeat(1, 6, 1)
        loose_mat_ub = torch.zeros_like(Aub, device=Aub.device)
        loose_mat_lb = torch.zeros_like(Alb, device=Alb.device)
        if eps > 0:
            ## The upper bound with loose tensor
            eubplus_mask = (Aub > 0) & (PF > 0)
            eubminus_mask = (Aub < 0) & (PF > 0)
            elbplus_mask = (Alb < 0) & (PF > 0)
            elbminus_mask = (Alb > 0) & (PF > 0)
            
            loose_mat_ub[eubplus_mask] = eps * self.k
            loose_mat_ub[eubminus_mask] = -eps * self.k
            loose_mat_lb[elbplus_mask] = eps * self.k
            loose_mat_lb[elbminus_mask] = -eps * self.k
            loose_mat_ub[loose_mat_ub < -PF] = -PF[loose_mat_ub < -PF]
            loose_mat_lb[loose_mat_lb < -PF] = -PF[loose_mat_lb < -PF]

        acc = torch.zeros((obj_verts.shape[0], 6), device=obj_verts.device).float()
        acc[:, :3] = (gravity_direction * 9.81).view(-1, 3)
        up_residual = F.relu(-(acc + torch.sum(Aub * (PF + loose_mat_ub), dim=-1)))
        low_residual = F.relu(acc + torch.sum(Alb * (PF + loose_mat_lb), dim=-1))
        # up_residual1 = F.relu(-(acc + torch.sum(Aub * (PF), dim=-1)))
        # low_residual1 = F.relu(acc + torch.sum(Alb * (PF), dim=-1))
        # loss1 = torch.log(torch.sum(up_residual1 + low_residual1, dim=-1) + 1)

        loss = torch.log(torch.sum(up_residual + low_residual, dim=-1) + 1)
        return loss


def calc_ub_lb(N: torch.Tensor, B: torch.Tensor, T: torch.Tensor, mu: float, sumbt=False):
    """
    Calculate the upper & lower bound of the Newton's Law II equations
    """
    quantB = torch.zeros_like(B, device=B.device)
    quantB[B > 0] = mu
    quantB[B < 0] = -mu

    quantT = torch.zeros_like(T, device=T.device)
    quantT[T > 0] = mu
    quantT[T < 0] = -mu

    BqB = B * quantB
    TqT = T * quantT
    if sumbt:
        BqB = torch.sum(BqB, dim=-1)
        TqT = torch.sum(TqT, dim=-1)

    ub = N + BqB + TqT
    lb = N - BqB - TqT
    return ub, lb
