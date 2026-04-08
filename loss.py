import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity

loss_fn = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())

#  The loss function call (this method will be called at each training iteration)
def loss_function(descriptors, labels):
    # we mine the pairs/triplets if there is an online mining strategy
    if miner is not None:
        miner_outputs = miner(descriptors, labels)
        loss = loss_fn(descriptors, labels, miner_outputs)
        # calculate the % of trivial pairs/triplets 
        # which do not contribute in the loss value
        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)

    else: # no online mining
        loss = loss_fn(descriptors, labels)
        batch_acc = 0.0
    return loss

# [LeJEPA]
#####################################################
# BCS (Batched Characteristic Slicing) loss for SIGReg
def all_reduce(x, op):
    """All-reduce operation for distributed training."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        op = dist.ReduceOp.__dict__[op]
        dist.all_reduce(x, op=op)
        return x
    else:
        return x


def epps_pulley(x, t_min=-3, t_max=3, n_points=10):
    """Epps-Pulley test statistic for Gaussianity."""
    # integration points
    t = torch.linspace(t_min, t_max, n_points, device=x.device)
    # theoretical CF for N(0, 1)
    exp_f = torch.exp(-0.5 * t**2)
    # ECF
    x_t = x.unsqueeze(2) * t  # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)
    ecf = all_reduce(ecf, op="AVG")
    # weighted L2 distance
    err = exp_f * (ecf - exp_f).abs() ** 2
    T = torch.trapz(err, t, dim=1)
    return T


class BCS(nn.Module):
    """BCS (Batched Characteristic Slicing) loss for SIGReg."""

    def __init__(self, num_slices=256, lmbd=10.0):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0
        self.lmbd = lmbd

    def forward(self, z1, z2=None, return_sim=True, return_sigreg=True):

        """
        z1, z2 : B, D latent vectors from view1 and view2

        """
        with torch.no_grad():
            dev = z1.device
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (z1.size(1), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g) # DxB
            A /= A.norm(p=2, dim=0)
        
        view1 = z1 @ A # BxD DxB == BxB
        if z2 != None:
            view2 = z2 @ A # BxB

        results = dict()
        # self.step += 1
        if return_sigreg:
            bcs = epps_pulley(view1).mean() # scalar
            if z2 != None:
                bcs += epps_pulley(view2).mean()
            results['bcs_loss'] = bcs
        if return_sim and z2 != None:
            invariance_loss = F.mse_loss(z1, z2).mean() # scalar
            results['invariance_loss'] = invariance_loss
        return results
        # total_loss = invariance_loss + self.lmbd * bcs #
        # return {"loss": total_loss, "bcs_loss": bcs, "invariance_loss": invariance_loss}
