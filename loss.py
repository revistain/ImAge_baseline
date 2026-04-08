from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
loss_fn = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())

# Global GapAligner instance (initialized in train.py)
gap_aligner = None
gap_loss_weight = 0.0

#  The loss function call (this method will be called at each training iteration)
def loss_function(descriptors, labels, f_thr=None, f_rgb=None):
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

    # Add gap alignment loss if available
    if gap_aligner is not None and gap_loss_weight > 0.0 and f_thr is not None and f_rgb is not None:
        gap_loss = gap_aligner.loss(f_thr, f_rgb.detach())
        loss = loss + gap_loss_weight * gap_loss

    return loss