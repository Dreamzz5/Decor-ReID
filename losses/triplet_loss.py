import math
import torch
import torch.nn.functional as F
from torch import nn
from losses.gather import GatherLayer


class TripletLoss(nn.Module):
    """ Triplet loss with hard example mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Args:
        margin (float): pre-defined margin.

    Note that we use cosine similarity, rather than Euclidean distance in the original paper.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.m = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        # l2-normlize
        inputs = F.normalize(inputs, p=2, dim=1)

        # gather all samples from different GPUs as gallery to compute pairwise loss.
        gallery_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)

        # compute distance
        dist = 1 - torch.matmul(inputs, gallery_inputs.t()) # values in [0, 2]

        # get positive and negative masks
        targets, gallery_targets = targets.view(-1,1), gallery_targets.view(-1,1)
        mask_pos = torch.eq(targets, gallery_targets.T).float().cuda()
        mask_neg = 1 - mask_pos

        # For each anchor, find the hardest positive and negative pairs
        dist_ap, _ = torch.max((dist - mask_neg * 99999999.), dim=1)
        dist_an, _ = torch.min((dist + mask_pos * 99999999.), dim=1)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
    
# # Adaptive weights
# def softmax_weights(dist, mask):
#     max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
#     diff = dist - max_v
#     Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
#     W = torch.exp(diff) * mask / Z
#     return W

# def normalize(x, axis=-1):
#     """Normalizing to unit length along the specified dimension.
#     Args:
#       x: pytorch Variable
#     Returns:
#       x: pytorch Variable, same shape as input
#     """
#     x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
#     return x

# class TripletLoss(nn.Module):
#     """Weighted Regularized Triplet'."""

#     def __init__(self, margin=0.3):
#         super(TripletLoss, self).__init__()
#         self.ranking_loss = nn.SoftMarginLoss()

#     def forward(self, inputs, targets, mode = 'pull', normalize_feature=True):
#         if normalize_feature:
#             inputs = normalize(inputs, axis=-1)
            
#         gallery_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
#         gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)
#         dist_mat = pdist_torch(inputs, gallery_inputs)

#         N = dist_mat.size(0)
#         # shape [N, N]
#         targets, gallery_targets = targets.view(-1,1), gallery_targets.view(-1,1)
#         mask_pos = torch.eq(targets, gallery_targets.T).float().cuda()
#         mask_neg = 1 - mask_pos

#         # `dist_ap` means distance(anchor, positive)
#         # both `dist_ap` and `relative_p_inds` with shape [N, 1]
#         dist_ap = dist_mat * mask_pos
#         dist_an = dist_mat * mask_neg

#         weights_ap = softmax_weights(dist_ap, mask_pos)
#         weights_an = softmax_weights(-dist_an, mask_neg)
#         furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
#         closest_negative = torch.sum(dist_an * weights_an, dim=1)

#         y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
#         if mode == 'pull':
#             loss = self.ranking_loss(closest_negative - furthest_positive, y)
#         else:
#             weights_an = softmax_weights(-dist_ap, mask_pos)
#             closest_postive = torch.sum(dist_ap * weights_an, dim=1)
#             loss = self.ranking_loss(closest_postive - furthest_positive, y)
#         # # compute accuracy
#         # correct = torch.ge(closest_negative, furthest_positive).sum().item()
#         return loss
        
# def pdist_torch(emb1, emb2):
#     '''
#     compute the eucilidean distance matrix between embeddings1 and embeddings2
#     using gpu
#     '''
#     m, n = emb1.shape[0], emb2.shape[0]
#     emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
#     emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
#     dist_mtx = emb1_pow + emb2_pow
#     dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
#     # dist_mtx = dist_mtx.clamp(min = 1e-12)
#     dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
#     return dist_mtx   