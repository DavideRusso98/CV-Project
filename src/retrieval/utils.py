import torch

def keypoint_similarity(gt_kpts, pred_kpts, sigmas, areas):
    """
    Params:
        gts_kpts: Ground-truth keypoints, Shape: [M, #kpts, 3],
                  where, M is the # of ground truth instances,
                         3 in the last dimension denotes coordinates: x,y, and visibility flag

        pred_kpts: Prediction keypoints, Shape: [N, #kpts, 3]
                   where  N is the # of predicted instances,

        areas: Represent ground truth areas of shape: [M,]

    Returns:
        oks: The Object Keypoint Similarity (OKS) score tensor of shape: [M, N]
    """

    # epsilon to take care of div by 0 exception.
    EPSILON = torch.finfo(torch.float32).eps

    # Eucleidian dist squared:
    # d^2 = (x1 - x2)^2 + (y1 - y2)^2
    # Shape: (M, N, #kpts) --> [M, N, 17]
    dist_sq = (gt_kpts[:, None, :, 0] - pred_kpts[..., 0]) ** 2 + (gt_kpts[:, None, :, 1] - pred_kpts[..., 1]) ** 2

    # Boolean ground-truth visibility mask for v_i > 0. Shape: [M, #kpts] --> [M, 17]
    vis_mask = gt_kpts[..., 2].int() > 0

    # COCO assigns k = 2σ.
    k = 2 * sigmas

    # Denominator in the exponent term. Shape: [M, 1, #kpts] --> [M, 1, 17]
    denom = 2 * (k ** 2) * (areas[:, None, None] + EPSILON)

    # Exponent term. Shape: [M, N, #kpts] --> [M, N, 17]
    exp_term = dist_sq / denom

    # Object Keypoint Similarity. Shape: (M, N)
    oks = (torch.exp(-exp_term) * vis_mask[:, None, :]).sum(-1) / (vis_mask[:, None, :].sum(-1) + EPSILON)

    return oks