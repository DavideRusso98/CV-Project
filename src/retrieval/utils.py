import torch
import torchvision
from matplotlib import pyplot as plt


class Inference:
    """
    Inference class to make prediction with a given pre-trained model over a given image.
    Params:
        image: Preprocessed image
        model: Pre-trained model
    """

    def __init__(self, image, model):
        self.image = image
        self.model = model

    def get_kpts(self, is_th=False, th=50):
        """
        Make predictions with given image and return a keypoint tensor with higher scores.
        Params:
            is_th: Boolean variable, if True it makes thresholding otherwise not.
            th: Threshold value.

        Return:
            Keypoint tensor with shape [20,3].
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model([self.image])
        self.boxes = predictions[0]['boxes'].cpu().detach().numpy()
        self.keypoints = predictions[0]['keypoints'].cpu().detach().numpy()
        self.scores = predictions[0]['scores'].cpu().detach().numpy()
        self.keypoint_scores = predictions[0]['keypoints_scores'].cpu().detach().numpy()
        # print(predictions[0]['keypoints_scores'])
        kpts = self.keypoints[self.scores.argmax()]
        if is_th:
            kpts_th = self.thresholding(th)
            kpts = kpts_th
        return kpts

    def thresholding(self, th):
        """
        Threshold function.
        If scores is below a given thershold value, set the corresponding element in the keypoints tensor to zero.
        Params:
            th: Threshold value.
        Return:
            Thresholded keypoint tensor.
        """
        scores = self.keypoint_scores[self.scores.argmax()]
        kpts = self.keypoints[self.scores.argmax()]
        for i in range(0, len(scores)):
            # print(scores[i])
            if scores[i] < th:
                kpts[i] = torch.tensor([0.0, 0.0, 0.0])
        return kpts

    def get_area(self):
        """
        Get H,W of higher score bbox for OKS evaluation.
        Return:
            A tensor with [2] shape.
        """
        bbox = torch.tensor(self.boxes[self.scores.argmax()]).reshape(4)
        return bbox[-2:]

    def plot_keypoint(self):
        plt.imshow(self.image.permute(1, 2, 0).cpu().numpy())
        ax = plt.gca()
        max_score_index = self.scores.argmax()
        keypoints = self.keypoints[max_score_index]
        bbox = torch.tensor(self.boxes[max_score_index]).reshape(4)

        ## Draws predicted bbox (red)
        x1, y1, w, h = torchvision.ops.box_convert(bbox, 'xyxy', 'xywh')
        rect = plt.Rectangle((x1, y1), w, h, fill=False, color='red')
        ax.add_patch(rect)

        ## Draws predicted keypoints (green)
        for j, keypoint in enumerate(keypoints):
            x, y, v = keypoint
            circle = plt.Circle((x, y), 2, color='red', fill=True)
            ax.add_patch(circle)

        plt.axis('off')
        plt.show()


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