from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class OksLoss(nn.Module):
    """A PyTorch implementation of the Object Keypoint Similarity (OKS) loss as
    described in the paper "YOLO-Pose: Enhancing YOLO for Multi Person Pose
    Estimation Using Object Keypoint Similarity Loss" by Debapriya et al.
    (2022).

    The OKS loss is used for keypoint-based object recognition and consists
    of a measure of the similarity between predicted and ground truth
    keypoint locations, adjusted by the size of the object in the image.

    The loss function takes as input the predicted keypoint locations, the
    ground truth keypoint locations, a mask indicating which keypoints are
    valid, and bounding boxes for the objects.

    Args:
        code_weight (float): Weight for the loss.
    """

    def __init__(self, code_weights: list = None):
        super().__init__()

        self.code_weights = np.array(code_weights, dtype=np.float32)
        self.code_weights = torch.from_numpy(self.code_weights).reshape(-1, 3).mean(-1)

    def forward(self,
                output: Tensor,
                target: Tensor,
                mask: Tensor,
                bboxes: Optional[Tensor] = None) -> Tensor:
        """Calculates the OKS loss.

        Args:
            output (Tensor): Predicted keypoints in shape N x k x 3, N
                is the number of anchors, k is the number of keypoints,
                and 3 are the xyz coordinates.
            target (Tensor): Ground truth keypoints in the same shape as
                output.
            mask (Tensor): Mask of valid keypoints in shape N x k,
                with 1 for valid and 0 for invalid.
            bboxes (Optional[Tensor]): Bounding boxes in shape N x 6,
                where 6 are the xyz dx dy dz coordinates.

        Returns:
            Tensor: The calculated OKS loss.
        """
        oks = self.compute_oks(output, target, mask,bboxes)
        loss = 1 - oks
        return loss

    def compute_oks(self,
                    output: Tensor,
                    target: Tensor,
                    mask: Tensor,
                    bboxes: Optional[Tensor] = None) -> Tensor:
        """Calculates the OKS loss.

        Args:
            output (Tensor): Predicted keypoints in shape N x k x 3, where N
                is batch size, k is the number of keypoints, and 3 are the
                xyz coordinates.
            target (Tensor): Ground truth keypoints in the same shape as
                output.
            mask (Tensor): Mask of valid keypoints in shape N x k,
                with 1 for valid and 0 for invalid.
            bboxes (Optional[Tensor]): Bounding boxes in shape N x 6,
                where 6 are the xyz dx dy dz coordinates.

        Returns:
            Tensor: The calculated OKS loss.
        """

        dist = torch.norm(output - target, dim=-1)

        if bboxes is not None:
            # area = torch.norm(bboxes[..., 3:6] - bboxes[..., :3], dim=-1)
            area = torch.norm(bboxes[..., 3:6], dim=-1)
            dist = dist / area.clip(min=1e-8).unsqueeze(-1)

        return (torch.exp(-dist.pow(2) / 2) * self.code_weights * mask).sum(
            dim=-1) / mask.sum(dim=-1).clip(min=1e-8)
