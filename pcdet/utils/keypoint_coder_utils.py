import numpy as np
import torch


class ResidualCoderKP(object):
    """
    Args:
        num_joints: 13 joints plus 1 faked center point.
    """
    def __init__(self, num_joints=14, **kwargs):
        super().__init__()
        self.code_size = num_joints * 3

    def encode_torch(self, keypoints, anchors):
        """
        Args:
            keypoints: (N, n, 3) [x, y, z, ..., center_x, center_y, center_z]
            anchors: (N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Note:
            center_x, center_y, center_z may be any 3D point.
        """
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, *_ = torch.split(anchors, 1, dim=-1)
        # *keypoints, center_x, center_y, center_z = torch.split(keypoints, 1, dim=-2)
        keypoints = keypoints[..., :13, :]
        center_x = keypoints[..., -1, 0:1]
        center_y = keypoints[..., -1, 1:2]
        center_z = keypoints[..., -1, 2:3]

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (center_x - xa) / diagonal
        yt = (center_y - ya) / diagonal
        zt = (center_z - za) / dza

        dxt = torch.log(keypoints[..., 0] / dxa)
        dyt = torch.log(keypoints[..., 1] / dya)
        dzt = torch.log(keypoints[..., 2] / dza)

        dt_all = torch.stack([dxt, dyt, dzt], dim=-1)

        return torch.cat([dt_all, torch.stack([xt, yt, zt], dim=-1)], dim=-2)

    def decode_torch(self, keypoint_encodings, anchors):
        """
        Args:
            keypoint_encodings: (B, N, n, 3) [x, y, z, ..., center_x, center_y, center_z]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
        """

        # Find the center point given the fixed hwd
        assert keypoint_encodings.size(-2) == self.code_size // 3 and keypoint_encodings.size(-1) == 3, keypoint_encodings.shape

        xa, ya, za, dxa, dya, dza, *_ = torch.split(anchors, 1, dim=-1)
        # *kps, center_x, center_y, center_z = torch.split(keypoint_encodings, 1, dim=-2)
        keypoints = keypoint_encodings[..., :13, :]
        center_x = keypoint_encodings[..., -1, 0:1]
        center_y = keypoint_encodings[..., -1, 1:2]
        center_z = keypoint_encodings[..., -1, 2:3]

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = center_x * diagonal + xa
        yg = center_y * diagonal + ya
        zg = center_z * dza + za

        dxg = torch.exp(keypoints[..., 0]) * dxa
        dyg = torch.exp(keypoints[..., 1]) * dya
        dzg = torch.exp(keypoints[..., 2]) * dza

        dg_all = torch.stack([dxg, dyg, dzg], dim=-1)

        return torch.cat([dg_all, torch.stack([xg, yg, zg], dim=-1)], dim=-2)
