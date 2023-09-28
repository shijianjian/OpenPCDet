import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import WaymoDatasetKP
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# python ./train.py --cfg_file ./cfgs/waymo_models/kp_multihead_pv_rcnn_with_centerhead_rpn.yaml
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/waymo_models/kp_multihead_pv_rcnn_with_centerhead_rpn.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='../data/waymo',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='/home/shij0c/git/KP3D/OpenPCDet/output/cfgs/waymo_models/kp_multihead_pv_rcnn_with_centerhead_rpn/default/ckpt/checkpoint_epoch_40.pth')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = WaymoDatasetKP(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,  # For visualizing augmentations
        root_path=Path(args.data_path), logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    # for idx, data_dict in enumerate(demo_dataset):
    #     data_dict = demo_dataset.collate_batch([data_dict])
    #     V.draw_scenes(
    #         points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0], gt_keypoints=data_dict['keypoint_location'][0]
    #     )

    #     if not OPEN3D_FLAG:
    #         mlab.show(stop=True)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                # points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                # ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0], gt_keypoints=data_dict['keypoint_location'][0],
                ref_boxes=pred_dicts[0]['pred_boxes'], ref_keypoints=pred_dicts[0]['pred_kps'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
