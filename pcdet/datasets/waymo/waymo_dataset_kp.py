# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved.

import os
import pickle
import copy
import numpy as np
import SharedArray
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path

from ...utils import box_utils, common_utils
from .waymo_dataset import WaymoDataset


class WaymoDatasetKP(WaymoDataset):

    def prepare_data(self, data_dict):
        return super().prepare_data(data_dict)

    def include_waymo_data(self, mode, keypoints_only=True):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []
        seq_name_to_infos = {}

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                if keypoints_only:
                    new_infos = []
                    for info in infos:
                        if len(info["annos"].keys()) != 11:  # Sequences that includes keypoints
                            new_infos.append(info)
                    infos = new_infos
                waymo_infos.extend(infos)
            
            if len(infos) == 0:
                num_skipped_infos += 1
                continue

            seq_name_to_infos[infos[0]['point_cloud']['lidar_sequence']] = infos

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))

        use_sequence_data = self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED
        if not use_sequence_data:
            seq_name_to_infos = None 
        return seq_name_to_infos

    def get_sequence_data(self, info, points, sequence_name, sample_idx, sequence_cfg, load_pred_boxes=False):
        """
        Args:
            info:
            points:
            sequence_name:
            sample_idx:
            sequence_cfg:
        Returns:
        """
        raise NotImplementedError

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        
        def get_template_prediction(num_samples):
            box_dim = 9 if self.dataset_cfg.get('TRAIN_WITH_SPEED', False) else 7
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 'pred_labels': np.zeros(num_samples),
                'kps_lidar': np.zeros([num_samples, 14, 3])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_kps = box_dict['pred_kps'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['kps_lidar'] = pred_kps
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        input_dict = {
            'sample_idx': sample_idx
        }
        if self.use_shared_memory and index < self.shared_memory_file_limit:
            sa_key = f'{sequence_name}___{sample_idx}'
            points = SharedArray.attach(f"shm://{sa_key}").copy()
        else:
            points = self.get_lidar(sequence_name, sample_idx)

        if self.dataset_cfg.get('SEQUENCE_CONFIG', None) is not None and self.dataset_cfg.SEQUENCE_CONFIG.ENABLED:
            points, num_points_all, sample_idx_pre_list, poses, pred_boxes, pred_scores, pred_labels = self.get_sequence_data(
                info, points, sequence_name, sample_idx, self.dataset_cfg.SEQUENCE_CONFIG,
                load_pred_boxes=self.dataset_cfg.get('USE_PREDBOX', False)
            )
            input_dict['poses'] = poses
            if self.dataset_cfg.get('USE_PREDBOX', False):
                input_dict.update({
                    'roi_boxes': pred_boxes,
                    'roi_scores': pred_scores,
                    'roi_labels': pred_labels,
                })

        input_dict.update({
            'points': points,
            'frame_id': info['frame_id'],
        })

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown', ignore_name="keypoint")

            # Override gt_boxes_lidar with keypoints only boxes
            global_speed = np.pad(annos[
                'keypoint_box_speed_global'], ((0, 0), (0, 1)), mode='constant', constant_values=0)  # (N, 3)
            speed = np.dot(global_speed, np.linalg.inv(info['pose'][:3, :3].T))
            speed = speed[:, :2]
            gt_boxes_lidar = np.concatenate([
                annos['keypoint_box_center'], annos['keypoint_box_dimensions'],
                annos['keypoint_box_heading_angles'][..., np.newaxis], speed],
                axis=1
            )
            # assert False, ([a for a in annos["gt_boxes_lidar"] if np.abs(a[:3] - gt_boxes_lidar[0][:3]).sum() <= 0.00001], gt_boxes_lidar)
            annos['gt_boxes_lidar'] = gt_boxes_lidar

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                assert False, "Need to update keypoints"
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.dataset_cfg.get('TRAIN_WITH_SPEED', False):
                assert gt_boxes_lidar.shape[-1] == 9
            else:
                gt_boxes_lidar = gt_boxes_lidar[:, 0:7]

            if self.training and self.dataset_cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN', False):
                mask = (annos['keypoint_box_num_points_in_gt'] > 0)  # filter empty boxes
                annos['keypoint_box_name'] = annos['keypoint_box_name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['keypoint_box_num_points_in_gt'] = annos['keypoint_box_num_points_in_gt'][mask]
            input_dict.update({
                'gt_names': annos['keypoint_box_name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('keypoint_box_num_points_in_gt', None),
                # Load keypoints
                # 'keypoint_index': annos["keypoint_index"],
                'keypoint_location': annos["keypoint_location"],
                'keypoint_visibility': annos["keypoint_visibility"],
                'keypoint_mask': annos["keypoint_mask"],
                # 'keypoint_dims': annos["keypoint_dims"],
                # 'keypoint_has_batch_dimension': annos["keypoint_has_batch_dimension"],
                # 'keypoint_box_center': annos["keypoint_box_center"],
                # 'keypoint_box_size': annos["keypoint_box_size"],
                # Corresponding box attributes if keypoint exists
                # 'keypoint_box_name': annos["keypoint_box_name"],
                # 'keypoint_box_difficulty': annos["keypoint_box_difficulty"],
                # 'keypoint_box_dimensions': annos["keypoint_box_dimensions"],
                # 'keypoint_box_location': annos["keypoint_box_location"],
                # 'keypoint_box_heading_angles': annos["keypoint_box_heading_angles"],
                # 'keypoint_box_obj_ids': annos["keypoint_box_obj_ids"],
                # 'keypoint_box_tracking_difficulty': annos["keypoint_box_tracking_difficulty"],
                # 'keypoint_box_num_points_in_gt': annos["keypoint_box_num_points_in_gt"],
                # 'keypoint_box_speed_global': annos["keypoint_box_speed_global"],
                # 'keypoint_box_accel_global': annos["keypoint_box_accel_global"],
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])
            print(eval_det_annos[0].keys(), eval_gt_annos[0].keys())
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict
