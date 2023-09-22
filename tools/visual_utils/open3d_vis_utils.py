"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def create_arrow(pos=[0, 0, 0], degree=0, color=[1, 0, 0]):
    length = 0.5
    radius = 0.05
    
    arrow = open3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius, cone_radius=radius * 2, cylinder_height=length, cone_height=length/2)
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color(color) # Red color
    import math
    radian = math.radians(degree)
    rotation_axis = [0, math.radians(90), radian] # rotate around z-axis
    R = arrow.get_rotation_matrix_from_axis_angle(rotation_axis) 
    arrow.rotate(R, center=[0, 0, 0])
    arrow.translate(pos)
    
    return arrow


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, gt_keypoints=None, ref_keypoints=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    print(gt_boxes, ref_boxes, ref_labels, gt_boxes[..., 6] * 180 / 3.14159265358979323846, ref_boxes[..., 6] * 180 / 3.14159265358979323846)
    if isinstance(gt_keypoints, torch.Tensor):
        gt_keypoints = gt_keypoints.cpu().numpy()
    if isinstance(ref_keypoints, torch.Tensor):
        ref_keypoints = ref_keypoints.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    if gt_keypoints is not None:
        vis = draw_human(vis, gt_keypoints, (1, .2, .5))
    
    if ref_keypoints is not None:
        vis = draw_human(vis, ref_keypoints, (0., .8, .8))

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)
        corners = box3d.get_box_points()
        vis.add_geometry(create_arrow(corners[5], gt_boxes[i][6] * 180 / 3.14159265358, color))
        # vis.add_geometry(text_3d(f"{gt_boxes[i][6]:.2f}", pos=corners[5], font_size=1000, density=10))

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


def draw_human(vis, gt_keypoints, color=(0, .5, .5)):

    for points in gt_keypoints:
        mid_hip = (points[7] + points[8]) / 2
        mid_shoulder = (points[1] + points[2]) / 2
        points = np.concatenate([points, [mid_hip, mid_shoulder]], axis=0)

        lines = [
            [13, 14],  # spine
            [15, 1], [1, 3], [3, 5],  # upper, left
            [15, 2], [2, 4], [4, 6],  # upper, right
            [7, 9], [9, 11],  # bottom, left
            [8, 10], [10, 12],  # bottom, right
            [14, 7], [14, 8]  # spine to hip
        ]
        colors = [color for i in range(len(lines))]
        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(points)
        line_set.lines = open3d.utility.Vector2iVector(lines)
        line_set.colors = open3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

    return vis
