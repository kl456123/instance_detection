# -*- coding: utf-8 -*-
import sys
sys.path.append('./')
from core import constants
from data import datasets
import bbox_coders
import torch
from utils.drawer import ImageVisualizer
from utils import geometry_utils


def build_dataset(dataset_type='kitti'):
    if dataset_type == 'kitti':
        dataset_config = {
            "type": "mono_3d_kitti",
            "classes": ["Car", "Pedestrian"],
            "cache_bev": False,
            "dataset_file": "data/demo.txt",
            "root_path": "/data"
        }
    elif dataset_type == 'nuscenes':
        dataset_config = {
            'type': 'nuscenes',
            'root_path': '/data/nuscenes',
            'dataset_file': 'trainval.json',
            'data_path': 'samples/CAM_FRONT',
            'label_path': '.',
            'classes': ['car', 'pedestrian', 'truck']
        }
    elif dataset_type == 'keypoint_kitti':
        dataset_config = {
            "type": "keypoint_kitti",
            "classes": ["Car", "Pedestrian"],
            "cache_bev": False,
            "dataset_file": "data/demo.txt",
            "root_path": "/data"
        }

    dataset = datasets.build(dataset_config, transform=None, training=True)

    return dataset


def test_bbox_coders():
    coder_config = {'type': constants.KEY_ORIENTS}
    bbox_coder = bbox_coders.build(coder_config)

    dataset = build_dataset()
    sample = dataset[0]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    p2 = torch.from_numpy(sample[constants.KEY_STEREO_CALIB_P2])
    proposals = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])
    num_instances = torch.from_numpy(sample[constants.KEY_NUM_INSTANCES])

    label_boxes_3d = torch.stack(3 * [label_boxes_3d[:num_instances]], dim=0)
    proposals = torch.stack(3 * [proposals[:num_instances]], dim=0)
    p2 = torch.stack(3 * [p2], dim=0)
    orients = bbox_coder.encode_batch(label_boxes_3d, proposals, p2)
    # import ipdb
    # ipdb.set_trace()
    print(orients.shape)
    encoded_cls_orients = torch.zeros_like(orients[:, :, :2])
    cls_orients = orients[:, :, :1].long()
    row = torch.arange(0, cls_orients.numel()).type_as(cls_orients)
    encoded_cls_orients.view(-1, 2)[row, cls_orients.view(-1)] = 1
    encoded_orients = torch.cat(
        [encoded_cls_orients, orients[:, :, 1:]], dim=-1)

    ry = bbox_coder.decode_batch(encoded_orients, proposals, proposals, p2)
    # import ipdb
    # ipdb.set_trace()
    print(ry)
    print(label_boxes_3d[:, :, -1])
    print(sample[constants.KEY_IMAGE_PATH])


def test_orient_coder():
    coder_config = {'type': constants.KEY_ORIENTS}
    bbox_coder = bbox_coders.build(coder_config)

    dataset = build_dataset()
    sample = dataset[0]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    p2 = torch.from_numpy(sample[constants.KEY_STEREO_CALIB_P2])
    proposals = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])

    orients = bbox_coder.encode_batch(label_boxes_3d, proposals, p2)
    print(orients.shape)


def test_orientv3_coder():
    coder_config = {'type': constants.KEY_ORIENTS_V3}
    orient_coder = bbox_coders.build(coder_config)

    dataset = build_dataset()
    sample = dataset[0]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    num_instances = torch.from_numpy(sample[constants.KEY_NUM_INSTANCES])

    label_boxes_3d = torch.stack(3 * [label_boxes_3d[:num_instances]], dim=0)
    orients = orient_coder.encode_batch(label_boxes_3d)
    print(orients)


def test_orientv2_coder():
    coder_config = {'type': constants.KEY_ORIENTS_V2}
    bbox_coder = bbox_coders.build(coder_config)

    dataset = build_dataset()
    sample = dataset[0]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    p2 = torch.from_numpy(sample[constants.KEY_STEREO_CALIB_P2])
    proposals = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])
    num_instances = torch.from_numpy(sample[constants.KEY_NUM_INSTANCES])

    label_boxes_3d = torch.stack(1 * [label_boxes_3d[:num_instances]], dim=0)
    proposals = torch.stack(1 * [proposals[:num_instances]], dim=0)
    p2 = torch.stack(1 * [p2], dim=0)
    # import ipdb
    # ipdb.set_trace()
    orients = bbox_coder.encode_batch(label_boxes_3d, proposals, p2)
    print(orients)

    encoded_cls_orients = torch.zeros_like(orients[:, :, :3])
    cls_orients = orients[:, :, :1].long()
    row = torch.arange(0, cls_orients.numel()).type_as(cls_orients)
    encoded_cls_orients.view(-1, 3)[row, cls_orients.view(-1)] = 1
    encoded_orients = torch.cat(
        [encoded_cls_orients, orients[:, :, 1:]], dim=-1)

    ry = bbox_coder.decode_batch(encoded_orients, proposals, p2)
    print(ry)
    print(label_boxes_3d[:, :, -1])
    print(sample[constants.KEY_IMAGE_PATH])


def test_rear_side_coder():
    coder_config = {'type': constants.KEY_REAR_SIDE}
    bbox_coder = bbox_coders.build(coder_config)

    dataset = build_dataset()
    sample = dataset[0]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    p2 = torch.from_numpy(sample[constants.KEY_STEREO_CALIB_P2])
    proposals = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])
    num_instances = torch.from_numpy(sample[constants.KEY_NUM_INSTANCES])

    label_boxes_3d = torch.stack(1 * [label_boxes_3d[:num_instances]], dim=0)
    proposals = torch.stack(1 * [proposals[:num_instances]], dim=0)
    p2 = torch.stack(1 * [p2], dim=0)
    # import ipdb
    # ipdb.set_trace()
    orients = bbox_coder.encode_batch(label_boxes_3d, proposals, p2)
    print(orients)

    encoded_cls_orients = torch.zeros_like(orients[:, :, :2])
    cls_orients = orients[:, :, :1].long()
    row = torch.arange(0, cls_orients.numel()).type_as(cls_orients)
    encoded_cls_orients.view(-1, 2)[row, cls_orients.view(-1)] = 1
    encoded_orients = torch.cat(
        [encoded_cls_orients, orients[:, :, 1:]], dim=-1)

    ry = bbox_coder.decode_batch(encoded_orients, proposals, p2)
    print(ry)
    print(label_boxes_3d[:, :, -1])
    print(sample[constants.KEY_IMAGE_PATH])


def test_corners_coder():

    coder_config = {'type': constants.KEY_CORNERS_2D_NEAREST_DEPTH}
    bbox_coder = bbox_coders.build(coder_config)

    dataset = build_dataset('kitti')
    sample = dataset[0]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    label_boxes_2d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])
    p2 = torch.from_numpy(sample[constants.KEY_STEREO_CALIB_P2])
    proposals = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])
    num_instances = torch.from_numpy(sample[constants.KEY_NUM_INSTANCES])
    image_info = torch.from_numpy(sample[constants.KEY_IMAGE_INFO])

    label_boxes_3d = torch.stack(1 * [label_boxes_3d[:num_instances]], dim=0)
    label_boxes_2d = torch.stack(1 * [label_boxes_2d[:num_instances]], dim=0)
    proposals = torch.stack(1 * [proposals[:num_instances]], dim=0)
    image_info = torch.stack(1 * [image_info], dim=0)
    p2 = torch.stack(1 * [p2], dim=0)

    encoded_corners_2d = bbox_coder.encode_batch(
        label_boxes_3d, label_boxes_2d, p2, image_info)
    #  torch.cat([encoded_corners_2d, ])
    num_boxes = encoded_corners_2d.shape[1]
    batch_size = encoded_corners_2d.shape[0]
    center_depth = encoded_corners_2d[:, :, -1:]
    encoded_corners_2d = encoded_corners_2d[:, :, :-1].view(
        batch_size, num_boxes, 8, 4)

    encoded_visibility = torch.zeros_like(encoded_corners_2d[:, :, :, :2])
    visibility = encoded_corners_2d[:, :, :, -1:].long()
    row = torch.arange(0, visibility.numel()).type_as(visibility)
    encoded_visibility.view(-1, 2)[row, visibility.view(-1)] = 1
    encoded_corners_2d = torch.cat(
        [encoded_corners_2d[:, :, :, :3], encoded_visibility], dim=-1)

    encoded_corners_2d = torch.cat(
        [encoded_corners_2d.view(batch_size, num_boxes, -1), center_depth],
        dim=-1)

    decoded_corners_2d = bbox_coder.decode_batch(encoded_corners_2d, proposals,
                                                 p2)

    decoded_corners_2d = decoded_corners_2d.cpu().detach().numpy()

    image_path = sample[constants.KEY_IMAGE_PATH]
    image_dir = '/data/object/training/image_2'
    result_dir = './results/data'
    save_dir = 'results/images'
    calib_dir = '/data/object/training/calib'
    label_dir = None
    calib_file = None
    visualizer = ImageVisualizer(
        image_dir,
        result_dir,
        label_dir=label_dir,
        calib_dir=calib_dir,
        calib_file=calib_file,
        online=False,
        save_dir=save_dir)
    # import ipdb
    # ipdb.set_trace()
    visualizer.render_image_corners_2d(
        image_path, corners_3d=decoded_corners_2d[0], p2=p2[0])


def compute_ray_angle(C):
    """
    Args:
        C: shape(N, 3), 3d center of object
    Returns:
        alpha: shape(N, ) observe angle
    """
    alpha = -torch.atan2(C[:, 2], C[:, 0])
    return alpha


def test_corners_3d_coder():

    # import ipdb
    # ipdb.set_trace()
    coder_config = {'type': constants.KEY_CORNERS_3D}
    bbox_coder = bbox_coders.build(coder_config)

    dataset = build_dataset()
    sample = dataset[0]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    label_boxes_2d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])
    p2 = torch.from_numpy(sample[constants.KEY_STEREO_CALIB_P2])
    proposals = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])
    num_instances = torch.from_numpy(sample[constants.KEY_NUM_INSTANCES])

    # ry = compute_ray_angle(label_boxes_3d[:, :3])
    # label_boxes_3d[:, -1] += ry

    label_boxes_3d = torch.stack(1 * [label_boxes_3d[:num_instances]], dim=0)
    label_boxes_2d = torch.stack(1 * [label_boxes_2d[:num_instances]], dim=0)
    proposals = torch.stack(1 * [proposals[:num_instances]], dim=0)
    p2 = torch.stack(1 * [p2], dim=0)

    # import ipdb
    # ipdb.set_trace()
    # label_boxes_3d[:, :, -1] = 0

    encoded_corners_3d = bbox_coder.encode_batch(label_boxes_3d,
                                                 label_boxes_2d, p2)
    #  torch.cat([encoded_corners_2d, ])
    num_boxes = encoded_corners_3d.shape[1]
    batch_size = encoded_corners_3d.shape[0]

    decoded_corners_3d = bbox_coder.decode_batch(
        encoded_corners_3d.view(batch_size, num_boxes, -1), proposals, p2)

    decoded_corners_2d = geometry_utils.torch_points_3d_to_points_2d(
        decoded_corners_3d[0].view(-1, 3), p2[0]).view(-1, 8, 2)
    decoded_corners_2d = decoded_corners_2d.cpu().detach().numpy()

    image_path = sample[constants.KEY_IMAGE_PATH]
    image_dir = '/data/object/training/image_2'
    result_dir = './results/data'
    save_dir = 'results/images'
    calib_dir = '/data/object/training/calib'
    label_dir = None
    calib_file = None
    visualizer = ImageVisualizer(
        image_dir,
        result_dir,
        label_dir=label_dir,
        calib_dir=calib_dir,
        calib_file=calib_file,
        online=False,
        save_dir=save_dir)
    visualizer.render_image_corners_2d(image_path, decoded_corners_2d)


def test_keypoint_coder():
    coder_config = {'type': constants.KEY_KEYPOINTS}
    bbox_coder = bbox_coders.build(coder_config)

    dataset = build_dataset(dataset_type='keypoint_kitti')
    sample = dataset[0]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    label_boxes_2d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])
    p2 = torch.from_numpy(sample[constants.KEY_STEREO_CALIB_P2])
    proposals = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])
    num_instances = torch.from_numpy(sample[constants.KEY_NUM_INSTANCES])
    keypoints = sample[constants.KEY_KEYPOINTS]

    # ry = compute_ray_angle(label_boxes_3d[:, :3])
    # label_boxes_3d[:, -1] += ry

    label_boxes_3d = torch.stack(1 * [label_boxes_3d[:num_instances]], dim=0)
    label_boxes_2d = torch.stack(1 * [label_boxes_2d[:num_instances]], dim=0)
    proposals = torch.stack(1 * [proposals[:num_instances]], dim=0)
    keypoints = torch.stack(1 * [keypoints[:num_instances]], dim=0)
    p2 = torch.stack(1 * [p2], dim=0)

    # label_boxes_3d[:, :, -1] = 0

    encoded_corners_3d = bbox_coder.encode_batch(proposals, keypoints)
    #  torch.cat([encoded_corners_2d, ])
    num_boxes = encoded_corners_3d.shape[1]
    batch_size = encoded_corners_3d.shape[0]

    keypoint = encoded_corners_3d.view(batch_size, num_boxes, 8,
                                       -1)[..., 0].long()
    keypoint_heatmap = torch.zeros((batch_size * num_boxes * 8, 56 * 56))
    row = torch.arange(keypoint.numel()).type_as(keypoint)
    keypoint_heatmap[row, keypoint.view(-1)] = 1
    keypoint_heatmap = torch.stack([keypoint_heatmap] * 3, dim=1)

    # reshape before decode
    keypoint_heatmap = keypoint_heatmap.view(batch_size, num_boxes, -1)

    decoded_corners_2d = bbox_coder.decode_batch(proposals, keypoint_heatmap)

    decoded_corners_2d = decoded_corners_2d.cpu().detach().numpy()

    image_path = sample[constants.KEY_IMAGE_PATH]
    image_dir = '/data/object/training/image_2'
    result_dir = './results/data'
    save_dir = 'results/images'
    calib_dir = '/data/object/training/calib'
    label_dir = None
    calib_file = None
    visualizer = ImageVisualizer(
        image_dir,
        result_dir,
        label_dir=label_dir,
        calib_dir=calib_dir,
        calib_file=calib_file,
        online=False,
        save_dir=save_dir)
    # import ipdb
    # ipdb.set_trace()
    visualizer.render_image_corners_2d(
        image_path, corners_2d=decoded_corners_2d[0])


def test_keypoint_hm_coder():
    coder_config = {'type': constants.KEY_KEYPOINTS_HEATMAP}
    bbox_coder = bbox_coders.build(coder_config)

    dataset = build_dataset(dataset_type='keypoint_kitti')
    sample = dataset[0]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    label_boxes_2d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])
    p2 = torch.from_numpy(sample[constants.KEY_STEREO_CALIB_P2])
    proposals = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])
    num_instances = torch.from_numpy(sample[constants.KEY_NUM_INSTANCES])
    keypoints = sample[constants.KEY_KEYPOINTS]

    # ry = compute_ray_angle(label_boxes_3d[:, :3])
    # label_boxes_3d[:, -1] += ry

    label_boxes_3d = torch.stack(1 * [label_boxes_3d[:num_instances]], dim=0)
    label_boxes_2d = torch.stack(1 * [label_boxes_2d[:num_instances]], dim=0)
    proposals = torch.stack(1 * [proposals[:num_instances]], dim=0)
    keypoints = torch.stack(1 * [keypoints[:num_instances]], dim=0)
    p2 = torch.stack(1 * [p2], dim=0)

    # label_boxes_3d[:, :, -1] = 0

    # import ipdb
    # ipdb.set_trace()
    encoded_corners_3d = bbox_coder.encode_batch(proposals, keypoints)
    #  torch.cat([encoded_corners_2d, ])
    num_boxes = encoded_corners_3d.shape[1]
    batch_size = encoded_corners_3d.shape[0]

    keypoint_heatmap = encoded_corners_3d.view(batch_size, num_boxes, 8,
                                               -1)[..., :-1]
    # resolution = bbox_coder.resolution
    # keypoint_heatmap = torch.zeros((batch_size * num_boxes * 8, resolution * resolution))
    # row = torch.arange(keypoint.numel()).type_as(keypoint)
    # keypoint_heatmap[row, keypoint.view(-1)] = 1
    # keypoint_heatmap = torch.stack([keypoint_heatmap] * 3, dim=1)

    # reshape before decode
    keypoint_heatmap = keypoint_heatmap.contiguous().view(
        batch_size, num_boxes, -1)

    decoded_corners_2d = bbox_coder.decode_batch(proposals, keypoint_heatmap)

    decoded_corners_2d = decoded_corners_2d.cpu().detach().numpy()

    image_path = sample[constants.KEY_IMAGE_PATH]
    image_dir = '/data/object/training/image_2'
    result_dir = './results/data'
    save_dir = 'results/images'
    calib_dir = '/data/object/training/calib'
    label_dir = None
    calib_file = None
    visualizer = ImageVisualizer(
        image_dir,
        result_dir,
        label_dir=label_dir,
        calib_dir=calib_dir,
        calib_file=calib_file,
        online=False,
        save_dir=save_dir)
    # import ipdb
    # ipdb.set_trace()
    visualizer.render_image_corners_2d(
        image_path, corners_2d=decoded_corners_2d[0])


if __name__ == '__main__':
    # test_bbox_coders()
    # test_orient_coder()
    # test_orientv3_coder()
    # test_orientv2_coder()
    #  test_rear_side_coder()
    # test_corners_coder()
    # test_corners_3d_coder()
    # test_keypoint_coder()
    test_keypoint_hm_coder()
