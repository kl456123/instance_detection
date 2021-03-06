# -*- coding: utf-8 -*-
import numpy as np
import torch
from core.utils import format_checker


##########################
# some common np ops
##########################
def bmm(a, b):
    """
    Args:
        a: shape(N, M, K)
        b: shape(N, K, S)
    """
    b_T = b.transpose(0, 2, 1)
    return (a[:, :, None, :] * b_T[:, None, :, :]).sum(axis=-1)


def py_area(boxes):
    """
    Args:
        boxes: shape(N,M,4)
    """
    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]
    area = width * height
    return area


def py_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: shape(N,4)
        boxes_b: shape(M,4)
    Returns:
        overlaps: shape(N, M)
    """
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]
    boxes_a = np.repeat(np.expand_dims(boxes_a, 1), M, axis=1)
    boxes_b = np.repeat(np.expand_dims(boxes_b, 0), N, axis=0)

    xmin = np.maximum(boxes_a[:, :, 0], boxes_b[:, :, 0])
    ymin = np.maximum(boxes_a[:, :, 1], boxes_b[:, :, 1])
    xmax = np.minimum(boxes_a[:, :, 2], boxes_b[:, :, 2])
    ymax = np.minimum(boxes_a[:, :, 3], boxes_b[:, :, 3])

    w = xmax - xmin
    h = ymax - ymin
    w[w < 0] = 0
    h[h < 0] = 0

    inner_area = w * h
    boxes_a_area = py_area(boxes_a)
    boxes_b_area = py_area(boxes_b)

    iou = inner_area / (boxes_a_area + boxes_b_area - inner_area)
    return iou


def calc_location(dims, dets_2d, ry, p2):
    K = p2[:3, :3]
    K_homo = np.eye(4)
    K_homo[:3, :3] = K

    # K*T
    KT = p2[:, -1]
    T = np.dot(np.linalg.inv(K), KT)

    num = dets_2d.shape[0]

    zeros = np.zeros_like(ry)
    ones = np.ones_like(ry)
    R = np.stack(
        [
            np.cos(ry), zeros, np.sin(ry), zeros, ones, zeros, -np.sin(ry),
            zeros, np.cos(ry)
        ],
        axis=-1).reshape(num, 3, 3)

    h = dims[:, 0]
    w = dims[:, 1]
    l = dims[:, 2]
    zeros = np.zeros_like(w)
    x_corners = np.stack(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], axis=-1)
    y_corners = np.stack([zeros, zeros, zeros, zeros, -h, -h, -h, -h], axis=-1)
    z_corners = np.stack(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], axis=-1)

    corners = np.stack([x_corners, y_corners, z_corners], axis=-1)

    # after rotation
    #  corners = np.dot(R, corners)

    top_corners = corners[:, -4:]
    bottom_corners = corners[:, :4]
    diag_corners = bottom_corners[:, [2, 3, 0, 1]]
    #  left_side_corners = bottom_corners[:, [1, 2, 3, 0]]
    #  right_side_corners = bottom_corners[:, [3, 0, 1, 2]]

    # meshgrid
    # 4x4x4 in all

    num_top = top_corners.shape[1]
    num_bottom = bottom_corners.shape[1]
    top_index, bottom_index, side_index = np.meshgrid(
        np.arange(num_top), np.arange(num_bottom), np.arange(num_bottom))

    # in object frame
    # 3d points may be in top and bottom side
    # all corners' shape: (N,M,3)
    top_side_corners = top_corners[:, top_index.ravel()]
    bottom_side_corners = bottom_corners[:, bottom_index.ravel()]

    # 3d points may be in left and right side
    # both left and right are not difference here
    left_side_corners = bottom_corners[:, side_index.ravel()]
    right_side_corners = diag_corners[:, side_index.ravel()]

    num_cases = top_side_corners.shape[1]
    # shape(N, 4)
    zeros = np.zeros_like(dets_2d[:, 0])

    coeff_left = np.stack([zeros, zeros, dets_2d[:, 0]], axis=-1) - K[0]
    M = bmm(bmm(K[np.newaxis, ...], R), left_side_corners.transpose(0, 2, 1))
    M = M.transpose(0, 2, 1)
    bias_left = M[:, :, 0] - M[:, :, 2] * dets_2d[:, 0][..., None]

    coeff_right = np.stack([zeros, zeros, dets_2d[:, 2]], axis=-1) - K[0]
    M = bmm(bmm(K[np.newaxis, ...], R), right_side_corners.transpose(0, 2, 1))
    M = M.transpose(0, 2, 1)
    bias_right = M[:, :, 0] - M[:, :, 2] * dets_2d[:, 2][..., None]

    coeff_top = np.stack([zeros, zeros, dets_2d[:, 1]], axis=-1) - K[1]
    M = bmm(bmm(K[np.newaxis, ...], R), top_side_corners.transpose(0, 2, 1))
    M = M.transpose(0, 2, 1)
    bias_top = M[:, :, 1] - M[:, :, 2] * dets_2d[:, 1][..., None]

    coeff_bottom = np.stack([zeros, zeros, dets_2d[:, 3]], axis=-1) - K[1]
    M = bmm(bmm(K[np.newaxis, ...], R), bottom_side_corners.transpose(0, 2, 1))
    M = M.transpose(0, 2, 1)
    bias_bottom = M[:, :, 1] - M[:, :, 2] * dets_2d[:, 3][..., None]

    # shape(N, 4, 3)
    A = np.stack([coeff_left, coeff_top, coeff_right, coeff_bottom], axis=-2)
    A = np.expand_dims(A, axis=1)
    # shape(N, 64, 4, 3)
    A = np.repeat(A, num_cases, axis=1)
    # shape(N, 64, 4)
    b = np.stack([bias_left, bias_top, bias_right, bias_bottom], axis=-1)

    results_x = []
    errors = []
    for i in range(num):
        for j in range(num_cases):
            res = np.linalg.lstsq(A[i, j], b[i, j])
            results_x.append(res[0] - T)
            if (len(res[1])):
                errors.append(res[1])
            else:
                errors.append(np.zeros(1))

    results_x = np.stack(results_x, axis=0).reshape(num, num_cases, -1)
    errors = np.stack(errors, axis=0).reshape(num, num_cases)
    # idx = errors.argmax(axis=-1)
    # translation = results_x[np.arange(num), idx]
    idx = match(dets_2d, corners, results_x, R, p2)
    translation = results_x[np.arange(num), idx]

    return translation


def match(boxes_2d, corners, trans_3d, r, p):
    """
    Args:
        boxes_2d: shape(N, 4)
        corners: shape(N, 8, 3)
        trans_3d: shape(N, 64,3)
        ry: shape(N, 3, 3)
    """

    corners_3d = bmm(r, corners.transpose(0, 2, 1))
    trans_3d = np.repeat(np.expand_dims(trans_3d, axis=-2), 8, axis=-2)
    corners_3d = np.expand_dims(
        corners_3d.transpose((0, 2, 1)), axis=1) + trans_3d
    corners_3d = corners_3d.reshape(-1, 3)
    corners_3d_homo = np.hstack((corners_3d, np.ones(
        (corners_3d.shape[0], 1))))

    corners_2d = np.dot(p, corners_3d_homo.T)
    corners_2d_xy = corners_2d[:2, :] / corners_2d[2, :]

    corners_2d_xy = corners_2d_xy.reshape(2, -1, 8)
    xmin = corners_2d_xy[0, :, :].min(axis=-1)
    ymin = corners_2d_xy[1, :, :].min(axis=-1)
    xmax = corners_2d_xy[0, :, :].max(axis=-1)
    ymax = corners_2d_xy[1, :, :].max(axis=-1)

    batch_size = boxes_2d.shape[0]
    boxes_2d_proj = np.stack(
        [xmin, ymin, xmax, ymax], axis=-1).reshape(batch_size, -1, 4)
    # import ipdb
    # ipdb.set_trace()
    # import ipdb
    # ipdb.set_trace()
    bbox_overlaps = []
    for i in range(batch_size):
        bbox_overlaps.append(
            py_iou(boxes_2d[i][np.newaxis, ...], boxes_2d_proj[i]))
    bbox_overlaps = np.concatenate(bbox_overlaps, axis=0)
    idx = bbox_overlaps.argmax(axis=-1)
    return idx


def ry_to_rotation_matrix(rotation_y):
    zeros = np.zeros_like(rotation_y)
    ones = np.ones_like(rotation_y)
    rotation_matrix = np.stack(
        [
            np.cos(rotation_y), zeros, np.sin(rotation_y), zeros, ones, zeros,
            -np.sin(rotation_y), zeros, np.cos(rotation_y)
        ],
        axis=-1).reshape(-1, 3, 3)
    return rotation_matrix


def boxes_3d_to_corners_3d(boxes):
    """
    Args:
        boxes: shape(N, 7), (xyz,hwl, ry)
        corners_3d: shape()
    """
    assert boxes.shape[-1] == 7
    ry = boxes[:, -1]
    location = boxes[:, :3]
    h = boxes[:, 3]
    w = boxes[:, 4]
    l = boxes[:, 5]
    zeros = np.zeros_like(l)
    rotation_matrix = ry_to_rotation_matrix(ry)

    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([zeros, zeros, zeros, zeros, -h, -h, -h, -h])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    # shape(N, 3, 8)
    box_points_coords = np.stack((x_corners, y_corners, z_corners), axis=0)
    # rotate and translate
    # shape(N, 3, 8)
    corners_3d = bmm(rotation_matrix, box_points_coords.transpose(2, 0, 1))
    corners_3d = corners_3d + location[..., None]
    # shape(N, 8, 3)
    return corners_3d.transpose(0, 2, 1)
    # shape(N, 4, 8)
    # corners_3d_homo = np.concatenate(


# (corners_3d, np.ones_like(corners_3d[:, -1:, :])), axis=1)


def points_3d_to_points_2d(points_3d, p2):
    """
    Args:
        points_3d: shape(N, 3)
        p2: shape(3,4)
    Returns:
        points_2d: shape(N, 2)
    """

    # import ipdb
    # ipdb.set_trace()
    points_3d_homo = np.concatenate(
        (points_3d, np.ones_like(points_3d[:, -1:])), axis=-1)
    points_2d_homo = np.dot(p2, points_3d_homo.T).T
    points_2d_homo /= points_2d_homo[:, -1:]
    return points_2d_homo[:, :2]


def boxes_3d_to_corners_2d(boxes, p2):
    """
    Args:
        boxes: shape(N, 7)
        corners_2d: shape(N, )
    """
    corners_3d = boxes_3d_to_corners_3d(boxes)
    corners_2d = points_3d_to_points_2d(corners_3d.reshape((-1, 3)),
                                        p2).reshape(-1, 8, 2)
    return corners_2d


def corners_2d_to_boxes_2d(corners_2d):
    """
    Find the closure
    Args:
        corners_2d: shape(N, 8, 2)
    """
    xmin = corners_2d[:, :, 0].min(axis=-1)
    xmax = corners_2d[:, :, 0].max(axis=-1)
    ymin = corners_2d[:, :, 1].min(axis=-1)
    ymax = corners_2d[:, :, 1].max(axis=-1)

    return np.stack([xmin, ymin, xmax, ymax], axis=-1)


def boxes_3d_to_boxes_2d(boxes_3d, p2):
    corners_2d = boxes_3d_to_corners_2d(boxes_3d, p2)
    boxes_2d = corners_2d_to_boxes_2d(corners_2d)
    return boxes_2d


###########################
# pytorch
###########################
def torch_boxes_3d_to_corners_3d(boxes):
    """
    Args:
        boxes: shape(N, 7), (xyz,lhw, ry)
        corners_3d: shape()
    """
    ry = boxes[:, -1]
    location = boxes[:, :3]
    h = boxes[:, 3]
    w = boxes[:, 4]
    l = boxes[:, 5]
    zeros = torch.zeros_like(l).type_as(l)
    rotation_matrix = torch_ry_to_rotation_matrix(ry)

    x_corners = torch.stack(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=0)
    y_corners = torch.stack(
        [zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=0)
    z_corners = torch.stack(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=0)

    # shape(N, 3, 8)
    box_points_coords = torch.stack((x_corners, y_corners, z_corners), dim=0)
    # rotate and translate
    # shape(N, 3, 8)
    corners_3d = torch.bmm(rotation_matrix, box_points_coords.permute(2, 0, 1))
    corners_3d = corners_3d + location.unsqueeze(-1)
    # shape(N, 8, 3)
    return corners_3d.permute(0, 2, 1)


def torch_ry_to_rotation_matrix(rotation_y):
    """
    Args:
        rotation_y: shape(N,)
    """
    format_checker.check_tensor_shape(rotation_y, [None])
    zeros = torch.zeros_like(rotation_y)
    ones = torch.ones_like(rotation_y)
    rotation_matrix = torch.stack(
        [
            torch.cos(rotation_y), zeros, torch.sin(rotation_y), zeros, ones,
            zeros, -torch.sin(rotation_y), zeros, torch.cos(rotation_y)
        ],
        dim=-1).reshape(-1, 3, 3)
    return rotation_matrix


def torch_corners_2d_to_boxes_2d(corners_2d):
    """
    Find the closure
    Args:
        corners_2d: shape(N, 8, 2)
    """
    xmin, _ = corners_2d[:, :, 0].min(dim=-1)
    xmax, _ = corners_2d[:, :, 0].max(dim=-1)
    ymin, _ = corners_2d[:, :, 1].min(dim=-1)
    ymax, _ = corners_2d[:, :, 1].max(dim=-1)

    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def torch_boxes_3d_to_corners_2d(boxes, p2):
    """
    Args:
    boxes: shape(N, 7)
    corners_2d: shape(N, )
    """
    corners_3d = torch_boxes_3d_to_corners_3d(boxes)
    corners_2d = torch_points_3d_to_points_2d(corners_3d.reshape((-1, 3)),
                                              p2).reshape(-1, 8, 2)
    return corners_2d


def torch_points_3d_to_points_2d(points_3d, p2):
    """
    Args:
        points_3d: shape(N, 3)
        p2: shape(3,4)
    Returns:
        points_2d: shape(N, 2)
    """

    # import ipdb
    # ipdb.set_trace()
    format_checker.check_tensor_shape(points_3d, [None, 3])
    format_checker.check_tensor_shape(p2, [3, 4])
    points_3d_homo = torch.cat((points_3d, torch.ones_like(points_3d[:, -1:])),
                               dim=-1)
    points_2d_homo = torch.matmul(p2, points_3d_homo.transpose(
        0, 1)).transpose(0, 1)
    points_2d_homo = points_2d_homo / points_2d_homo[:, -1:]
    return points_2d_homo[:, :2]


def torch_xyxy_to_xywh(boxes):
    format_checker.check_tensor_shape(boxes, [None, None, 4])
    format_checker.check_tensor_type(boxes, 'float')
    xymin = boxes[:, :, :2]
    xymax = boxes[:, :, 2:4]
    xy = (xymin + xymax) / 2
    wh = xymax - xymin
    return torch.cat([xy, wh], dim=-1)


def torch_xywh_to_xyxy(boxes):
    format_checker.check_tensor_shape(boxes, [None, None, 4])
    format_checker.check_tensor_type(boxes, 'float')
    xy = boxes[:, :, :2]
    wh = boxes[:, :, 2:4]
    xymin = xy - wh / 2
    xymax = xy + wh / 2
    return torch.cat([xymin, xymax], dim=-1)


def torch_dir_to_angle(x, y):
    """
        Note that use kitti format(clockwise is positive) here
    """
    return -torch.atan2(y, x)


def torch_pts_2d_to_dir_3d(lines, p2):
    A = lines[:, :, 3] - lines[:, :, 1]
    B = lines[:, :, 0] - lines[:, :, 2]
    C = lines[:, :, 2] * lines[:, :, 1] - lines[:, :, 0] * lines[:, :, 3]
    plane = torch.bmm(p2.permute(0, 2, 1),
                      torch.stack(
                          [A, B, C], dim=-1).permute(0, 2, 1)).permute(0, 2, 1)
    a = plane[:, :, 0]
    c = plane[:, :, 2]
    ry = torch_dir_to_angle(c, -a)
    return ry


class ProjectMatrixTransform(object):
    def _format_check(p2, dtype=np.float32):
        pass

    @staticmethod
    def decompose_matrix(p2):
        K = p2[:3, :3]
        KT = p2[:, 3]
        T = np.dot(np.linalg.inv(K), KT)
        return K, T

    @classmethod
    def resize(cls, P, image_scale):
        cls._format_check(P)
        K, T = cls.decompose_matrix(P)

        K[0, :] = K[0, :] * image_scale[1]
        K[1, :] = K[1, :] * image_scale[0]
        K[2, 2] = 1
        KT = np.dot(K, T)

        return np.concatenate([K, KT[..., np.newaxis]], axis=-1)

    @classmethod
    def horizontal_flip(cls, P, w):
        cls._format_check(P)
        K, T = cls.decompose_matrix(P)
        K[0, 0] = -K[0, 0]
        K[0, 2] = w - K[0, 2]
        KT = np.dot(K, T)

        return np.concatenate([K, KT[..., np.newaxis]], axis=-1)

    @classmethod
    def crop(cls, P, offset):
        """
            Note here offset
        """
        cls._format_check(P)
        K, T = cls.decompose_matrix(P)

        K[0, 2] -= offset[0]
        K[1, 2] -= offset[1]
        KT = np.dot(K, T)

        return np.concatenate([K, KT[..., np.newaxis]], axis=-1)


def compute_ray_angle(center_2d, p2, format='kitti'):
    """
    Note that in kitti dataset, clockwise is positive direction,
    But in math(numpy is the same), counterclockwise is positive
    Args:
        center_2d: shape(N, M, 2)
        p2: shape(N, 3, 4)
    """
    assert format in ['kitti', 'normal'], 'kitti or normal can be accept'
    M = p2[:, :, :3]
    center_2d_homo = torch.cat(
        [center_2d, torch.ones_like(center_2d[:, :, -1:])], dim=-1)

    direction_vector = torch.bmm(
        torch.inverse(M), center_2d_homo.permute(0, 2, 1)).permute(0, 2, 1)
    ray_angle = torch.atan2(direction_vector[:, :, 2],
                            direction_vector[:, :, 0])

    if format == 'kitti':
        return -ray_angle
    return ray_angle


def torch_points_3d_distance(points1, points2):
    """
    Args:
        points1: shape(N, 3)
        points2: shape(N, 3)
    """
    torch.norm()


def torch_line_to_orientation(points1, points2):
    """
    If return positive number, turn to the right side,
    otherwise turn to the left side
    Note that if equal to zeros, line is horizontal or vertical
    Args:
        points1: shape(N, 2)
        points2: shape(N, 2)
    Return:
        res: shape(N, )
    """

    deltas = points1 - points2
    return deltas[:, 1] * deltas[:, 0]


def torch_window_filter(points_2d, window_shape, deltas=0):
    """
    Args:
        points_2d: shape(N, M, 2)
        window_shape: shape(N, 4),
        each item is like (xmin,ymin, xmax, ymax)
        deltas: soft interval
    """
    # if len(window_shape.shape) == 1:
    # window_shape = window_shape.unsqueeze(0)
    # else:
    # assert window_shape.shape[0] == points_2d.shape[0]

    format_checker.check_tensor_shape(points_2d, [None, None, 2])
    format_checker.check_tensor_shape(window_shape, [None, 4])

    window_shape = window_shape.unsqueeze(1)

    x_filter = (points_2d[:, :, 0] >= window_shape[:, :, 0] - deltas) & (
        points_2d[:, :, 0] <= window_shape[:, :, 2] + deltas)
    y_filter = (points_2d[:, :, 1] >= window_shape[:, :, 1] - deltas) & (
        points_2d[:, :, 1] <= window_shape[:, :, 3] + deltas)

    return x_filter & y_filter


def torch_points_2d_to_points_3d(points_2d, depth, p2):
    """
    Args:
        points_2d: shape(N, 2)
        depth: shape(N, ) or shape(N, 1)
        p2: shape(3, 4)
    """
    if len(depth.shape) == 1:
        depth = depth.unsqueeze(-1)
    format_checker.check_tensor_shape(points_2d, [None, 2])
    format_checker.check_tensor_shape(depth, [None, 1])
    format_checker.check_tensor_shape(p2, [3, 4])

    points_2d_homo = torch.cat([points_2d, torch.ones_like(points_2d[:, -1:])],
                               dim=-1)
    K = p2[:3, :3]
    KT = p2[:, 3]
    T = torch.matmul(torch.inverse(K), KT)
    K_inv = torch.inverse(K)
    points_3d = torch.matmul(K_inv, (depth * points_2d_homo).permute(
        1, 0)).permute(1, 0)

    # no rotation
    return points_3d - T


def torch_xyxy_to_corner_4c(label_boxes_2d):
    """
    Args:
        boxes_2d: shape(N, M, 4)
    Returns:
        boxes_4c: shape(N, M, 4, 2)
    """
    format_checker.check_tensor_shape(label_boxes_2d, [None, None, 4])
    left_top = label_boxes_2d[:, :, :2]
    right_down = label_boxes_2d[:, :, 2:]
    left_down = label_boxes_2d[:, :, [0, 3]]
    right_top = label_boxes_2d[:, :, [2, 1]]
    label_boxes_4c = torch.stack(
        [right_down, left_down, left_top, right_top], dim=2)

    format_checker.check_tensor_shape(label_boxes_4c, [None, None, 4, 2])
    return label_boxes_4c


def _reorder_corners_2d(corners_2d):
    """
    Args:
        corners_2d: shape(N, M, 8, 2)
    """
    pass


def torch_corner_4c_to_offset(corners_2d, label_boxes_2d):
    """
    Args:
        corners_2d: shape(N, M, 8, 2)
        label_boxes_2d: shape(N, M, 4)
    Return:
        offsets: shape(N, M, 8, 2)
    """
    label_corners_4c = torch_xyxy_to_corner_4c(label_boxes_2d)
    corners_4c = _reorder_corners_2d(corners_2d)
    return corners_4c - label_corners_4c
