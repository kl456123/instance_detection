# -*- coding: utf-8 -*-
import torch

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils

from core.utils import format_checker

WHICH_PLANE = 'face'


class Order():
    which_plane = WHICH_PLANE
    FRONT_PLANE = [0, 1, 5, 4]
    REAR_PLANE = [3, 2, 6, 7]
    LEFT_PLANE = [3, 0, 4, 7]
    RIGHT_PLANE = [2, 1, 5, 6]
    SIDE_REORDER = [1, 5, 4, 0, 2, 6, 7, 3]
    FACE_REORDER = [0, 1, 5, 4, 3, 2, 6, 7]

    @classmethod
    def planes(cls):
        which_plane = cls.which_plane
        if which_plane == 'side':
            return cls.LEFT_PLANE, cls.RIGHT_PLANE
        elif which_plane == 'face':
            return cls.FRONT_PLANE, cls.REAR_PLANE
        else:
            raise ValueError('unknown plane')

    @classmethod
    def reorder(cls):
        which_plane = cls.which_plane
        if which_plane == 'side':
            return cls.SIDE_REORDER
        elif which_plane == 'face':
            return cls.FACE_REORDER
        else:
            raise ValueError('unknown plane')


@BBOX_CODERS.register('corners_2d_face')
class Corner2DFaceCoder(object):
    def __init__(self, config):
        pass

    def decode_batch(self, encoded_corners_2d_all, auxiliary_dict):
        """
        Args:
            encoded_all: shape(N, 8*2 + 8)
        """
        final_boxes_2d = auxiliary_dict[constants.KEY_BOXES_2D]
        # import ipdb
        # ipdb.set_trace()
        N, M = encoded_corners_2d_all.shape[:2]
        # encoded_corners_2d = torch.cat([encoded_corners_2d_all[:,:,::4],encoded_corners_2d_all[:,:,1::4]],dim=-1)
        # visibility = torch.cat([encoded_corners_2d_all[:,:,2::4],encoded_corners_2d_all[:,:,3::4]],dim=-1)
        encoded_corners_2d_all = encoded_corners_2d_all.view(N, M, 8, 4)
        encoded_corners_2d = encoded_corners_2d_all[:, :, :, :
                                                    2].contiguous().view(
                                                        N, M, -1)
        visibility = encoded_corners_2d_all[:, :, :, 2:].contiguous().view(
            N, M, -1)

        format_checker.check_tensor_shape(encoded_corners_2d, [None, None, 16])
        format_checker.check_tensor_shape(visibility, [None, None, 16])
        format_checker.check_tensor_shape(final_boxes_2d, [None, None, 4])

        encoded_corners_2d = encoded_corners_2d.view(N, M, 8, 2)
        encoded_front_plane = encoded_corners_2d[:, :, :4]
        encoded_rear_plane = encoded_corners_2d[:, :, 4:]

        front_plane = self.decode_with_bbox(encoded_front_plane,
                                            final_boxes_2d)
        rear_plane = self.decode_with_bbox(encoded_rear_plane, final_boxes_2d)
        # reoder the corners

        # shape(N,M, 8, 2)
        # front_deltas = front_plane[:, :, 0] - front_plane[:, :, 1]
        # rear_deltas = rear_plane[:, :, 0] - rear_plane[:, :, 1]
        # cond = (front_deltas[:, :, 0] * front_deltas[:, :, 1]) * (
        # rear_deltas[:, :, 0] * rear_deltas[:, :, 1]) < 0

        # rear_plane[cond] = rear_plane[cond][:, [1, 0, 3, 2]]
        # front_plane = Corner2DNearestCoder.reorder_boxes_4c_decode(front_plane)
        # rear_plane = Corner2DNearestCoder.reorder_boxes_4c_decode(rear_plane)

        corners_2d = torch.cat([front_plane, rear_plane], dim=2)
        return corners_2d[:, :, Order.reorder()]

    def reorder_boxes_4c_decode(self, boxes_4c):
        """
        Args:
            boxes_4c: shape(N, M, 4, 2)
        """
        cond = (boxes_4c[:, :, 0, 1] - boxes_4c[:, :, 1, 1]) < 0
        new_boxes_4c = torch.clone(boxes_4c)
        new_boxes_4c[cond] = boxes_4c[cond][:, [1, 0, 3, 2]]
        return new_boxes_4c

    def reorder_boxes_4c_encode(self, boxes_4c):
        """
        Reorder boxes_4c by clockwise
        Args:
            boxes_4c: shape(N, 4, 2)
        """
        cond = (boxes_4c[:, 1, 0] - boxes_4c[:, 0, 0]) > 0
        new_boxes_4c = torch.clone(boxes_4c)
        # import ipdb
        # ipdb.set_trace()
        new_boxes_4c[cond] = boxes_4c[cond][:, [1, 0, 3, 2]]
        return new_boxes_4c

    def encode_with_bbox(self, boxes_4c, label_boxes_2d):
        """
        start from right down, ordered by clockwise
        Args:
            plane_2d: shape(N, 4, 2)
            label_boxes_2d: shape(N, 4)
        """
        # import ipdb
        # ipdb.set_trace()
        # extend bbox to box_4c
        left_top = label_boxes_2d[:, :2]
        right_down = label_boxes_2d[:, 2:]
        left_down = label_boxes_2d[:, [0, 3]]
        right_top = label_boxes_2d[:, [2, 1]]
        label_boxes_4c = torch.stack(
            [right_down, left_down, left_top, right_top], dim=1)
        # label_boxes_4c = torch.stack(
        # [left_top, left_top, left_top, left_top], dim=1)

        label_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(
            label_boxes_2d.unsqueeze(0)).squeeze(0)

        # ordered like label_boxes_4c
        # import ipdb
        # ipdb.set_trace()
        boxes_4c = self.reorder_boxes_4c_encode(boxes_4c)
        return (boxes_4c - label_boxes_4c
                ) / label_boxes_2d_xywh[:, 2:].unsqueeze(1), boxes_4c

    def decode_with_bbox(self, encoded_boxes_4c, label_boxes_2d):
        """
        start from right down, ordered by clockwise
        Args:
            plane_2d: shape(N, 4, 2)
            label_boxes_2d: shape(N, 4)
        """
        # extend bbox to box_4c
        left_top = label_boxes_2d[:, :, :2]
        right_down = label_boxes_2d[:, :, 2:]
        left_down = label_boxes_2d[:, :, [0, 3]]
        right_top = label_boxes_2d[:, :, [2, 1]]

        label_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(label_boxes_2d)
        label_boxes_4c = torch.stack(
            [right_down, left_down, left_top, right_top], dim=2)
        # label_boxes_4c = torch.stack(
        # [left_top, left_top, left_top, left_top], dim=2)
        return encoded_boxes_4c * label_boxes_2d_xywh[:, :, 2:].unsqueeze(
            -2) + label_boxes_4c

    def encode(self, label_boxes_3d, label_boxes_2d, p2, image_info):
        corners_2d = geometry_utils.torch_boxes_3d_to_corners_2d(
            label_boxes_3d, p2)
        front_plane = corners_2d[:, Order.planes()[0]]
        rear_plane = corners_2d[:, Order.planes()[1]]
        encoded_front_plane, reorder_front_plane = self.encode_with_bbox(
            front_plane, label_boxes_2d)
        encoded_rear_plane, reorder_rear_plane = self.encode_with_bbox(
            rear_plane, label_boxes_2d)

        encoded_points = torch.cat(
            [encoded_front_plane, encoded_rear_plane], dim=1)
        # boxes_2d_filter = geometry_utils.torch_window_filter(corners_2d,
        # label_boxes_2d)
        image_shape = torch.tensor([0, 0, image_info[1], image_info[0]])
        image_shape = image_shape.type_as(corners_2d).view(1, 4)
        # DONE fix bugs of reorder for visibility
        reorder_corners_2d = torch.cat(
            [reorder_front_plane, reorder_rear_plane], dim=1)
        image_filter = geometry_utils.torch_window_filter(
            reorder_corners_2d, image_shape, deltas=200)
        visibility = image_filter
        # visibility = torch.cat(
        # [visibility[:, Order.planes()[0]], visibility[:, Order.planes()[1]]], dim=-1)
        encoded_all = torch.cat(
            [encoded_points, visibility.unsqueeze(-1).float()], dim=-1)

        return encoded_all.view(encoded_all.shape[0], -1)

    def encode_batch(self, label_boxes_3d, auxiliary_dict):
        label_boxes_2d = auxiliary_dict[constants.KEY_BOXES_2D]
        p2 = auxiliary_dict[constants.KEY_STEREO_CALIB_P2]
        image_info = auxiliary_dict[constants.KEY_IMAGE_INFO]
        batch_size = label_boxes_3d.shape[0]
        orients_batch = []
        for batch_ind in range(batch_size):
            orients_batch.append(
                self.encode(label_boxes_3d[batch_ind],
                            label_boxes_2d[batch_ind], p2[batch_ind],
                            image_info[batch_ind]))
        return torch.stack(orients_batch, dim=0)
