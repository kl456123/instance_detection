# -*- coding: utf-8 -*-
import torch

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils


@BBOX_CODERS.register('corner')
class CornerCoder(object):
    def __init__(self, config):
        pass

    def decode_batch(self, deltas, auxiliary_dict):
        """
        Args:
            deltas: shape(N, M, 4)
            boxes: shape(N, M, 4)
        """
        anchors = auxiliary_dict[constants.KEY_BOXES_2D]
        variances = [0.1, 0.2]
        anchors_xywh = geometry_utils.torch_xyxy_to_xywh(anchors)
        wh = anchors_xywh[:, :, 2:]
        xymin = anchors[:, :, :2] + deltas[:, :, :2] * wh * variances[0]
        xymax = anchors[:, :, 2:] + deltas[:, :, 2:] * wh * variances[0]
        return torch.cat([xymin, xymax], dim=-1)

    def encode_batch(self, gt_boxes, auxiliary_dict):
        """
        xyxy
        Args:
            anchors: shape(N, M, 4)
            gt_boxes: shape(N, M, 4)
        Returns:
            target: shape(N, M, 4)
        """
        anchors = auxiliary_dict[constants.KEY_BOXES_2D]
        variances = [0.1, 0.2]
        anchors_xywh = geometry_utils.torch_xyxy_to_xywh(anchors)
        wh = anchors_xywh[:, :, 2:]
        xymin = (gt_boxes[:, :, :2] - anchors[:, :, :2]) / (variances[0] * wh)
        xymax = (gt_boxes[:, :, 2:] - anchors[:, :, 2:]) / (variances[0] * wh)
        return torch.cat([xymin, xymax], dim=-1)
