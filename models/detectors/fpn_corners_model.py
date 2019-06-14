# -*- coding: utf-8 -*-

import torch

from models.detectors.fpn_faster_rcnn_model import FPNFasterRCNN
from core.filler import Filler
from core import constants

from utils.registry import DETECTORS
from utils import box_ops

from models import feature_extractors
from models import detectors


@DETECTORS.register('fpn_corners_2d')
class FPNCornersModel(FPNFasterRCNN):
    def init_weights(self):
        super().init_weights()

        # self.freeze_modules()
        # for param in self.rcnn_depth_preds.parameters():
        # param.requires_grad = True

        # for param in self.third_stage_feature.parameters():
        # param.requires_grad = True

        # self.freeze_bn(self)
        # self.unfreeze_bn(self.third_stage_feature)
