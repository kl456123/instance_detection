# -*- coding: utf-8 -*-

import torch.nn as nn
from lib.model.roi_layers import ROIAlign
from lib.model.roi_layers import AdaptiveROIAlign

from core.model import Model
from core.filler import Filler
from core import constants

from models.losses.focal_loss import FocalLoss

from utils.registry import DETECTORS
from utils import box_ops

from models import feature_extractors
from models import detectors

from core.instances import Instance
import samplers
from core.utils.common import UncoverDict


@DETECTORS.register('faster_rcnn')
class FasterRCNN(Model):
    def forward(self, feed_dict):
        output_dict = {}
        losses_dict = UncoverDict()
        auxiliary_dict = {}
        multi_stage_stats = []

        # base model
        base_feat = self.feature_extractor.first_stage_feature(
            feed_dict[constants.KEY_IMAGE])
        feed_dict.update({'base_feat': base_feat})

        # rpn model
        instance, losses = self.rpn_model.forward(feed_dict)

        # output_dict.update()
        auxiliary_dict.update(instance)
        proposals = auxiliary_dict[constants.KEY_BOXES_2D]

        for i in range(self.num_stages):

            if self.training:

                losses_units = self.instance_info.generate_losses(
                    output_dict, feed_dict, auxiliary_dict)

                losses_units, subsampled_mask = self.sampler.subsample_instance(
                    losses_units)
                proposals, _ = self.sampler.subsample_instance(
                    proposals, subsampled_mask)
                # update auxiliary dict
                # TODO subsample for all auxiliary_dict
                auxiliary_dict[constants.KEY_BOXES_2D] = proposals

                auxiliary_dict[constants.KEY_NUM_INSTANCES] = feed_dict[
                    constants.KEY_NUM_INSTANCES]
                multi_stage_stats.append(
                    self.instance_info.generate_stats(auxiliary_dict))

            rois = box_ops.box2rois(proposals)
            pooled_feat = self.rcnn_pooling(base_feat, rois.view(-1, 5),
                                            1 / 16)

            # shape(N,C,1,1)
            pooled_feat = self.feature_extractor.second_stage_feature(
                pooled_feat)
            pooled_feat = pooled_feat.mean(3).mean(2)

            # collect output from network to output_dict
            for attr_name in self.branches:
                attr_preds = self.branches[attr_name][i](pooled_feat)
                output_dict[attr_name] = attr_preds

            # unsqueeze before calc loss
            batch_size = rois.shape[0]
            output_dict = self.instance_info.unsqueeze(output_dict, batch_size)
            if self.training:
                losses_units.update_from_output(output_dict)

            # decode
            instance = self.instance_info.generate_instance(
                output_dict, auxiliary_dict)

        if self.training:
            losses_dict.update(losses)
            losses = self.instance_info.calc_loss(losses_units)
            losses_dict.update(losses)
        else:
            losses_dict = None
            multi_stage_stats = None
            # rescale
            im_info = feed_dict[constants.KEY_IMAGE_INFO]
            instance = self.instance_info.affine_transform(instance, im_info)
        return instance, losses_dict, multi_stage_stats

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        # init branches
        for attr_name in self.branches:
            Filler.normal_init(self.branches[attr_name][0], 0, 0.01,
                               self.truncated)

    def init_modules(self):
        self.feature_extractor = feature_extractors.build(
            self.feature_extractor_config)
        self.rpn_model = detectors.build(self.rpn_config)
        #  self.rcnn_pooling = ROIAlign(
        #  (self.pooling_size, self.pooling_size), 1.0 / 16.0, 2)
        self.rcnn_pooling = AdaptiveROIAlign(
            (self.pooling_size, self.pooling_size), 2)

        # construct  many branches for each attr of instance
        branches = {}
        for attr in self.instance_info:
            num_channels = self.instance_info[attr].num_channels
            branches[attr] = nn.ModuleList([
                nn.Linear(self.in_channels, num_channels)
                for _ in range(self.num_stages)
            ])
        self.branches = nn.ModuleDict(branches)

        self.rcnn_cls_loss = nn.CrossEntropyLoss(reduce=False)
        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

    def init_param(self, model_config):
        classes = model_config['classes']
        self.classes = classes
        self.n_classes = len(classes) + 1
        self.class_agnostic = model_config['class_agnostic']
        self.pooling_size = model_config['pooling_size']
        self.pooling_mode = model_config['pooling_mode']
        self.truncated = model_config['truncated']
        self.use_focal_loss = model_config['use_focal_loss']

        # some submodule config
        self.feature_extractor_config = model_config[
            'feature_extractor_config']
        self.rpn_config = model_config['rpn_config']

        self.instance_info = Instance(model_config['instance'])

        self.sampler = samplers.build(model_config['sampler_config'])

        self.num_stages = 1

        self.in_channels = model_config['in_channels']
