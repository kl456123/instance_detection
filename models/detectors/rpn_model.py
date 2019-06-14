# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F

from core.model import Model
from core.filler import Filler
from models.losses.focal_loss import FocalLoss

from utils import box_ops
from lib.model.roi_layers import nms
from utils.registry import DETECTORS
# import samplers
import anchor_generators
from core import constants
from core.instances import Instance
import samplers


@DETECTORS.register('rpn')
class RPNModel(Model):
    def init_param(self, model_config):
        self.in_channels = model_config['din']
        self.post_nms_topN = model_config['post_nms_topN']
        self.pre_nms_topN = model_config['pre_nms_topN']
        self.nms_thresh = model_config['nms_thresh']
        self.use_focal_loss = model_config['use_focal_loss']

        # anchor generator
        self.anchor_generator = anchor_generators.build(
            model_config['anchor_generator_config'])
        self.num_anchors = self.anchor_generator.num_anchors

        self.instance = Instance(model_config['instance'])
        self.sampler = samplers.build(model_config['sampler_config'])

    def init_weights(self):
        self.truncated = False

        Filler.normal_init(self.rpn_conv, 0, 0.01, self.truncated)

        for attr_name in self.branches:
            Filler.normal_init(self.branches[attr_name], 0, 0.01,
                               self.truncated)

    def init_modules(self):
        # define the convrelu layers processing input feature map
        self.rpn_conv = nn.Conv2d(self.in_channels, 512, 3, 1, 1, bias=True)

        branches = {}
        for attr in self.instance:
            num_channels = self.instance[attr].num_channels * self.num_anchors
            branches[attr] = nn.Conv2d(512, num_channels, 1, 1, 0)

        self.branches = nn.ModuleDict(branches)

    def postprocess(self, instance, im_info):
        # TODO create a new Function
        """
        Args:
        rpn_cls_probs: FloatTensor,shape(N,2*num_anchors,H,W)
        rpn_bbox_preds: FloatTensor,shape(N,num_anchors*4,H,W)
        anchors: FloatTensor,shape(N,4,H,W)

        Returns:
        proposals_batch: FloatTensor, shape(N,post_nms_topN,4)
        fg_probs_batch: FloatTensor, shape(N,post_nms_topN)
        """
        proposals = instance[constants.KEY_BOXES_2D]
        rpn_cls_probs = instance[constants.KEY_OBJECTNESS]

        batch_size = rpn_cls_probs.shape[0]

        # filer and clip
        proposals = box_ops.clip_boxes(proposals, im_info)

        # fg prob
        fg_probs = rpn_cls_probs[..., 1]

        # sort fg
        _, fg_probs_order = torch.sort(fg_probs, dim=1, descending=True)

        proposals_batch = torch.zeros(batch_size, self.post_nms_topN,
                                      4).type_as(proposals)
        proposals_order = torch.zeros(
            batch_size, self.post_nms_topN).fill_(-1).type_as(fg_probs_order)

        for i in range(batch_size):
            proposals_single = proposals[i]
            fg_probs_single = fg_probs[i]
            fg_order_single = fg_probs_order[i]
            # pre nms
            if self.pre_nms_topN > 0:
                fg_order_single = fg_order_single[:self.pre_nms_topN]
            proposals_single = proposals_single[fg_order_single]
            fg_probs_single = fg_probs_single[fg_order_single]

            # nms
            keep_idx_i = nms(proposals_single, fg_probs_single,
                             self.nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            # post nms
            if self.post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:self.post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            fg_probs_single = fg_probs_single[keep_idx_i]
            fg_order_single = fg_order_single[keep_idx_i]

            # padding 0 at the end.
            num_proposal = keep_idx_i.numel()
            proposals_batch[i, :num_proposal, :] = proposals_single
            # fg_probs_batch[i, :num_proposal] = fg_probs_single
            proposals_order[i, :num_proposal] = fg_order_single

        instance[constants.KEY_BOXES_2D] = proposals_batch
        # TODO(assign rpn_cls_probs)

        return instance

    def forward(self, feed_dict):
        output_dict = {}
        auxiliary_dict = {}

        base_feat = feed_dict['base_feat']
        batch_size = base_feat.shape[0]
        im_info = feed_dict[constants.KEY_IMAGE_INFO]

        # rpn conv
        rpn_conv = F.relu(self.rpn_conv(base_feat), inplace=True)

        for attr_name in self.branches:
            attr_preds = self.branches[attr_name](rpn_conv)
            output_dict[attr_name] = attr_preds

        # generate anchors
        feature_map_list = [base_feat.size()[-2:]]
        anchors = self.anchor_generator.generate(feature_map_list,
                                                 im_info[0][:-1])
        anchors = anchors.unsqueeze(0).repeat(batch_size, 1, 1)
        auxiliary_dict[constants.KEY_BOXES_2D] = anchors

        output_dict = self.instance.reshape(output_dict)
        instance = self.instance.generate_instance(output_dict, auxiliary_dict)
        instance = self.postprocess(instance, im_info)

        if self.training:
            # generate loss units first
            losses_units = self.instance.generate_losses(
                output_dict, feed_dict, auxiliary_dict)

            # then subsample
            losses_units, _ = self.sampler.subsample_instance(losses_units)

            # at last calc loss
            losses = self.instance.calc_loss(losses_units)
        else:
            losses = None

        return instance, losses

    def append_gt(self, proposals_batch, label_boxes_2d):
        """
        Args:
            proposals_batch: shape(N, M, 4)
            label_boxes_2d: shape(N, m, 4)
            num_instances: shape(N,) valid num of bboxes in each image
        Returns:
            proposals_batch: shape(N, M+m, 4)
        """
        return torch.cat([proposals_batch, label_boxes_2d], dim=1)
