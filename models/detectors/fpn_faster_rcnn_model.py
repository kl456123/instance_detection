# -*- coding: utf-8 -*-

import torch

from lib.model.roi_layers import AdaptiveROIAlign

from models.detectors.faster_rcnn_model import FasterRCNN
from core import constants

from models.losses import common_loss
from utils import box_ops
from utils.registry import DETECTORS


@DETECTORS.register('fpn')
class FPNFasterRCNN(FasterRCNN):
    def calculate_roi_level(self, rois_batch):
        h = rois_batch[:, 4] - rois_batch[:, 2] + 1
        w = rois_batch[:, 3] - rois_batch[:, 1] + 1
        roi_level = torch.log(torch.sqrt(w * h) / 224.0)

        isnan = torch.isnan(roi_level).any()
        assert not isnan, 'incorrect value in w: {}, h: {}'.format(w, h)

        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        # roi_level[...] = 2
        return roi_level

    def calculate_stride_level(self, idx):
        return 1 / ((idx + 1) * 8)

    def pyramid_rcnn_pooling(self, rcnn_feat_maps, rois_batch, input_size):
        pooled_feats = []
        box_to_levels = []
        # determine which layer to get feat
        # import ipdb
        # ipdb.set_trace()
        roi_level = self.calculate_roi_level(rois_batch)
        for idx, rcnn_feat_map in enumerate(rcnn_feat_maps):
            idx += 2
            mask = roi_level == idx
            rois_batch_per_stage = rois_batch[mask]
            if rois_batch_per_stage.shape[0] == 0:
                continue
            box_to_levels.append(mask.nonzero())
            feat_map_shape = rcnn_feat_map.shape[-2:]
            stride = feat_map_shape[0] / input_size[0]
            pooled_feats.append(
                self.rcnn_pooling(rcnn_feat_map, rois_batch_per_stage, stride))

        # (Important!)Note that you should keep it original order
        pooled_feat = torch.cat(pooled_feats, dim=0)
        box_to_levels = torch.cat(box_to_levels, dim=0).squeeze()
        idx_sorted, order = torch.sort(box_to_levels)
        pooled_feat = pooled_feat[order]
        assert pooled_feat.shape[0] == rois_batch.shape[0]
        return pooled_feat

    def forward(self, feed_dict):
        im_info = feed_dict[constants.KEY_IMAGE_INFO]

        auxiliary_dict = {}
        output_dict = {}
        losses_dict = {}

        # TODO move all auxiliary item from feed_dict to auxiliary_dict
        # before get data from dataloader
        if feed_dict.get(constants.KEY_STEREO_CALIB_P2) is not None:
            auxiliary_dict[constants.KEY_STEREO_CALIB_P2] = feed_dict[
                constants.KEY_STEREO_CALIB_P2]

        # base model
        rpn_feat_maps, rcnn_feat_maps = self.feature_extractor.first_stage_feature(
            feed_dict[constants.KEY_IMAGE])
        feed_dict.update({'base_feat': rpn_feat_maps})

        # rpn model
        #  prediction_dict.update(self.rpn_model.forward(feed_dict))
        instance, rpn_losses = self.rpn_model.forward(feed_dict)

        auxiliary_dict.update(instance)
        proposals = auxiliary_dict[constants.KEY_BOXES_2D]

        multi_stage_stats = []
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
            pooled_feat = self.pyramid_rcnn_pooling(rcnn_feat_maps,
                                                    rois.view(-1, 5),
                                                    im_info[0][:2])

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
            losses_dict.update(rpn_losses)
            losses = self.instance_info.calc_loss(losses_units)
            losses_dict.update(losses)
        else:
            losses_dict = None
            multi_stage_stats = None
            # rescale
            im_info = feed_dict[constants.KEY_IMAGE_INFO]
            instance = self.instance_info.affine_transform(instance, im_info)
        return instance, losses_dict, multi_stage_stats
