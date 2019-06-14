# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

from models.detectors.rpn_model import RPNModel
from utils.registry import DETECTORS
from core import constants


@DETECTORS.register('fpn_rpn')
class FPNRPNModel(RPNModel):
    def forward(self, feed_dict):
        output_dict = {}
        auxiliary_dict = {}

        base_feats = feed_dict['base_feat']
        batch_size = base_feats[0].shape[0]
        im_info = feed_dict[constants.KEY_IMAGE_INFO]

        for base_feat in base_feats:
            # rpn conv
            rpn_conv = F.relu(self.rpn_conv(base_feat), inplace=True)

            for attr_name in self.branches:
                attr_preds = self.branches[attr_name](rpn_conv)
                if attr_name in output_dict:
                    output_dict[attr_name].append(attr_preds)
                else:
                    output_dict[attr_name] = [attr_preds]

        # cat them
        #  for attr_name in output_dict:
            #  output_dict[attr_name] = torch.cat(output_dict[attr_name], dim=1)

        # generate pyramid anchors
        feature_map_list = [base_feat.shape[-2:] for base_feat in base_feats]
        anchors = self.anchor_generator.generate_pyramid(
            feature_map_list, im_info[0][:-1])
        anchors = anchors.unsqueeze(0).repeat(batch_size, 1, 1)
        auxiliary_dict[constants.KEY_BOXES_2D] = anchors

        # generate instance for next stage
        output_dict = self.instance.reshape_list(output_dict)
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
