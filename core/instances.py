# -*- coding: utf-8 -*-
import torch

import assigners
from models.losses import losses
from core import constants
import matchers
import similarity_calcs
from core.utils.analyzer import Analyzer
from core.utils import format_checker


class InstanceAssigner(dict):
    def __init__(self, config):
        super().__init__()
        # import ipdb
        # ipdb.set_trace()
        for attr_name in config:
            self[attr_name] = assigners.build(config[attr_name])


class Attr(object):
    def __init__(self, config):
        self.assigner = assigners.build(config['assigner_config'])
        self.loss = losses.build(config['losses_config'])
        self.num_channels = config['num_channels']


class AttrDict(dict):
    def __init__(self, config):
        for attr_name in config:
            self[attr_name] = Attr(config[attr_name])


class LossDict(dict):
    KEY_PREDS = 'preds'
    KEY_TARGETS = 'targets'
    KEY_WEIGHTS = 'weights'

    def update_loss_unit(self, name, loss_unit):
        if self.get(name) is not None:
            self[name].update(loss_unit)
        else:
            self[name] = loss_unit

    @property
    def pos_indicator(self):
        return self[self.KEY_POS_INDICATOR][self.KEY_WEIGHTS] > 0

    @property
    def KEY_INDICATOR(self):
        if constants.KEY_OBJECTNESS in self:
            return constants.KEY_OBJECTNESS
        elif constants.KEY_CLASSES in self:
            return constants.KEY_CLASSES
        else:
            return RuntimeError(
                'cannot specify key of indicator, please determine it youself')

    @property
    def KEY_POS_INDICATOR(self):
        if constants.KEY_BOXES_2D in self:
            return constants.KEY_BOXES_2D
        else:
            return RuntimeError('cannot specify key of pos indicator, \
                please determine it youself')

    @property
    def indicator(self):
        return self[self.KEY_INDICATOR][self.KEY_WEIGHTS] > 0

    def update_from_output(self, output_dict):
        for key in self:
            if key in output_dict:
                preds = output_dict[key]
                self.update_loss_unit(key, {'preds': preds})

    def get_preds(self, attr_name):
        return self[attr_name][self.KEY_PREDS]

    def get_targets(self, attr_name):
        return self[attr_name][self.KEY_TARGETS]

    def get_weights(self, attr_name):
        return self[attr_name][self.KEY_WEIGHTS]

    @staticmethod
    def calc_loss(module, targets, normalize=True):
        """
            Args:
                weight: shape(N)
                pred: shape(N, num_classes)
                target: shape(N)
            """
        weight = targets[LossDict.KEY_WEIGHTS]
        preds = targets[LossDict.KEY_PREDS]
        target = targets[LossDict.KEY_TARGETS]
        batch_size = weight.shape[0]

        # how to reshape them
        preds_shape = preds.shape
        target_shape = target.shape
        if len(preds_shape) == len(target_shape):
            # assume one2one match(reg loss)
            #  import ipdb
            #  ipdb.set_trace()
            loss = module(preds, target)
            format_checker.check_tensor_dims(loss, 3)
            loss = loss * weight.unsqueeze(-1)
            loss = loss.sum(dim=-1)

        elif len(preds_shape) == len(target_shape) + 1:
            # assume cls loss
            # weight = weight.view(-1)
            # target = target.view(-1)
            # preds = preds.view(-1, preds_shape[-1])

            loss = module(preds.view(-1, preds_shape[-1]), target.view(-1))
            format_checker.check_tensor_dims(loss, 1)
            loss = loss * weight.view(-1)
            loss = loss.view(batch_size, -1)
        else:
            raise ValueError('can not assume any possible loss type')

        if normalize:
            num_valid = (weight > 0).float().sum().clamp(min=1)
            return loss.sum() / num_valid
        else:
            return loss.sum()


class Instance(dict):
    def __init__(self, config):
        self.update(AttrDict(config['attr_config']))
        self.similarity_calc = similarity_calcs.build(
            config['similarity_calc_config'])
        self.matcher = matchers.build(config['matcher_config'])
        self.fg_thresh = config['fg_thresh']

        self.fake_fg_thresh = config.get('fake_fg_thresh', 0.7)

    @property
    def instance_assigners(self):
        instance_assigners = {}
        for attr_name in self:
            instance_assigners[attr_name] = self[attr_name].assigner
        return instance_assigners

    @property
    def instance_losses(self):
        instance_losses = {}
        for attr_name in self:
            instance_losses[attr_name] = self[attr_name].loss
        return instance_losses

    @property
    def instance_coders(self):
        instance_coders = {}
        for attr_name in self:
            instance_coders[attr_name] = self[attr_name].coder
        return instance_coders

    def generate_stats(self, auxiliary_dict):
        # rpn recall stats
        fake_match = auxiliary_dict[constants.KEY_FAKE_MATCH]
        num_instances = auxiliary_dict[constants.KEY_NUM_INSTANCES]
        append_num_gt = 0
        return Analyzer.analyze_recall(fake_match, num_instances,
                                       append_num_gt)

    def affine_transform(self, instance, image_info):
        """
        Args:
            image_info: shape(N, 4) ([h, w, scale_h, scale_w])
            instance: dict(N, M, num_channels)
        """
        boxes_2d = instance[constants.KEY_BOXES_2D]
        image_info = image_info.unsqueeze(-1).unsqueeze(-1)
        boxes_2d[:, :, ::2] = boxes_2d[:, :, ::2] / image_info[:, 3]
        boxes_2d[:, :, 1::2] = boxes_2d[:, :, 1::2] / image_info[:, 2]

        instance[constants.KEY_BOXES_2D] = boxes_2d
        return instance

    def generate_losses(self, output_dict, feed_dict, auxiliary_dict):
        proposals_primary = auxiliary_dict[constants.KEY_BOXES_2D]
        gt_primary = feed_dict[constants.KEY_BOXES_2D]

        # match them
        match_quality_matrix = self.similarity_calc.compare_batch(
            proposals_primary, gt_primary)
        num_instances = feed_dict[constants.KEY_NUM_INSTANCES]
        match, assigned_overlaps_batch = self.matcher.match_batch(
            match_quality_matrix, num_instances, self.fg_thresh)

        # used for stats
        fake_match, _ = self.matcher.match_batch(
            match_quality_matrix, num_instances, self.fake_fg_thresh)
        auxiliary_dict[constants.KEY_FAKE_MATCH] = fake_match

        auxiliary_dict[constants.KEY_MATCH] = match
        auxiliary_dict[
            constants.KEY_ASSIGNED_OVERLAPS] = assigned_overlaps_batch

        losses = LossDict()
        # assign targets and weights
        for attr_name in self.instance_assigners:
            assigner = self.instance_assigners[attr_name]
            # generate preds, targets and weights
            targets, weights = assigner.assign_targets_and_weights(
                feed_dict, auxiliary_dict)
            preds = output_dict.get(attr_name)

            # update losses
            losses.update_loss_unit(
                attr_name, {
                    LossDict.KEY_TARGETS: targets,
                    LossDict.KEY_WEIGHTS: weights,
                    LossDict.KEY_PREDS: preds
                })

        return losses

    def calc_loss(self, losses):

        loss_dict = dict()
        for attr_name in self.instance_losses:
            instance_loss_fn = self.instance_losses[attr_name]
            loss_dict[attr_name] = LossDict.calc_loss(instance_loss_fn,
                                                      losses[attr_name])
        return loss_dict

    def reshape(self, output_dict):
        """
        Reshape output dict to format like as (N, M, num_channels) from (N,C,H,W)
        """
        for attr_name in output_dict:
            attr_preds = output_dict[attr_name]
            attr = self[attr_name]
            batch_size = attr_preds.shape[0]
            num_channels = attr.num_channels
            attr_preds = attr_preds.permute(0, 2, 3, 1).contiguous().view(
                batch_size, -1, num_channels)
            output_dict[attr_name] = attr_preds
        return output_dict

    def reshape_list(self, output_dict):
        """
            reshape and cat
        """
        for attr_name in output_dict:
            attr_preds_list = output_dict[attr_name]
            attr = self[attr_name]

            for ind, attr_preds in enumerate(attr_preds_list):
                batch_size = attr_preds.shape[0]
                num_channels = attr.num_channels
                attr_preds = attr_preds.permute(0, 2, 3, 1).contiguous().view(
                    batch_size, -1, num_channels)
                output_dict[attr_name][ind] = attr_preds
            output_dict[attr_name] = torch.cat(output_dict[attr_name], dim=1)
        return output_dict

    def unsqueeze(self, output_dict, batch_size):
        for attr_name in output_dict:
            attr_preds = output_dict[attr_name]
            attr = self[attr_name]
            num_channels = attr.num_channels
            attr_preds = attr_preds.view(batch_size, -1, num_channels)
            output_dict[attr_name] = attr_preds
        return output_dict

    def generate_instance(self, output_dict, auxiliary_dict):
        """
        Each attr preds in output dict has the shape like as (N,C*num_anchors, H, W) or (N,C,H,W)
        reshape them first, then decode them,
        Args:
            output_dict: dict of network outputs
        """
        instance = {}
        for attr_name in output_dict:
            attr_preds = output_dict[attr_name].detach()
            attr = self[attr_name]

            # decode
            # use the same coder as like that in encoding time to decode them
            instance[attr_name] = attr.assigner.coder.decode_batch(
                attr_preds, auxiliary_dict)

        return instance
