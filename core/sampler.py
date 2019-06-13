# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import torch
from .utils import format_checker
from utils import batch_ops


class Sampler(ABC):
    def __init__(self, sampler_config):
        """
        Note that the scores here is the critation of samples,it can be
        confidence or IoU e,c
        """
        self.fg_fraction = sampler_config['fg_fraction']
        self.num_samples = sampler_config['num_samples']

    @abstractmethod
    def subsample(self,
                  num_samples,
                  pos_indicator,
                  criterion=None,
                  indicator=None):
        pass

    def subsample_batch(self, pos_indicator, criterion=None, indicator=None):
        """
            batch version of subsample
        """
        pos_indicator = pos_indicator.detach()
        if indicator is None:
            indicator = torch.ones_like(pos_indicator)
        indicator = indicator.detach()

        # check format
        format_checker.check_tensor_dims(pos_indicator, 2)
        format_checker.check_tensor_dims(indicator, 2)

        batch_size = pos_indicator.shape[0]
        if criterion is None:
            criterion = [None] * batch_size
        else:
            criterion = criterion.detach()

        # assert self.num_samples % batch_size == 0, 'can not distribute samples evenly'
        if not self.num_samples % batch_size == 0:
            print('can not distribute samples evenly {}/{}'.format(
                self.num_samples, batch_size))
        num_samples_per_img = self.num_samples // batch_size
        num_samples = [num_samples_per_img for _ in range(batch_size)]
        num_remain = self.num_samples - batch_size * num_samples_per_img
        num_samples[0] = num_samples_per_img + num_remain

        sample_mask = []
        for i in range(batch_size):
            sample_mask.append(
                self.subsample(
                    num_samples[i],
                    pos_indicator[i],
                    criterion=criterion[i],
                    indicator=indicator[i]))

        sample_mask = torch.stack(sample_mask)
        return sample_mask

    def subsample_instance(self, loss_units, batch_sampled_mask=None):
        if batch_sampled_mask is None:
            # some params from loss_units
            pos_indicator = loss_units.pos_indicator
            indicator = loss_units.indicator
            cls_criterion = None
            batch_sampled_mask = self.subsample_batch(
                pos_indicator, indicator=indicator, criterion=cls_criterion)
        loss_units = batch_ops.filter_tensor_container(loss_units,
                                                       batch_sampled_mask)
        return loss_units, batch_sampled_mask

    # def subsample_instance(self, instance, batch_sampled_mask=None):
    # assert batch_sampled_mask is not None, \
    # 'it is not supported for subsample instance at present !'
    # instance = batch_ops.filter_tensor_container(
    # instance, batch_sampled_mask)
