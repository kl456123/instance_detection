# -*- coding: utf-8 -*-
"""
Wrap pytorch loss
"""
import torch.nn as nn
import torch
from core.utils import format_checker
from core.instances import LossDict


# TODO (bugs of normalize=True)
# lr too small for sgd
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


# CrossEntropyLoss = WeightedLossWrapper(nn.CrossEntropyLoss)
# SmoothL1Loss = WeightedLossWrapper(nn.SmoothL1Loss)
