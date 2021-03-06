# -*- coding: utf-8 -*-
"""
Utils for generating backbone
"""
from torchvision import models
from utils.registry import BACKBONES
from models.backbones.resnet18_pruned import resnet18

_net_arch_model_map = {
    'res18': models.resnet18,
    'res34': models.resnet34,
    'res50': models.resnet50,
    'res101': models.resnet101,
    'res152': models.resnet152,
    'res18_pruned': resnet18
}

_net_arch_fn_map = {
    'res18': 'resnet18-5c106cde.pth',
    'res34': 'resnet34-333f7ec4.pth',
    'res50': 'resnet50-19c8e357.pth',
    'res101': 'resnet101-5d3b4d8f.pth',
    'res152': 'resnet152-b121ed2d.pth',
    'res18_pruned': 'resnet18_pruned0.5.pth'
}


def register_all_backbones():
    for backbone_name in _net_arch_model_map:
        BACKBONES.register(backbone_name, _net_arch_model_map[backbone_name])


def build_backbone(net_arch):
    return _net_arch_model_map[net_arch]


def build_weights_fname(net_arch):
    _net_arch_fn_map[net_arch]
