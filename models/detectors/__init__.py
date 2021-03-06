# -*- coding: utf-8 -*-

from core.utils.imports import import_dir
import os

from core.utils.common import build as _build
from utils.registry import DETECTORS


def build(config):
    return _build(config, DETECTORS)


# import all for register all modules into registry dict
import_dir(os.path.dirname(__file__))

# only export build function to outside
__all__ = ['build']
