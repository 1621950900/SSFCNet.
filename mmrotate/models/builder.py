# Copyright (c) OpenMMLab. All rights reserved.
import warnings

# 导入MMDet全局模型注册器
from mmdet.models.builder import MODELS


# 定义旋转检测专用组件别名

# 旋转检测骨干网络
ROTATED_BACKBONES = MODELS
#旋转检测损失函数
ROTATED_LOSSES = MODELS
# 旋转检测的检测器模型
ROTATED_DETECTORS = MODELS
# 旋转检测感兴趣区域提取器
ROTATED_ROI_EXTRACTORS = MODELS
# 检测头 分类+回归+角度预测头
ROTATED_HEADS = MODELS
# 颈部融合模块
ROTATED_NECKS = MODELS
# 多个检测头共享的网络层
ROTATED_SHARED_HEADS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return ROTATED_BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return ROTATED_NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROTATED_ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head."""
    return ROTATED_SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return ROTATED_HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return ROTATED_LOSSES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return ROTATED_DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
