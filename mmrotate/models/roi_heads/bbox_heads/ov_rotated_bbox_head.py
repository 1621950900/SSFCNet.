# Copyright (c) 2026.
# Open-vocabulary rotated bbox head for MMRotate 0.x / MMDetection 2.x.
# This module is detector/backbone agnostic: it only consumes RoI features.

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models.losses import accuracy

from mmrotate.models.builder import ROTATED_HEADS, build_loss
from mmrotate.models.roi_heads.bbox_heads.convfc_rbbox_head import RotatedConvFCBBoxHead


def _load_text_prototypes(path: Union[str, Path]) -> torch.Tensor:
    """Load text prototypes from .npy/.npz/.pt/.pth.

    Expected shape: [num_classes, text_dim]. Do not include background.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'text_proto_path not found: {path}')

    if path.suffix == '.npy':
        tensor = torch.from_numpy(np.load(str(path))).float()
    elif path.suffix == '.npz':
        data = np.load(str(path))
        key = 'prototypes' if 'prototypes' in data else list(data.keys())[0]
        tensor = torch.from_numpy(data[key]).float()
    elif path.suffix in ['.pt', '.pth']:
        obj = torch.load(str(path), map_location='cpu')
        if isinstance(obj, dict):
            obj = obj.get('prototypes', obj.get('text_features', obj))
        if not torch.is_tensor(obj):
            raise TypeError('Prototype checkpoint must contain a Tensor or key prototypes/text_features')
        tensor = obj.float()
    else:
        raise ValueError(f'Unsupported prototype format: {path.suffix}')

    if tensor.dim() != 2:
        raise ValueError(f'Text prototypes must be 2D [C, D], got {tuple(tensor.shape)}')
    return F.normalize(tensor, dim=-1)


@ROTATED_HEADS.register_module()
class OVRotatedShared2FCBBoxHead(RotatedConvFCBBoxHead):
    """Open-vocabulary rotated bbox head.

    What it changes:
        fixed linear cls head -> RoI/text-prototype cosine matching.

    What it keeps:
        MMRotate RPN, RoI extractor, bbox target assignment, rotated bbox coder,
        bbox regression, rotated NMS and evaluator.

    Why it is easy to reuse:
        the head only depends on RoI features, so the backbone/detector can be
        Oriented R-CNN, Strip-R-CNN-like two-stage models, PKINet, LSKNet, etc.

    Important for open-vocabulary evaluation:
        - text prototypes are registered with persistent=False by default, so
          checkpoints trained with base vocab can be loaded with all-class vocab.
        - use reg_class_agnostic=True if train/test vocab sizes differ.
    """

    def __init__(self,
                 text_proto_path: str,
                 text_dim: int = 768,
                 learnable_temperature: bool = True,
                 temperature: float = 0.01,
                 bg_bias_init: float = 0.0,
                 learnable_text: bool = False,
                 save_text_prototypes: bool = False,
                 loss_align: Optional[dict] = dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1),
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # 关键修复：
        # 原始 bbox head 的 init_cfg 会初始化 fc_cls，
        # 但开放词表 head 不使用 fc_cls，所以必须关掉默认 init_cfg。
        self.init_cfg = None

        protos = _load_text_prototypes(text_proto_path)
        assert protos.size(0) == self.num_classes, (
            f'Prototype class count {protos.size(0)} must equal num_classes {self.num_classes}.')
        assert protos.size(1) == text_dim, (
            f'Prototype dim {protos.size(1)} must equal text_dim {text_dim}.')

        # Parent class creates fc_cls, but OV classification does not use it.
        # Setting it to None avoids state_dict shape mismatch when changing vocab size.
        self.fc_cls = None

        if learnable_text:
            self.text_prototypes = nn.Parameter(protos)
        else:
            self.register_buffer(
                'text_prototypes', protos, persistent=bool(save_text_prototypes))

        self.region_proj = nn.Sequential(
            nn.Linear(self.cls_last_dim, self.cls_last_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.cls_last_dim, text_dim)
        )

        # labels == num_classes is background in MMDetection/MMRotate bbox heads.
        self.bg_logit = nn.Parameter(torch.tensor(float(bg_bias_init)))

        logit_scale_init = np.log(1.0 / float(temperature))
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
        else:
            self.register_buffer('logit_scale', torch.tensor(logit_scale_init).float())

        self.loss_align = build_loss(loss_align) if loss_align is not None else None
        self.ov_last_region_emb = None
        self.ov_last_text_emb = None

    def set_text_prototypes(self,
                            prototypes: torch.Tensor,
                            class_names: Optional[Sequence[str]] = None):
        """Switch vocabulary at inference time.

        In normal MMRotate evaluation it is cleaner to load a config whose
        ``num_classes``, dataset classes, and ``text_proto_path`` already match.
        This method is useful for custom demos.
        """
        prototypes = F.normalize(prototypes.float(), dim=-1)
        if isinstance(self.text_prototypes, nn.Parameter):
            self.text_prototypes = nn.Parameter(prototypes.to(self.bg_logit.device))
        else:
            self.text_prototypes = prototypes.to(self.bg_logit.device)
        self.num_classes = int(prototypes.size(0))
        self.class_names = list(class_names) if class_names is not None else None

    def _forward_conv_fc(self, x):
        """Follow RotatedConvFCBBoxHead.forward but expose cls/reg features."""
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        return x_cls, x_reg

    @auto_fp16()
    def forward(self, x):
        x_cls, x_reg = self._forward_conv_fc(x)

        region_emb = self.region_proj(x_cls)
        region_emb = F.normalize(region_emb, dim=-1)

        text_emb = F.normalize(self.text_prototypes, dim=-1)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        fg_logits = logit_scale * region_emb @ text_emb.t()

        bg_logits = self.bg_logit.expand(fg_logits.size(0), 1)
        cls_score = torch.cat([fg_logits, bg_logits], dim=1)

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        self.ov_last_region_emb = region_emb
        self.ov_last_text_emb = text_emb
        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score, labels, label_weights, avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                losses['acc'] = accuracy(cls_score, labels)

                # Extra foreground-only CE over visual/text logits.
                if self.loss_align is not None:
                    pos_inds = (labels >= 0) & (labels < self.num_classes)
                    if pos_inds.any():
                        align_logits = cls_score[pos_inds, :self.num_classes]
                        align_labels = labels[pos_inds]
                        align_weights = label_weights[pos_inds]
                        losses['loss_ov_align'] = self.loss_align(
                            align_logits, align_labels, align_weights,
                            avg_factor=max(float(pos_inds.sum()), 1.0),
                            reduction_override=reduction_override)

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 5)[pos_inds]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 5)[
                        pos_inds, labels[pos_inds]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds],
                    bbox_weights[pos_inds],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
