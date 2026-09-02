import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from ..builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer
from pytorch_wavelets import DWTForward
import numpy as np

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GRN(nn.Module):
    """
    Global Response Normalization (ConvNeXt V2 style)
    x: (B, C, H, W) -> y = x + gamma * (x / ||x||_2) + beta
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))   # learnable scale
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))  # learnable bias
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)     # L2 over spatial dims
        nx = x / (gx + self.eps)
        return x + self.gamma * nx + self.beta




class DPAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(3, 1, 1)
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.high_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.modulator = nn.Sequential(
            nn.Conv2d(dim, dim, 1),  # PW
            nn.Sigmoid()
        )
        self.out_norm = GRN(dim)
        #self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x, debug=False):
        local_mean = self.avg_pool(x)
        local_max  = self.max_pool(x)

        contrast   = x - local_mean
        edge       = F.relu(local_max - x)
        high_feat = self.high_proj(contrast + edge)
        mod = self.modulator(high_feat)

        feat = local_mean + high_feat * (2.0 * mod)
        out = x + self.out_norm(feat)

        if debug:
            return out, {
                "low": local_mean,
                "local_max":local_max,
                "contrast":contrast,
                "edge":edge,
                "high": high_feat,
                "mod": mod,
                "feat": feat,
                "DPAM_out": out
            }

        return out


class SSAM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv_small = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_large = nn.Conv2d(dim, dim, 7, padding=9, dilation=3, groups=dim)

        self.conv_h = nn.Conv2d(dim, dim, (1,13), padding=(0,6), groups=dim) # self.conv_h = nn.Conv2d(dim, dim, (1,9), padding=(0,4), groups=dim)
        self.conv_v = nn.Conv2d(dim, dim, (13,1), padding=(6,0), groups=dim) #


    def forward(self, xx, yy, zz, debug=False):
        B, C, H, W = xx.shape

        f1 = self.conv_large(self.conv_small(xx))
        f2 = self.conv_v(self.conv_h(yy))
        f3 = zz

        fused = torch.cat([f1, f2, f3], dim=1)
        out = fused

        if debug:
            return out, {
                "f1":f1,
                "f2":f2,
                "f3":f3,
                "SDAM_out": out
            }
        return out


class DynamicModulatorv2(nn.Module):
    def __init__(self, c, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1, 1,
                              kernel_size=k_size, padding=(k_size-1)//2,bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x) # B,C,1,1
        y = y.squeeze(-1).transpose(-1,-2) # B, 1, C
        y = self.conv(y)
        y = y.transpose(-1,-2).unsqueeze(-1)
        y = self.act(y)
        return x * y


class SSCM_1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = dim // 4
        #self.reduce = nn.Conv2d(dim, hidden_dim*4, 1, groups=4)
        self.reduce = DynamicModulatorv2(dim)
        self.dpam = DPAM(hidden_dim)
        self.sdam = SSAM(hidden_dim)

        self.fusion = nn.Conv2d(dim, dim, 1)
        #self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * 1e-2)  # 1e-2

    def forward(self, x, debug=False):
        B, C, H, W = x.shape

        # ===== 低频 =====
        #xx = self.reduce(x)
        cx = self.reduce(x)
        xx, yy, zz, hh = list(cx.chunk(4, 1))
        if debug:
            small_feat, dism_dbg = self.dpam(hh, debug=True)
            feats, lsk_dbg = self.sdam(xx, yy, zz, debug=True)
        else:
            small_feat = self.dpam(hh)
            feats = self.sdam(xx, yy, zz)

        out = torch.cat([feats, small_feat], dim=1)
        att = self.fusion(out)

        out = x * att
        if debug:
            return out, {'channel_cs':cx,'att':att,'out':out, **dism_dbg, **lsk_dbg}

        return out



class SSCM_2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = dim // 4
        #self.reduce = nn.Conv2d(dim, hidden_dim*4, 1, groups=4)
        self.reduce = DynamicModulatorv2(dim)
        self.dpam = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        self.sdam = SSAM(hidden_dim)
        self.fusion = nn.Conv2d(dim, dim, 1)


    def forward(self, x, debug=False):
        B, C, H, W = x.shape

        # ===== 低频 =====
        cx = self.reduce(x)
        xx, yy, zz, hh = list(cx.chunk(4, 1))
        if debug:
            small_feat = self.dpam(hh)
            feats, lsk_dbg = self.sdam(xx, yy, zz, debug=True)
        else:
            small_feat = self.dpam(hh)
            feats = self.sdam(xx, yy, zz)

        out = torch.cat([feats, small_feat], dim=1)
        att = self.fusion(out)

        out = x * att
        if debug:
            return out, {'channel_cs': cx, 'att': att, 'out': out, 'small_feat':small_feat, **lsk_dbg}

        return out

class SemanticSpatialGateV3(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv_small = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_large = nn.Conv2d(
            dim, dim, 7, padding=9, dilation=3, groups=dim
        )

        self.conv_h = nn.Conv2d(dim, dim, (1,13), padding=(0,6), groups=dim) # self.conv_h = nn.Conv2d(dim, dim, (1,9), padding=(0,4), groups=dim)
        self.conv_v = nn.Conv2d(dim, dim, (13,1), padding=(6,0), groups=dim) #

        self.gate = nn.Sequential(
            nn.Conv2d(dim*3, 3, 1),
            nn.Softmax(dim=1)
        )
        #self.fuse = nn.Conv2d(dim*3, dim*3, 1, groups=3)

    def forward(self, xx, yy, zz, debug=False):

        f1 = self.conv_large(self.conv_small(xx))
        f2 = self.conv_v(self.conv_h(yy))
        f3 = zz
        fused = torch.cat([f1, f2, f3], dim=1)
        w = self.gate(fused)
        f1 = f1 * w[:, 0:1]
        f2 = f2 * w[:, 1:2]
        f3 = f3 * w[:, 2:3]
        out = torch.cat([f1, f2, f3], dim=1)

        if debug:
            return out, {
                "w_f1": f1,
                "w_f2": f2,
                "w_f3": f3,
                "SDAM_out": out
            }
        return out


class SSCM_3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = dim // 4
        self.reduce = DynamicModulatorv2(dim)
        self.dpam = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        self.sdam = SemanticSpatialGateV3(hidden_dim)

        self.fusion = nn.Conv2d(dim, dim, 1)
    def forward(self, x, debug=False):
        B, C, H, W = x.shape

        cx = self.reduce(x)
        xx, yy, zz, hh = list(cx.chunk(4, 1))
        if debug:
            small_feat = self.dpam(hh)
            feats, lsk_dbg = self.sdam(xx, yy, zz, debug=True)
        else:
            small_feat = self.dpam(hh)
            feats = self.sdam(xx, yy, zz)

        out = torch.cat([feats, small_feat], dim=1)
        att = self.fusion(out)
        out = x * att
        if debug:
            return out, {'channel_cs': cx, 'att': att, 'out': out, 'small_feat':small_feat, **lsk_dbg}

        return out

class Attention1(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SSCM_1(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        out = self.spatial_gating_unit(x)
        if isinstance(out, tuple):
            x, _ = out  # 只拿tensor
        else:
            x = out
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Attention2(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SSCM_2(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)

        out = self.spatial_gating_unit(x)
        if isinstance(out, tuple):
            x, _ = out
        else:
            x = out
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Attention3(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SSCM_3(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        # LSK 模块
        #out = self.spatial_gating_unit(x,debug=True)
        #print("Running SDAM")
        out = self.spatial_gating_unit(x)
        if isinstance(out, tuple):
            x, _ = out  # 只拿tensor
        else:
            x = out
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=None, stage_num=1):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        if stage_num==1:
        #if stage_num <=2:
            self.attn = Attention1(dim)
        elif stage_num <= 3:
            self.attn = Attention2(dim)
        else:
            self.attn = Attention3(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):

        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))

        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


@ROTATED_BACKBONES.register_module()
class SSFCNet(BaseModule):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4,
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths  # [3, 4, 6, 3]
        self.num_stages = num_stages  # 4

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            stage_number = i + 1
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j], norm_cfg=norm_cfg,stage_num=stage_number)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            import torch
            import argparse
            torch.serialization.add_safe_globals([argparse.Namespace])
            from mmcv.runner import load_checkpoint
            checkpoint_path = self.init_cfg['checkpoint']
            load_checkpoint(
                self,
                checkpoint_path,
                strict=False,
                logger=None
            )

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

