import argparse
import os

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter

from mmdet.apis import init_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

import sys

sys.path.insert(0, '/home/xhh/home/codespace/LSKNet-main/LSKNet-main')

import mmrotate.models.detectors.strip_rcnn
import mmrotate.models.backbones.stripnet
import mmrotate.models.roi_heads.bbox_heads.strip_head

def obb2poly_np(rbboxes):
    """
    将旋转框 [cx, cy, w, h, angle] 转成四点 polygon。
    angle 单位通常是弧度。
    输出 shape: [N, 8]
    """
    polys = []
    for box in rbboxes:
        cx, cy, w, h, angle = box[:5]

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        dw = w / 2
        dh = h / 2

        # 未旋转前四个角点，相对于中心点
        corners = np.array([
            [-dw, -dh],
            [ dw, -dh],
            [ dw,  dh],
            [-dw,  dh]
        ], dtype=np.float32)

        # 旋转矩阵
        rot = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ], dtype=np.float32)

        rotated = corners @ rot.T
        rotated[:, 0] += cx
        rotated[:, 1] += cy

        polys.append(rotated.reshape(-1))

    return np.array(polys, dtype=np.float32)


def draw_obb_proposals(img, proposals, score_thr=0.0, topk=200, out_file='rpn_vis.jpg'):
    """
    在原图上画 RPN proposals。
    proposals shape 一般是 [N, 6]:
        [cx, cy, w, h, angle, score]
    或者 [N, 5]:
        [cx, cy, w, h, angle]
    """
    img_show = img.copy()

    proposals = proposals.detach().cpu().numpy()

    if proposals.shape[1] >= 6:
        scores = proposals[:, 5]
        keep = scores >= score_thr
        proposals = proposals[keep]
        scores = scores[keep]

        order = np.argsort(-scores)
        proposals = proposals[order[:topk]]
        scores = scores[order[:topk]]
    else:
        proposals = proposals[:topk]
        scores = None

    rbboxes = proposals[:, :5]
    polys = obb2poly_np(rbboxes)

    for i, poly in enumerate(polys):
        pts = poly.reshape(4, 2).astype(np.int32)

        # 画旋转框
        cv2.polylines(
            img_show,
            [pts],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

        # 写 score
        if scores is not None:
            x, y = pts[0]
            cv2.putText(
                img_show,
                f'{scores[i]:.2f}',
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1
            )

    mmcv.imwrite(img_show, out_file)
    print(f'[OK] saved to {out_file}')
    print(f'[INFO] visualized proposals: {len(polys)}')


def get_rpn_proposals(model, data):
    """
    前向到 RPN 阶段，返回 proposal_list。
    """
    with torch.no_grad():
        img = data['img'][0]
        img_metas = data['img_metas'][0]

        if hasattr(img_metas, 'data'):
            img_metas = img_metas.data[0]

        if isinstance(img_metas, dict):
            img_metas = [img_metas]

        # 1. backbone + neck 提特征
        feats = model.extract_feat(img)

        # 2. RPN 前向
        rpn_outs = model.rpn_head(feats)

        # 3. RPN 输出解码成 proposals
        # 不同 MMDetection / MMRotate 版本接口可能不同
        if hasattr(model.rpn_head, 'get_bboxes'):
            proposal_list = model.rpn_head.get_bboxes(
                *rpn_outs,
                img_metas=img_metas,
                cfg=model.test_cfg.rpn
            )
        else:
            proposal_list = model.rpn_head.predict_by_feat(
                *rpn_outs,
                batch_img_metas=img_metas,
                cfg=model.test_cfg.rpn,
                rescale=False
            )

    return proposal_list


def build_data(cfg, img_path, device):
    """
    按 test pipeline 构造单张图像数据。
    """
    cfg = cfg.copy()

    # test pipeline
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    data = dict(
        img_info=dict(filename=img_path),
        img_prefix=None
    )

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        data = scatter(data, [torch.device(device)])[0]

    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/xhh/home/codespace/LSKNet-main/LSKNet-main/configs/strip_rcnn/strip_rcnn_s_fpn_1x_dior_le90.py', help='config file path')
    parser.add_argument('--checkpoint', default='/home/xhh/home/dataset/model_bank/strip_rcnn_s_dior.pth', help='checkpoint file path')
    parser.add_argument('--img', default="/home/xhh/home/dataset/DIOR/JPEGImages-test/JPEGImages-test/11777.jpg", help='image path')
    parser.add_argument('--out', default="/home/xhh/home/result/Strip-RCNN/dior_vis_rnp/vis_rpn_11777.jpg", help='output image path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--topk', type=int, default=300)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    cfg = Config.fromfile(args.config)

    model = init_detector(
        args.config,
        args.checkpoint,
        device=args.device
    )
    model.eval()

    # 注意：这里用全局 model，是为了 build_data 里判断 cuda
    globals()['model'] = model

    data = build_data(cfg, args.img, args.device)

    proposal_list = get_rpn_proposals(model, data)

    # 单张图，所以取第 0 张
    proposals = proposal_list[0]

    # 读原图
    img = mmcv.imread(args.img)

    draw_obb_proposals(
        img=img,
        proposals=proposals,
        score_thr=args.score_thr,
        topk=args.topk,
        out_file=args.out
    )