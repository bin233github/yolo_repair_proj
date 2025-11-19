# -*- coding: utf-8 -*-
"""
阶段A：中间层可信修复（GD 微调少量中间层），核心是：
- 用“成功样本”建立类原型（在“分类 1×1 conv 的输入特征图”上，做 ROIAlign -> 取均值，EMA 累积）
- 对“失败样本”最小化与对应类原型的 L2/Huber 距离（把特征拉回正确 pre-image）
- 对“成功样本”添加一致性（蒸馏）约束（修复前后 ROI 特征接近），减少副作用
- 只**解冻**分类末端**上一层的 stem（如 cv2）**或同层级极少参数；**不改**分类 1×1 conv 自身
"""
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import roi_align
from tqdm import tqdm
import numpy as np
from .utils.yolo_hooks import load_yolov8_model, find_cls_1x1_convs, InputFeatHook, choose_p3_by_resolution, try_unfreeze_stem_of_cls_conv, freeze_module, unfreeze_module
from .utils.data import load_data_yaml, read_id_list, make_loader
from .utils.geometry import to_tensor_chw_rgb01, boxes_to_letterboxed_xyxy

class EMAPrototypes(nn.Module):
    """每类一个向量原型，EMA 累积"""
    def __init__(self, num_classes, feat_dim, momentum=0.9, device='cuda'):
        super().__init__()
        self.register_buffer('vec', torch.zeros(num_classes, feat_dim, device=device))
        self.register_buffer('cnt', torch.zeros(num_classes, dtype=torch.long, device=device))
        self.m = momentum

    @torch.no_grad()
    def update(self, cls_ids, feats):
        for c in cls_ids.unique().tolist():
            m = (cls_ids==c)
            mu = feats[m].mean(0)
            if self.cnt[c] == 0:
                self.vec[c] = mu
            else:
                self.vec[c] = self.m*self.vec[c] + (1-self.m)*mu
            self.cnt[c] += int(m.sum())

    def get(self, cls_ids):
        return self.vec[cls_ids]

class StageAFeatureRepairRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        dev = cfg['stageA']['device']
        self.device = f"cuda:{dev}" if dev>=0 and torch.cuda.is_available() else "cpu"
        self.img_size = cfg['img_size']

        # 1) 载入模型
        self.yolo = load_yolov8_model(cfg['paths']['weights'], self.device)
        self.model = self.yolo.model

        # 2) 取 num_classes，查找所有分类 1×1 conv，并注册输入特征钩子
        data_info = load_data_yaml(cfg['paths']['data_yaml'])
        self.class_names = data_info['names']
        nc = len(self.class_names)
        self.nc = nc

        self.cls_convs = find_cls_1x1_convs(self.model, nc)
        self.hooks = [InputFeatHook(conv, detach=False) for (_,conv) in self.cls_convs]

        # 跑一次前向拿到各尺度特征图尺寸
        # 随便抽一张 split 图
        split = cfg['paths']['split']
        img_dir = data_info['images'][split]
        lbl_dir = data_info['labels'][split]
        import glob
        img0 = sorted(glob.glob(str(img_dir/'*.*')))[0]
        import cv2
        im = cv2.imread(img0)
        t, meta = to_tensor_chw_rgb01(im, img_size=self.img_size)
        with torch.no_grad():
            _ = self.model(t.to(self.device))
        # 选择 P3（分辨率最大）
        self.p3_idx = choose_p3_by_resolution(self.hooks)
        self.target_conv_name, self.target_cls_conv = self.cls_convs[self.p3_idx]
        print(f"[StageA] 选择 P3 分类 1×1 Conv：{self.target_conv_name}")

        # 3) 仅解冻该尺度的“前置 stem（如 cv2）”，其他全部冻结（保持结构）
        freeze_module(self.model)
        ok = try_unfreeze_stem_of_cls_conv(self.model, self.target_conv_name)
        if not ok:
            print("[StageA][Warn] 未能自动定位前置 stem，将回退到解冻分类 1×1 Conv 所在模块。")
            if not self._unfreeze_target_parent():
                unfreeze_module(self.target_cls_conv)

        # 4) 构建数据加载器（按 id 列表采样）
        succ_ids = read_id_list(cfg['paths']['success_ids'])
        fail_ids = read_id_list(cfg['paths']['failure_ids'])
        self.loader_pos = make_loader(img_dir, lbl_dir, succ_ids, batch_size=1, shuffle=True, num_workers=cfg['stageA']['num_workers'])
        self.loader_neg = make_loader(img_dir, lbl_dir, fail_ids, batch_size=1, shuffle=True, num_workers=cfg['stageA']['num_workers'])

        # 5) 构建类原型容器（维度来自一次前向后的 hook）
        feat = self.hooks[self.p3_idx].last_in   # [1, C, H, W]
        assert feat is not None and feat.ndim==4
        self.feat_dim = int(feat.shape[1])
        self.proto = EMAPrototypes(self.nc, self.feat_dim, momentum=0.9, device=self.device)

        # 6) 优化器（仅含解冻的参数）
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = optim.AdamW(params, lr=cfg['stageA']['lr'], weight_decay=cfg['stageA']['weight_decay'])

        # 7) 记录 spatial_scale（ROIAlign 用），自动计算：H_feat / img_size
        Hf, Wf = int(feat.shape[2]), int(feat.shape[3])
        self.spatial_scale = Wf / float(self.img_size)  # 宽高相等

    @torch.no_grad()
    def build_prototypes(self, max_batches=300):
        print("[StageA] 构建类原型（EMA）...")
        cnt = 0
        for batch in tqdm(self.loader_pos, total=min(len(self.loader_pos), max_batches)):
            img_bgr, gt, path, stem = batch[0]
            # 没 GT 就跳过
            if gt.shape[0]==0:
                continue
            # 取进入模型的 tensor 和 meta，并跑一次前向（hook 捕捉到输入特征图）
            t, meta = to_tensor_chw_rgb01(img_bgr, img_size=self.img_size)
            with torch.no_grad():
                _ = self.model(t.to(self.device))
                fmap = self.hooks[self.p3_idx].last_in   # [1,C,H,W]
            # GT 框转到 letterbox 坐标
            boxes = gt[:,1:5]
            boxes_lb = boxes_to_letterboxed_xyxy(boxes.copy(), meta)
            # 组 ROIs: [N,5]=(batch_idx, x1,y1,x2,y2)
            rois = torch.from_numpy(np.concatenate([np.zeros((boxes_lb.shape[0],1)), boxes_lb], axis=1)).float().to(self.device)
            pooled = roi_align(fmap, rois, output_size=1, spatial_scale=self.spatial_scale, aligned=True).squeeze(-1).squeeze(-1)  # [N,C]
            labels = torch.from_numpy(gt[:,0]).long().to(self.device)
            self.proto.update(labels, pooled)
            cnt += 1
            if cnt >= max_batches: break
        print("[StageA] 类原型构建完成。")

    def _loss_align(self, pooled, labels):
        target = self.proto.get(labels)
        return torch.nn.functional.mse_loss(pooled, target)

    def _loss_consistency(self, fmap_before, fmap_after, rois):
        p0 = roi_align(fmap_before, rois, output_size=1, spatial_scale=self.spatial_scale, aligned=True).squeeze(-1).squeeze(-1)
        p1 = roi_align(fmap_after,  rois, output_size=1, spatial_scale=self.spatial_scale, aligned=True).squeeze(-1).squeeze(-1)
        return torch.nn.functional.smooth_l1_loss(p1, p0)

    def _unfreeze_target_parent(self):
        parent = self.model
        parts = self.target_conv_name.split('.')
        for p in parts[:-1]:
            if not hasattr(parent, p):
                return False
            parent = getattr(parent, p)
        unfreeze_module(parent)
        return True

    def train(self, epochs=5, alpha=1.0, beta=0.1, save_ckpt='runs_repair/stageA_feature_repaired.safetensors'):
        print("[StageA] 开始中间层可信修复训练...")
        for ep in range(epochs):
            # 交替两个阶段：先修失败（对齐原型），再稳成功（保持一致）
            for phase, loader in (('neg', self.loader_neg), ('pos', self.loader_pos)):
                for batch in tqdm(loader, desc=f'E{ep+1}-{phase}', total=len(loader)):
                    img_bgr, gt, path, stem = batch[0]
                    if gt.shape[0]==0:
                        continue
                    # 前向前抓“before”特征
                    t, meta = to_tensor_chw_rgb01(img_bgr, img_size=self.img_size)
                    t = t.to(self.device)
                    with torch.no_grad():
                        _ = self.model(t)
                        fmap_before = self.hooks[self.p3_idx].last_in.clone()
                    # 前向（带梯度）
                    out = self.model(t)
                    fmap_after = self.hooks[self.p3_idx].last_in

                    boxes_lb = boxes_to_letterboxed_xyxy(gt[:,1:5].copy(), meta)
                    rois = torch.from_numpy(np.concatenate([np.zeros((boxes_lb.shape[0],1)), boxes_lb], axis=1)).float().to(self.device)
                    labels = torch.from_numpy(gt[:,0]).long().to(self.device)
                    pooled = roi_align(fmap_after, rois, output_size=1, spatial_scale=self.spatial_scale, aligned=True).squeeze(-1).squeeze(-1)

                    if phase=='neg':
                        loss = alpha * self._loss_align(pooled, labels)
                    else:
                        loss = beta  * self._loss_consistency(fmap_before, fmap_after, rois)

                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 5.0)
                    self.opt.step()
        # 保存快照（仅保存整个 model 的 state_dict，轻量）
        os.makedirs(Path(save_ckpt).parent, exist_ok=True)
        torch.save(self.model.state_dict(), save_ckpt)
        print(f"[StageA] 完成，已保存：{save_ckpt}")
