# -*- coding: utf-8 -*-
"""
阶段B：把“分类 1×1 Conv”看作线性层（只改这一层的少量输出通道），
把“失败样本修正 + 成功样本不伤害”写成 **线性约束**，最小化 L1(ΔW,Δb)，
用 LP 求解（ECOS/GLPK/OSQP），返回“可行解”或“无解”。
"""
import os, json
import numpy as np
import torch
import cvxpy as cp
from torchvision.ops import roi_align

from .utils.yolo_hooks import find_cls_1x1_convs, InputFeatHook, choose_p3_by_resolution
from .utils.geometry import to_tensor_chw_rgb01, boxes_to_letterboxed_xyxy
from .utils.data import make_loader, load_data_yaml, read_id_list

def build_roi_features(model, hook, loader, img_size, spatial_scale, device, maxN=6000):
    """
    用 ROIAlign 从“分类 1×1 Conv 的输入特征图”抽特征：
    - 输入：DataLoader（逐图），GT 框与类别
    - 输出：X:[N,C], y:[N]
    """
    Xs, Ys = [], []
    tot = 0
    for batch in loader:
        img_bgr, gt, path, stem = batch[0]
        if gt.shape[0]==0: continue
        t, meta = to_tensor_chw_rgb01(img_bgr, img_size)
        t = t.to(device)
        with torch.no_grad():
            _ = model(t)
            fmap = hook.last_in   # [1,C,H,W]
        boxes_lb = boxes_to_letterboxed_xyxy(gt[:,1:5].copy(), meta)
        rois = torch.from_numpy(np.concatenate([np.zeros((boxes_lb.shape[0],1)), boxes_lb], axis=1)).float().to(device)
        pooled = roi_align(fmap, rois, output_size=1, spatial_scale=spatial_scale, aligned=True).squeeze(-1).squeeze(-1)  # [N,C]
        labels = torch.from_numpy(gt[:,0]).long().to(device)
        Xs.append(pooled.cpu().numpy())
        Ys.append(labels.cpu().numpy())
        tot += pooled.size(0)
        if tot >= maxN: break
    if tot==0:
        return np.zeros((0, hook.last_in.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.concatenate(Xs, 0).astype(np.float64)
    y = np.concatenate(Ys, 0).astype(np.int64)
    return X, y

def pick_topk_out_channels_by_grad(model, cls_conv, hook, loader_neg, img_size, spatial_scale, device, K=16):
    """
    用一次“对真类 logit 的梯度敏感度”粗打分，挑选最相关的 K 个输出通道做变量。
    """
    if K<=0:   # 0 表示不用筛选，开放全部
        return np.arange(cls_conv.out_channels)

    model.zero_grad(set_to_none=True)
    for p in model.parameters(): p.requires_grad_(False)
    cls_conv.weight.requires_grad_(True)
    if cls_conv.bias is not None: cls_conv.bias.requires_grad_(True)

    scores = np.zeros(cls_conv.out_channels, dtype=np.float64)
    cnt = 0
    for batch in loader_neg:
        img_bgr, gt, path, stem = batch[0]
        if gt.shape[0]==0: continue
        t, meta = to_tensor_chw_rgb01(img_bgr, img_size)
        t = t.to(device)
        with torch.enable_grad():
            out = model(t)
            fmap = hook.last_in  # [1,C,H,W]
            boxes_lb = boxes_to_letterboxed_xyxy(gt[:,1:5].copy(), meta)
            rois = torch.from_numpy(np.concatenate([np.zeros((boxes_lb.shape[0],1)), boxes_lb], axis=1)).float().to(device)
            pooled = roi_align(fmap, rois, output_size=1, spatial_scale=spatial_scale, aligned=True).squeeze(-1).squeeze(-1)  # [N,C]
            labels = torch.from_numpy(gt[:,0]).long().to(device)
            # 对每个 ROI，选择对应真类的 logit = w_t · x + b_t
            for i in range(pooled.size(0)):
                cls = int(labels[i].item())
                # 取 y_true = W_t x + b_t，对于 conv.weight 的 grad 是 x
                # 在计算图里我们构造 s = y_true 的和（防止 retain_graph 太多）
                xi = pooled[i:i+1]  # 1,C
                y = (cls_conv.weight[cls:cls+1].view(1, -1) @ xi.t()).sum()
                if cls_conv.bias is not None:
                    y = y + cls_conv.bias[cls:cls+1].sum()
                model.zero_grad(set_to_none=True)
                y.backward(retain_graph=True)
                # grad w.r.t. weight [oc, ic,1,1]，对每个输出通道按 |grad| 聚合
                g = cls_conv.weight.grad[:, :, 0, 0].abs().sum(dim=1).detach().cpu().numpy()
                scores += g
                cnt += 1
                if cnt >= 64: break
        if cnt >= 64: break
    top_idx = np.argsort(-scores)[:K]
    return np.sort(top_idx)

def solve_lp_deltaW(W0, b0, X_fix, y_fix, X_keep, y_keep, m_fix=0.6, m_keep=0.2, l1_reg=1.0, out_mask=None):
    """
    线性规划：
    变量：
        dW_sel: [K, Cin]（只开放选中的输出通道），db_sel: [K]
    约束：
        对每个样本 i、每个负类 c≠t（若 c 不在 out_mask 内，对应 dW=0、db=0）：
        (W_t - W_c)·x + (b_t - b_c) + (dW_t - dW_c)·x + (db_t - db_c) >= margin
    目标：
        minimize l1_reg * (||dW_sel||_1 + ||db_sel||_1)
    返回：
        dW_full[nc,C], db_full[nc]
    """
    nc, C = W0.shape
    if out_mask is None:
        out_mask = np.arange(nc)
    K = len(out_mask)
    # 建立索引映射：全通道 <-> 变量通道
    pos = { int(out_mask[k]): k for k in range(K) }

    # 变量
    dW = cp.Variable((K, C))
    db = cp.Variable(K)

    cons = []

    def get_delta_row(cls_idx):
        # 若该输出类在变量集合内，返回 (row, bias_var)；否则返回 None
        if int(cls_idx) in pos:
            j = pos[int(cls_idx)]
            return dW[j, :], db[j]
        else:
            return None, None

    # 修复集合（失败样本）
    for i in range(X_fix.shape[0]):
        xi = X_fix[i]  # shape [C]
        t  = int(y_fix[i])
        for c in range(nc):
            if c == t: continue
            base = (W0[t]-W0[c]).dot(xi) + (b0[t]-b0[c])
            dt_w, dt_b = get_delta_row(t)
            dc_w, dc_b = get_delta_row(c)
            expr = base
            if dt_w is not None: expr = expr + dt_w @ xi + dt_b
            if dc_w is not None: expr = expr - (dc_w @ xi + dc_b)
            cons.append(expr >= m_fix)

    # 保持集合（成功样本）
    for i in range(X_keep.shape[0]):
        xi = X_keep[i]
        t  = int(y_keep[i])
        for c in range(nc):
            if c == t: continue
            base = (W0[t]-W0[c]).dot(xi) + (b0[t]-b0[c])
            dt_w, dt_b = get_delta_row(t)
            dc_w, dc_b = get_delta_row(c)
            expr = base
            if dt_w is not None: expr = expr + dt_w @ xi + dt_b
            if dc_w is not None: expr = expr - (dc_w @ xi + dc_b)
            cons.append(expr >= m_keep)

    # L1 最小化（引入非负变量上界）
    uW = cp.Variable((K, C), nonneg=True)
    ub = cp.Variable(K, nonneg=True)
    cons += [ dW <= uW, -dW <= uW, db <= ub, -db <= ub ]
    obj = cp.Minimize( l1_reg*(cp.sum(uW) + cp.sum(ub)) )

    prob = cp.Problem(obj, cons)
    status = None
    try:
        prob.solve(solver=cp.ECOS, verbose=False, abstol=1e-6, reltol=1e-6, feastol=1e-6, max_iters=20000)
        status = prob.status
    except Exception as e:
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            status = prob.status
        except Exception as e2:
            prob.solve(solver=cp.GLPK, verbose=False)
            status = prob.status

    if status not in ('optimal','optimal_inaccurate'):
        return None, None, status

    dW_sel = dW.value
    db_sel = db.value
    # 拼回全通道
    dW_full = np.zeros_like(W0)
    db_full = np.zeros_like(b0)
    for k, oc in enumerate(out_mask):
        dW_full[oc,:] = dW_sel[k,:]
        db_full[oc]   = db_sel[k]
    return dW_full, db_full, status
