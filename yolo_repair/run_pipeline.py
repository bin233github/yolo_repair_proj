# -*- coding: utf-8 -*-
"""
一键串联：
StageA（中间层可信修复） -> StageB（分类末端 LP 修复） -> （可选）验证
"""
import os, json, argparse, yaml
import torch
from pathlib import Path
import numpy as np
from .stageA_feature_repair import StageAFeatureRepairRunner
from .stageB_head_lp_repair import build_roi_features, pick_topk_out_channels_by_grad, solve_lp_deltaW
from .utils.data import load_data_yaml, read_id_list, make_loader
from .utils.yolo_hooks import find_cls_1x1_convs, InputFeatHook, choose_p3_by_resolution
from .utils.geometry import to_tensor_chw_rgb01

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config,'r',encoding='utf-8'))

    # ============= 阶段A：中间层可信修复（GD） =============
    A = StageAFeatureRepairRunner(cfg)
    A.build_prototypes(max_batches=cfg['stageA']['max_pos_batches'])
    A.train(epochs=cfg['stageA']['epochs'],
            alpha=cfg['stageA']['alpha_align'],
            beta=cfg['stageA']['beta_consist'],
            save_ckpt=cfg['stageA']['save_ckpt'])

    # ============= 阶段B：末端 LP 线性修复 =============
    # 1) 重新挂钩 P3 的分类 1×1 conv 的输入（确保与 StageA 一致）
    nc = len(A.class_names)
    cls_convs = find_cls_1x1_convs(A.model, nc)
    hooks = [InputFeatHook(conv) for _,conv in cls_convs]

    # 跑一次前向拿到特征图尺寸与 P3 索引
    import glob, cv2
    data_info = load_data_yaml(cfg['paths']['data_yaml'])
    split = cfg['paths']['split']
    img0 = sorted(glob.glob(str(data_info['images'][split] / '*.*')))[0]
    t, meta = to_tensor_chw_rgb01(cv2.imread(img0), img_size=cfg['img_size'])
    with torch.no_grad(): _ = A.model(t.to(A.device))
    p3_idx = choose_p3_by_resolution(hooks)
    hook = hooks[p3_idx]
    feat = hook.last_in
    Hf, Wf = int(feat.shape[2]), int(feat.shape[3])
    spatial_scale = Wf / float(cfg['img_size'])

    # 2) 构建“失败/成功”的 ROI 特征与标签
    succ_ids = read_id_list(cfg['paths']['success_ids'])
    fail_ids = read_id_list(cfg['paths']['failure_ids'])
    loader_pos = make_loader(data_info['images'][split], data_info['labels'][split], succ_ids, batch_size=1, shuffle=True, num_workers=2)
    loader_neg = make_loader(data_info['images'][split], data_info['labels'][split], fail_ids, batch_size=1, shuffle=True, num_workers=2)

    X_fix, y_fix   = build_roi_features(A.model, hook, loader_neg, cfg['img_size'], spatial_scale, A.device, maxN=cfg['stageB']['max_fix'])
    X_keep, y_keep = build_roi_features(A.model, hook, loader_pos, cfg['img_size'], spatial_scale, A.device, maxN=cfg['stageB']['max_keep'])
    print(f"[StageB] 失败 ROI 数={len(y_fix)}, 成功 ROI 数={len(y_keep)}")

    # 3) 取出 P3 的分类 1×1 Conv 权重
    target_name, cls_conv = cls_convs[p3_idx]
    W0 = cls_conv.weight.detach().cpu().numpy().reshape(cls_conv.out_channels, cls_conv.in_channels)
    b0 = cls_conv.bias.detach().cpu().numpy() if cls_conv.bias is not None else np.zeros(cls_conv.out_channels, dtype=np.float64)

    # 4) 选择开放变量的输出通道（Top-K 最大敏感度）
    K = int(cfg['stageB']['topk_out_channels'])
    out_mask = None
    if K>0 and K<cls_conv.out_channels:
        out_mask = pick_topk_out_channels_by_grad(A.model, cls_conv, hook, loader_neg, cfg['img_size'], spatial_scale, A.device, K)
        print(f"[StageB] 选中的输出通道（变量）索引：{out_mask.tolist()}")
    else:
        out_mask = np.arange(cls_conv.out_channels)
        print("[StageB] 开放全部输出通道作为变量（规模较大，慎用）。")

    # 5) 线性规划求解 ΔW, Δb
    dW, db, status = solve_lp_deltaW(
        W0, b0, X_fix, y_fix, X_keep, y_keep,
        m_fix=cfg['stageB']['m_fix'], m_keep=cfg['stageB']['m_keep'],
        l1_reg=cfg['stageB']['l1_reg'], out_mask=out_mask
    )
    os.makedirs(Path(cfg['stageB']['save_certificate']).parent, exist_ok=True)
    cert = {
        "status": status,
        "num_fix": int(len(y_fix)),
        "num_keep": int(len(y_keep)),
        "m_fix": cfg['stageB']['m_fix'],
        "m_keep": cfg['stageB']['m_keep'],
        "l1_reg": cfg['stageB']['l1_reg'],
        "topk_out_channels": cfg['stageB']['topk_out_channels'],
        "target_conv_name": target_name
    }

    if dW is None:
        with open(cfg['stageB']['save_certificate'],'w',encoding='utf-8') as f:
            json.dump(cert, f, ensure_ascii=False, indent=2)
        print(f"[StageB][RESULT] LP 无解，证书已写入 {cfg['stageB']['save_certificate']}。")
        return

    cert["norm1_dW"] = float(np.abs(dW).sum())
    cert["norm1_db"] = float(np.abs(db).sum())

    # 6) 应用 ΔW, Δb 到模型
    with torch.no_grad():
        W_new = W0 + dW
        b_new = b0 + db
        cls_conv.weight.copy_(torch.from_numpy(W_new).view_as(cls_conv.weight))
        if cls_conv.bias is not None:
            cls_conv.bias.copy_(torch.from_numpy(b_new))
    print("[StageB] 已应用 ΔW/Δb 到分类 1×1 Conv。")

    # 7) 保存权重与证书
    os.makedirs(Path(cfg['stageB']['save_pt']).parent, exist_ok=True)
    A.yolo.model = A.model
    A.yolo.save(cfg['stageB']['save_pt'])
    with open(cfg['stageB']['save_certificate'],'w',encoding='utf-8') as f:
        json.dump(cert, f, ensure_ascii=False, indent=2)
    print(f"[StageB][RESULT] 已保存 {cfg['stageB']['save_pt']} 与证书 {cfg['stageB']['save_certificate']}。")

if __name__ == "__main__":
    main()
