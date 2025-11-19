# -*- coding: utf-8 -*-
"""
（可选）修复后快速验证：在指定 split 上评估 mAP@0.5、统计“失败样本回收数”等。
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import glob

def list_images(img_dir):
    exts = ('*.jpg','*.jpeg','*.png','*.bmp')
    files = []
    for e in exts: files += glob.glob(str(Path(img_dir)/e))
    return sorted(files)

def quick_eval(yolo, split_img_dir, conf_thres=0.25, iou_thres=0.7, out_dir='runs_repair/verify'):
    os.makedirs(out_dir, exist_ok=True)
    # 用 ultralytics 的 val 更稳，这里简化直接 predict 一遍；详细 mAP 建议用 model.val
    from ultralytics.utils.metrics import ConfusionMatrix
    # 省略复杂计算：建议直接：
    metrics = yolo.val(data=yolo.overrides['data'], split=yolo.overrides.get('split','val'))
    # 保存结果
    metrics_df = pd.DataFrame([metrics.results_dict])
    metrics_df.to_csv(Path(out_dir)/'val_metrics.csv', index=False)
    print("[Verify] 指标已写入", Path(out_dir)/'val_metrics.csv')
