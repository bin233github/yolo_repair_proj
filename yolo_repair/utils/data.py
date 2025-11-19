# -*- coding: utf-8 -*-
"""
数据读取工具：从 data.yaml 读取 split 的 images/labels 目录；
按 success_ids / failure_ids 采样；提供 ROI 对齐所需的 GT 框与类别。
"""
import os, glob, yaml
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_data_yaml(path_yaml):
    data = yaml.safe_load(open(path_yaml,'r',encoding='utf-8'))
    root = Path(data['path'])
    names = data['names'] if isinstance(data['names'], list) else [data['names'][k] for k in sorted(data['names'].keys(), key=int)]
    return dict(
        root=root,
        images=dict(
            train=root / data.get('train','images/train'),
            val=root / data.get('val','images/val'),
            test=root / data.get('test','images/test'),
        ),
        labels=dict(
            train=root / 'labels/train',
            val=root / 'labels/val',
            test=root / 'labels/test',
        ),
        names=names
    )

def list_images(img_dir: Path):
    exts = ('*.jpg','*.jpeg','*.png','*.bmp')
    files = []
    for e in exts: files += glob.glob(str(img_dir/e))
    return sorted(files)

def read_id_list(txt_path):
    ids = []
    with open(txt_path,'r',encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if ln: ids.append(ln)
    return set(ids)

class YoloSplitDataset(Dataset):
    """
    逐图读取（不做复杂增强），返回：
    - img_bgr (H,W,3) 的 numpy
    - gt_xyxy: [N,4] float32（像素坐标）
    - gt_cls:  [N]   long
    - img_path, stem
    """
    def __init__(self, img_dir: Path, lbl_dir: Path, id_set=None):
        self.imgs = list_images(img_dir)
        self.lbl_dir = lbl_dir
        if id_set is not None:
            self.imgs = [p for p in self.imgs if Path(p).stem in id_set]

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        path = self.imgs[i]
        img = cv2.imread(path)
        h,w = img.shape[:2]
        stem = Path(path).stem
        lbl = Path(self.lbl_dir/(stem+'.txt'))
        gts = []
        if lbl.exists():
            for ln in open(lbl,'r',encoding='utf-8'):
                parts = ln.strip().split()
                if len(parts)<5: continue
                c, cx, cy, bw, bh = map(float, parts[:5])
                x1 = (cx - bw/2)*w; y1=(cy - bh/2)*h
                x2 = (cx + bw/2)*w; y2=(cy + bh/2)*h
                gts.append([int(c), x1,y1,x2,y2])
        gt = np.array(gts, dtype=np.float32)
        if gt.size==0:
            gt = np.zeros((0,5), dtype=np.float32)
        return img, gt, path, stem

def make_loader(img_dir, lbl_dir, id_set, batch_size=1, num_workers=4, shuffle=True):
    ds = YoloSplitDataset(img_dir, lbl_dir, id_set)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=lambda x: x)
