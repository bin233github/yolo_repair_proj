


## 4) `scripts/make_id_lists_from_csv.py`


# -*- coding: utf-8 -*-
"""
从 false_negatives.csv / borderline_cases.csv 自动生成 failure_ids.txt；
把 split 目录下的所有图片 stem 减去 failure 得到 success_ids.txt。
"""

import os
import pandas as pd
from pathlib import Path
import argparse
import glob
import yaml

def list_images(img_dir):
    exts = ('*.jpg','*.jpeg','*.png','*.bmp')
    files = []
    for e in exts: files += glob.glob(str(Path(img_dir)/e))
    return sorted(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_yaml', default='datasets/neu_det_yolo/neu_det.yaml')
    ap.add_argument('--split', default='val')
    ap.add_argument('--fn_csv', default='data/false_negatives.csv')
    ap.add_argument('--bl_csv', default='data/borderline_cases.csv')
    ap.add_argument('--out_dir', default='data')
    args = ap.parse_args()

    data = yaml.safe_load(open(args.data_yaml,'r',encoding='utf-8'))
    root = Path(data['path'])
    img_dir = root / f"images/{args.split}"

    stems_all = [Path(p).stem for p in list_images(img_dir)]

    stems_fail = set()
    if os.path.exists(args.fn_csv):
        df = pd.read_csv(args.fn_csv)
        stems_fail |= set(Path(p).stem for p in df['image'].tolist())
    if os.path.exists(args.bl_csv):
        df = pd.read_csv(args.bl_csv)
        if 'image' in df.columns:
            stems_fail |= set(Path(p).stem for p in df['image'].tolist())

    # 输出 failure_ids.txt
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir)/'failure_ids.txt','w',encoding='utf-8') as f:
        for s in sorted(stems_fail):
            f.write(s+'\n')

    # success = all - failure
    stems_succ = [s for s in stems_all if s not in stems_fail]
    with open(Path(args.out_dir)/'success_ids.txt','w',encoding='utf-8') as f:
        for s in stems_succ:
            f.write(s+'\n')

    print(f"[OK] 写入 {args.out_dir}/failure_ids.txt ({len(stems_fail)}) 与 success_ids.txt ({len(stems_succ)})")

if __name__ == "__main__":
    main()
