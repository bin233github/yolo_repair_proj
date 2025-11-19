# -*- coding: utf-8 -*-
"""根据指定 YOLO 权重在数据集 split 上的检测结果，自动划分成功/失败样本列表。
成功样本：所有 GT 框都能被正确预测（同类且 IoU>=阈值）
失败样本：至少有 1 个 GT 未被正确预测。
"""
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm

def load_split_paths(data_yaml: str, split: str) -> Tuple[List[Path], Path]:
    yaml_path = Path(data_yaml)
    data = yaml.safe_load(open(data_yaml, 'r', encoding='utf-8'))
    root = Path(data['path'])
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    img_dir = root / f"images/{split}"
    lbl_dir = root / f"labels/{split}"
    img_paths = []
    for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
        img_paths += sorted(img_dir.glob(ext))
    return sorted(img_paths), lbl_dir

def load_gt(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if label_path.exists():
        for ln in open(label_path,'r',encoding='utf-8'):
            parts = ln.strip().split()
            if len(parts) < 5:
                continue
            c, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw/2) * img_w
            y1 = (cy - bh/2) * img_h
            x2 = (cx + bw/2) * img_w
            y2 = (cy + bh/2) * img_h
            boxes.append((int(c), [x1,y1,x2,y2]))
    if boxes:
        cls = np.array([b[0] for b in boxes], dtype=np.int64)
        xyxy = np.array([b[1] for b in boxes], dtype=np.float32)
    else:
        cls = np.zeros((0,), dtype=np.int64)
        xyxy = np.zeros((0,4), dtype=np.float32)
    return cls, xyxy

def iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray):
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    # boxes: [N,4]
    area1 = (boxes1[:,2]-boxes1[:,0]).clip(min=0) * (boxes1[:,3]-boxes1[:,1]).clip(min=0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clip(min=0) * (boxes2[:,3]-boxes2[:,1]).clip(min=0)
    inter = np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    for i in range(boxes1.shape[0]):
        x1 = np.maximum(boxes1[i,0], boxes2[:,0])
        y1 = np.maximum(boxes1[i,1], boxes2[:,1])
        x2 = np.minimum(boxes1[i,2], boxes2[:,2])
        y2 = np.minimum(boxes1[i,3], boxes2[:,3])
        inter_w = np.clip(x2 - x1, a_min=0, a_max=None)
        inter_h = np.clip(y2 - y1, a_min=0, a_max=None)
        inter[i] = inter_w * inter_h
    union = area1[:,None] + area2[None,:] - inter
    union = np.clip(union, a_min=1e-6, a_max=None)
    return inter / union

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights/best.pt')
    ap.add_argument('--data-yaml', default='datasets/neu_det_yolo/neu_det.yaml')
    ap.add_argument('--split', default='val')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou-thres', type=float, default=0.5, help='GT 匹配 IoU 阈值')
    ap.add_argument('--device', default='0')
    ap.add_argument('--success-out', default='data/success_ids.txt')
    ap.add_argument('--failure-out', default='data/failure_ids.txt')
    args = ap.parse_args()

    img_paths, lbl_dir = load_split_paths(args.data_yaml, args.split)
    if len(img_paths) == 0:
        raise SystemExit('No images found in split. check data yaml or split name')
    img_dir = img_paths[0].parent if img_paths else None

    device = args.device
    if isinstance(device, str):
        d = device.strip().lower()
        if d == 'cpu':
            device_arg = 'cpu'
        elif d.startswith('cuda'):
            device_arg = device
        elif d.isdigit():
            device_arg = f"cuda:{d}" if torch.cuda.is_available() else 'cpu'
        else:
            device_arg = device
    elif isinstance(device, int):
        device_arg = f"cuda:{device}" if torch.cuda.is_available() else 'cpu'
    else:
        device_arg = device
    model = YOLO(args.weights)
    model.to(device_arg)

    successes, failures = [], []
    pred_iter = model.predict(str(img_dir), imgsz=args.imgsz, conf=args.conf, iou=0.7,
                              device=device_arg, verbose=False, stream=True, batch=16)
    for r0 in tqdm(pred_iter, total=len(img_paths), desc='Predict'):
        img_path = Path(r0.path)
        img = str(img_path)
        preds = r0.boxes
        if preds is None:
            pred_cls = np.zeros((0,), dtype=np.int64)
            pred_boxes = np.zeros((0,4), dtype=np.float32)
        else:
            pred_boxes = preds.xyxy.cpu().numpy().astype(np.float32)
            pred_cls = preds.cls.cpu().numpy().astype(np.int64)
        # load GT
        import cv2
        im = cv2.imread(img)
        h, w = im.shape[:2]
        gt_cls, gt_boxes = load_gt(lbl_dir/(img_path.stem + '.txt'), w, h)
        if gt_boxes.shape[0] == 0:
            # treat no-gt as success if没有预测? 允许任意
            successes.append(img_path.stem)
            continue
        ious = iou_matrix(gt_boxes, pred_boxes)
        matched_preds = set()
        all_hit = True
        for gi in range(gt_boxes.shape[0]):
            best_j = -1
            best_iou = 0.0
            for pj in range(pred_boxes.shape[0]):
                if pj in matched_preds:
                    continue
                if pred_cls[pj] != gt_cls[gi]:
                    continue
                if ious[gi, pj] > best_iou:
                    best_iou = ious[gi, pj]
                    best_j = pj
            if best_j >= 0 and best_iou >= args.iou_thres:
                matched_preds.add(best_j)
            else:
                all_hit = False
                break
        if all_hit:
            successes.append(img_path.stem)
        else:
            failures.append(img_path.stem)
    successes = sorted(set(successes))
    failures = sorted(set(failures))
    Path(args.success_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.success_out,'w',encoding='utf-8') as f:
        for s in successes:
            f.write(s+'\n')
    with open(args.failure_out,'w',encoding='utf-8') as f:
        for s in failures:
            f.write(s+'\n')
    print(f'[Done] split={args.split}, success={len(successes)}, failure={len(failures)} -> {args.success_out}, {args.failure_out}')

if __name__ == '__main__':
    main()
