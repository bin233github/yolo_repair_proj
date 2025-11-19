# -*- coding: utf-8 -*-
import cv2
import numpy as np

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """
    与 Ultralytics 一致的 letterbox：返回 resized_img, ratio, (dw, dh)
    ratio: 缩放比例; (dw,dh): padding
    """
    shape = im.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2; dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def to_tensor_chw_rgb01(im_bgr, img_size=640):
    """BGR->RGB，letterbox到 img_size，归一化到 [0,1]，返回 tensor(NCHW=1) 与 meta"""
    import torch
    im = im_bgr.copy()
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_resz, r, (dw, dh) = letterbox(im_rgb, new_shape=(img_size, img_size))
    im_resz = im_resz.astype(np.float32) / 255.0
    im_chw = np.transpose(im_resz, (2,0,1))
    t = torch.from_numpy(im_chw).unsqueeze(0)
    meta = {"ratio": r, "pad": (dw, dh), "orig_hw": im_bgr.shape[:2], "inp_hw": (img_size, img_size)}
    return t, meta

def boxes_to_letterboxed_xyxy(boxes_xyxy, meta):
    """
    原图坐标下的 xyxy -> letterbox 后坐标（像素）
    """
    r = meta["ratio"]; dw, dh = meta["pad"]
    b = boxes_xyxy.copy()
    b[:, [0,2]] = b[:, [0,2]] * r + dw
    b[:, [1,3]] = b[:, [1,3]] * r + dh
    return b
