# -*- coding: utf-8 -*-
"""
YOLOv8 模型装载、分类末端 1×1 Conv 查找、输入特征钩子等工具。
全部以“分类支路 1×1 Conv 的输入/输出”为锚点，避免版本差异。
"""
import torch
import torch.nn as nn
from ultralytics import YOLO

# ---- 基础 ----
def load_yolov8_model(weights_path, device='cuda'):
    yolo = YOLO(weights_path)
    yolo.model.to(device)
    yolo.model.eval()
    return yolo

# ---- 查找分类末端 1×1 Conv（nc 输出） ----
def find_cls_1x1_convs(model, num_classes):
    """
    遍历所有模块，返回 (name, conv_module) 列表：
    - Conv2d
    - kernel_size == 1x1
    - out_channels == num_classes
    这通常对应 P3/P4/P5 三个尺度的分类末端 1×1 Conv。
    """
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and tuple(m.kernel_size)==(1,1) and m.out_channels==num_classes:
            out.append((name, m))
    if not out:
        raise RuntimeError("未找到分类末端 1×1 Conv，请检查模型结构或 num_classes。")
    return out

# ---- 钩子：抓“分类 1×1 Conv 的输入”作为 ROI 的特征图 ----
class InputFeatHook:
    """
    注册到分类 1×1 Conv 上，抓取其 **输入特征图**（即 conv 前的张量）。
    """
    def __init__(self, conv_module: nn.Conv2d):
        self.last_in = None
        self.h = conv_module.register_forward_hook(self._hook)

    def _hook(self, m, inp, out):
        # inp 是一个 tuple，仅取第一个
        self.last_in = inp[0].detach()

    def close(self):
        self.h.remove()

# ---- 选择尺度（P3/P4/P5）：取 H*W 最大者视为 P3 ----
def choose_p3_by_resolution(hooks):
    sizes = []
    for hk in hooks:
        t = hk.last_in
        assert t is not None and t.ndim==4, "钩子未捕获到输入特征，请先跑一次前向。"
        sizes.append(int(t.shape[2]) * int(t.shape[3]))
    p3_idx = int(torch.tensor(sizes).argmax().item())
    return p3_idx

# ---- 冻结/解冻 ----
def freeze_module(mod: nn.Module):
    for p in mod.parameters():
        p.requires_grad_(False)

def unfreeze_module(mod: nn.Module):
    for p in mod.parameters():
        p.requires_grad_(True)

# ---- 通过分类 conv 的“同一父层”去尝试找到“前置stem(conv/bn/act)”并解冻 ----
def try_unfreeze_stem_of_cls_conv(model, cls_conv_name: str):
    """
    经验性做法：以分类 1×1 conv 的名字为锚，尝试在相同 parent 下找到前置的 stem（如 cv2），
    若找到则只解冻该 stem；否则不做解冻（保持稳健）。
    """
    # 根据 name 走路径拿到 parent
    parts = cls_conv_name.split('.')
    parent = model
    for p in parts[:-2]:  # 去掉 ".conv" 前两级
        if not hasattr(parent, p):
            return False
        parent = getattr(parent, p)
    # 常见写法： parent.cv2 是 stem，parent.cv3.conv 是分类 1×1
    if hasattr(parent, 'cv2'):
        unfreeze_module(parent.cv2)
        return True
    return False
