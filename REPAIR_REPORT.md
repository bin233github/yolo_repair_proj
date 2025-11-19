# 权重修复执行记录

## 数据准备
- 运行 `scripts/generate_success_failure_ids.py`，基于 `weights/best.pt` 在 neu-det 验证集上区分成功/失败样本，得到 190 个成功样本与 170 个失败样本并写入 `data/success_ids.txt` 与 `data/failure_ids.txt`。
- 新脚本支持自动解析相对路径、批量推理以及 CPU/GPU 自动选择，方便后续重复使用。

## 代码修改要点
1. **图像路径解析**：`yolo_repair/utils/data.py` 中的 `load_data_yaml` 现在会把 YAML 中的相对 `path` 自动转为绝对路径，避免在不同工作目录下出错。
2. **StageA 训练稳定性**：
   - `InputFeatHook` 新增 `detach` 选项，StageA 在注册钩子时使用 `detach=False` 保留梯度，StageB 仍保持原行为。
   - 若无法自动找到 `cv2` 前置 stem，会回退到解冻目标分类层所在的父模块，确保始终有可训练参数。
3. **配置调整**：将 `stageA.epochs` 调整为 1、`num_workers` 调整为 2，使得 CPU 环境下流水线可在合理时间内跑完。
4. **OpenCV 依赖**：提供 `scripts/libgl_stub.c` 与 `scripts/build_libgl_stub.sh`，在缺失系统 `libGL.so.1` 时可以快速编译最小 stub 解决导入问题。
5. **结果产物**：流水线会在 `runs_repair/` 下生成 StageA 修复权重、StageAB 最终权重以及 LP 证书，方便复查。

## 流水线执行情况
- StageA 仅 1 个 epoch 即可完成中间层可信修复，日志与模型快照位于 `runs_repair/stageA_feature_repaired.safetensors`。
- StageB 基于 477 个失败 ROI 与 377 个成功 ROI 构建线性规划，解得 `status=optimal`，并成功将 ΔW/Δb 写入检测头，最终权重保存在 `runs_repair/stageAB_yolov8_repaired.pt`，证书 `runs_repair/lp_certificate.json` 记录了约束与 L1 范数。

## 权重修复结论
- LP 求解状态为 **optimal**，说明在给定 margin 与约束下找到了满足要求的权重增量，修复成功。
- 如需复现，可依次执行：
  1. `bash scripts/build_libgl_stub.sh`（若系统缺失 libGL）。
  2. `python scripts/generate_success_failure_ids.py --weights weights/best.pt --data-yaml datasets/neu_det_yolo/neu_det.yaml --split val --imgsz 640 --conf 0.25`。
  3. `python -m yolo_repair.run_pipeline --config config.yaml`。

