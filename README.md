# YOLO Repair Pipeline 使用指南

本仓库包含一个针对 YOLOv8 检测模型的两阶段“可信修复”流水线：

1. **Stage A – 中间层特征修复**：微调分类分支前的少量 stem，使失败样本的特征对齐到成功样本的“类原型”，并对成功样本加入一致性约束，保持既有正确预测。
2. **Stage B – 头部线性规划修复**：把分类 1×1 卷积视为线性层，构造“修复失败 / 保持成功”约束，并以 L1 最小化求解 ΔW/Δb，最终生成新的权重与 LP 证书。

下文提供在本地环境复现整个流程的分步说明，以及代码库内各文件/目录的作用介绍。

---

## 1. 环境准备

### 1.1 硬件/系统要求
- 建议使用具备 NVIDIA GPU 的 Linux 主机；若无 GPU，可将配置中的 `stageA.device`/`stageB.device` 设为 `-1` 或 `cpu`，但 Stage A 将显著变慢。
- 至少 10 GB 可用磁盘空间，用于模型权重、数据集和 `runs_repair/` 输出。

### 1.2 Python 依赖
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
依赖列表见 `requirements.txt`，其中 `ultralytics` 提供 YOLOv8，`cvxpy` 及其求解器 (`ecos`/`osqp`/`glpk`) 支持 Stage B 的线性规划，`opencv-python` 用于图像读写与 letterbox 预处理。

### 1.3 OpenCV `libGL` 依赖（可选）
若服务器缺失 `libGL.so.1` 导致 `cv2` 导入失败，可执行：
```bash
bash scripts/build_libgl_stub.sh
export LD_LIBRARY_PATH="$(pwd)/scripts:$LD_LIBRARY_PATH"
```
脚本会基于 `scripts/libgl_stub.c` 生成最小 stub，从而满足 OpenCV 的动态链接需求。

---

## 2. 数据与权重准备

1. **数据集**：`datasets/neu_det_yolo/neu_det.yaml` 指向 NEU-DET 数据。确保 YAML 中的 `path` 指向实际图片/标签根目录（`images/{train,val,test}` 与 `labels/{train,val,test}`）。如 YAML 中使用相对路径，`yolo_repair/utils/data.py` 会自动转为绝对路径。
2. **初始权重**：将待修复的 YOLOv8 权重（如 `best.pt`）放在 `weights/`，并在 `config.yaml` 的 `paths.weights` 指向该文件。
3. **划分成功/失败样本**：
   - 推荐运行 `scripts/generate_success_failure_ids.py`，它会在指定 split 上推理，并根据 IoU/类别匹配结果自动写入 `data/success_ids.txt` 与 `data/failure_ids.txt`：
     ```bash
     python scripts/generate_success_failure_ids.py \
        --weights weights/best.pt \
        --data-yaml datasets/neu_det_yolo/neu_det.yaml \
        --split val --imgsz 640 --conf 0.25
     ```
   - 若已有人为标注的失败样本 CSV，可使用 `scripts/make_id_lists_from_csv.py` 将 `false_negatives.csv` / `borderline_cases.csv` 转成 success/failure 列表。
4. **配置检查**：`config.yaml` 已给出默认路径、训练超参及 Stage B 约束，可根据实际 GPU/CPU 资源调整 `device`、`epochs`、`num_workers` 等字段。

---

## 3. 运行流水线

> **注意**：以下命令说明仅用于本地环境，请勿在只读沙箱中执行，以免生成无法提交的权重文件。

### Step 1 — 构建类原型（Stage A 预处理）
`python -m yolo_repair.run_pipeline` 会自动调用 `StageAFeatureRepairRunner.build_prototypes`，使用成功样本计算 EMA 类原型。可通过 `stageA.max_pos_batches` 限制用于构建原型的 batch 数量。

### Step 2 — Stage A 训练
Stage A 只解冻分类分支上一层的 stem（如 `cv2`），交替优化失败样本的“类对齐”损失与成功样本的“一致性”损失。`stageA.epochs` 默认为 1，可根据需求增加。完成后会在 `runs_repair/stageA_feature_repaired.safetensors` 保存中间层权重快照。

### Step 3 — Stage B 线性规划修复
1. `run_pipeline` 会重新挂钩 P3 分类 1×1 Conv 的输入特征，利用 `data/success_ids.txt` / `data/failure_ids.txt` 构建 ROI 特征矩阵：
   - `build_roi_features`：对每张图进行 letterbox、ROIAlign，提取分类 conv 输入特征 (`[N, C]`) 与标签。
2. `pick_topk_out_channels_by_grad`（若 `stageB.topk_out_channels>0`）会根据梯度敏感度挑选最相关的输出通道作为 LP 变量，降低求解规模。
3. `solve_lp_deltaW` 将“修复失败 ROI、保持成功 ROI”转成线性不等式，使用 `cvxpy` + ECOS/OSQP/GLPK 求解 L1 最小化问题，得到 ΔW、Δb：
   - `stageB.m_fix`：失败样本需要达到的 margin。
   - `stageB.m_keep`：成功样本保持的 margin。
   - `stageB.l1_reg`：L1 正则权重，越大越鼓励稀疏更新。
4. 求得的 ΔW/Δb 会写回模型并保存到 `runs_repair/stageAB_yolov8_repaired.pt`，同时生成 `runs_repair/lp_certificate.json` 记录 LP 状态、样本数、范数等信息。

### Step 4 — 可选验证
如需在本地验证修复效果，可调用 `yolo_repair/verify_after.py` 或直接使用 `ultralytics.YOLO.val`：
```bash
python -m yolo_repair.verify_after \
   --weights runs_repair/stageAB_yolov8_repaired.pt \
   --config config.yaml
```
（脚本会在 `runs_repair/verify/` 输出 `val_metrics.csv`。）

### 典型整合命令
```bash
python -m yolo_repair.run_pipeline --config config.yaml
```
流水线会依次完成 Stage A & Stage B，输出位于 `runs_repair/`。

---

## 4. 目录与文件说明

| 路径 | 类型 | 作用 |
| --- | --- | --- |
| `config.yaml` | 配置 | 定义数据/权重路径、图像尺寸、Stage A/B 超参与输出位置。 |
| `requirements.txt` | 依赖 | 声明需要安装的 Python 包。 |
| `REPAIR_REPORT.md` | 文档 | 记录上一轮修复的执行细节、结论与再现命令，可用于理解默认参数来源。 |
| `weights/` | 目录 | 存放原始 YOLOv8 权重以及修复过程中的权重文件。 |
| `data/` | 目录 | 保存 `success_ids.txt` / `failure_ids.txt` 等辅助文件。 |
| `datasets/neu_det_yolo/` | 数据 | 包含 `neu_det.yaml` 与实际图片/标签目录（NEU-DET）。 |
| `runs_repair/` | 输出 | Stage A/B 产物（特征快照、最终权重、LP 证书、验证指标等）。 |
| `scripts/generate_success_failure_ids.py` | 脚本 | 自动划分成功/失败样本 ID，供 Stage A/B 采样。 |
| `scripts/make_id_lists_from_csv.py` | 脚本 | 从人工标注 CSV 生成 success/failure 列表。 |
| `scripts/build_libgl_stub.sh` & `scripts/libgl_stub.c` | 脚本 | 用于编译 `libGL.so` stub，以修复 OpenCV 依赖缺失。 |
| `yolo_repair/run_pipeline.py` | 模块 | 串联 Stage A/B，并负责 LP 求解、保存证书。 |
| `yolo_repair/stageA_feature_repair.py` | 模块 | Stage A 主逻辑：构建类原型、只解冻 stem、交替训练。 |
| `yolo_repair/stageB_head_lp_repair.py` | 模块 | Stage B 主逻辑：ROI 特征提取、敏感度筛通道、构建/求解 LP。 |
| `yolo_repair/verify_after.py` | 模块 | （可选）修复后验证脚本。 |
| `yolo_repair/utils/data.py` | 工具 | 数据 YAML 解析、ID 过滤数据集、DataLoader 构建等。 |
| `yolo_repair/utils/geometry.py` | 工具 | letterbox、张量转换、BBox 坐标变换。 |
| `yolo_repair/utils/yolo_hooks.py` | 工具 | YOLOv8 模型加载、分类 1×1 Conv 检索、输入特征钩子、层冻结/解冻等。

---

## 5. 自定义与故障排查

1. **更换数据集**：修改 `config.yaml` 中的 `paths.data_yaml`，并确保新的 YAML `path` 字段正确；重新生成 success/failure 列表。
2. **单独运行 Stage A**：如需仅微调中间层，可直接调用 `StageAFeatureRepairRunner`：
   ```bash
   python -c "from yolo_repair.stageA_feature_repair import StageAFeatureRepairRunner;\
   import yaml; cfg=yaml.safe_load(open('config.yaml'));\
   A=StageAFeatureRepairRunner(cfg); A.build_prototypes(); A.train()"
   ```
3. **LP 无解**：若 `runs_repair/lp_certificate.json` 中 `status` 为 `infeasible`，可尝试：
   - 降低 `stageB.m_fix`（失败样本 margin）或 `m_keep`（成功 margin）。
   - 增大 `stageB.topk_out_channels` 以开放更多输出通道。
   - 减小 `stageB.l1_reg`，允许更大的 ΔW/Δb。
4. **CPU 环境调优**：`config.yaml` 中的 `stageA.epochs=1`、`num_workers=2` 是针对 CPU 的折衷；如显存充足，可增加 batch size / epochs 以提升修复强度。

---

## 6. 结果复现

根据 `REPAIR_REPORT.md`，一次完整运行应产生：
- `runs_repair/stageA_feature_repaired.safetensors`：Stage A 结束后中间层快照。
- `runs_repair/stageAB_yolov8_repaired.pt`：应用 Stage B ΔW/Δb 后的最终权重。
- `runs_repair/lp_certificate.json`：记录 LP 求解状态、样本数量、L1 范数等，可作为可验证证书。
- （可选）`runs_repair/verify/val_metrics.csv`：修复后指标。

如需再次复现，只需按照“环境准备 → 数据/权重准备 → 运行流水线”三步执行。
