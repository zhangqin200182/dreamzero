# DreamZero NPU 训练操作手册

## 架构

```
本地机器 ──SSH──→ 113.46.41.54 (跳板机) ──SSH──→ 192.168.106.239 (NPU 训练服务器)
                                                      ├── 16 × Ascend 910 (64GB/卡)
                                                      ├── Docker: dreamzero_train
                                                      ├── 代码: /workspace/dreamzero/ (bind mount)
                                                      ├── 数据: /data/droid/ (bind mount)
                                                      └── 模型: /checkpoints/ (bind mount)
```

**关键原则**：所有操作都必须指向 192.168.106.239（通过 ProxyCommand 跳板机转发），绝不能直接在跳板机执行 Docker 命令。

## 1. 连接服务器

### SSH 命令模板

```bash
sshpass -p 'Huawei@123' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
  -o ProxyCommand="sshpass -p 'Huawei@123' ssh -o StrictHostKeyChecking=no -W %h:%p root@113.46.41.54" \
  root@192.168.106.239 "<your-command>"
```

### 简化（写入 ~/.ssh/config 后可直接 `ssh 239`）

```
Host 239
    Hostname 192.168.106.239
    User root
    ProxyCommand sshpass -p 'Huawei@123' ssh -W %h:%p root@113.46.41.54
```

### 网络问题

- 跳板机密码认证偶尔超时/被拒，重试 1-2 次通常恢复
- 连续 3 次失败则停止，可能需更换 IP

## 2. 容器管理

### 查看状态

```bash
# 在 239 主机上
docker ps -a | grep dreamzero_train
```

### 启动容器（如果停了）

```bash
docker start dreamzero_train
```

### 确认 NPU 空闲

```bash
npu-smi info | grep -E "NPU|AICore|Memory"
# AICore 应为 0%，Memory-Usage 应为基线 ~3000/65536 MB
```

### 容器重启后验证修复

容器通过 bind mount 共享主机文件，代码修改持久化。但如果容器被删除重建，需要重新部署修复。

```bash
docker exec dreamzero_train bash -c '
  grep -c set_device /workspace/dreamzero/groot/vla/experiment/base.py   # 应为 1
  grep -c save_optimizer /workspace/dreamzero/groot/vla/experiment/base.py # 应为 2
  grep -c RESUME_CKPT /workspace/dreamzero/groot/vla/experiment/base.py    # 应为 2
  grep -c "if model is None" /workspace/dreamzero/groot/vla/experiment/base.py # 应为 >=1 (Fix 7)
  grep -c "CRITICAL: actually load" /workspace/dreamzero/groot/vla/experiment/base.py # 应为 1
'
```

如果缺失，重新部署（见第 5 节）。

## 3. 训练

### 训练脚本

脚本路径：`/workspace/dreamzero/scripts/train/droid_16gpu.sh`

关键参数：
| 参数 | 含义 | 默认值 |
|------|------|--------|
| `output_dir` | checkpoint 保存目录 | `/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v2` |
| `max_steps` | 总训练步数 | `100000` |
| `save_steps` | 每 N 步保存 checkpoint | `200` |
| `per_device_train_batch_size` | 每卡 batch | `1` |
| `gradient_accumulation_steps` | 梯度累积 | `2` |
| `learning_rate` | 学习率 | `1e-4` |
| `warmup_ratio` | warmup 比例 | `0.05` |
| `save_lora_only` | 仅保存 LoRA 权重（关键！） | `true` |

完整的 Hydra CLI 参数参见脚本内容。

### 从头启动训练

```bash
docker exec -d -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512 \
  dreamzero_train bash -c 'cd /workspace/dreamzero && \
  nohup bash scripts/train/droid_16gpu.sh \
  output_dir=/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v4 \
  &> /tmp/train_v11.log &'
```

**必须**：`-e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512`，否则 NPU 内存碎片导致 OOM。

### 从 checkpoint 续训

```bash
docker exec -d -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512 \
  -e RESUME_CKPT=/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v3/checkpoint-1000 \
  dreamzero_train bash -c 'cd /workspace/dreamzero && \
  nohup bash scripts/train/droid_16gpu.sh \
  output_dir=/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v4 \
  &> /tmp/train_v11.log &'
```

**重要**：
- `RESUME_CKPT` 环境变量指向要恢复的 checkpoint 目录（Fix 4）
- 使用新的 `output_dir` 避免覆盖已有 checkpoint（脚本通过 `"$@"` 透传 Fix）

### 启动后验证

```bash
# 约 5-10 分钟后检查（模型加载需时间）
docker exec dreamzero_train bash -c '
  echo -n "进程数: "; ps aux | grep -c "[e]xperiment.py"   # 应为 33（1 launcher + 16 ranks × 2）
  echo -n "当前步数: "; grep -oE "[0-9]+/100000" /tmp/train_v11.log | tail -1
  echo -n "续训确认: "; grep -c "Loaded LoRA adapter" /tmp/train_v11.log  # 应为 >=1，证明 Fix 7 生效
  echo -n "错误: "; grep -ciE "Traceback|OOM|childfailed" /tmp/train_v11.log
'
```

### 周期监控命令

```bash
# 每 5-10 分钟运行一次
docker exec dreamzero_train bash -c '
  T=$(date +%H:%M)
  S=$(grep -oE "[0-9]+/100000" /tmp/train_v11.log | tail -1)
  E=$(grep -ciE "Traceback|OOM|sigkill|childfailed" /tmp/train_v11.log)
  P=$(ps aux | grep -c "[e]xperiment.py")
  C=$(ls -d /data/droid/checkpoints/dreamzero_droid_npu_16gpu_v4/checkpoint-* 2>/dev/null | wc -l)
  D=$(df -h /checkpoints | awk "NR==2{print \$5}")
  echo "T=$T S=$S E=$E P=$P C=$C D=$D"
'
```

### 训练日志位置

| Run | 日志文件 | 步数范围 | 输出目录 |
|-----|---------|---------|---------|
| v9 (Run2) | `/tmp/train_v9.log` | 1000→2000 | `/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v2` |
| v10 (Run3) | `/tmp/train_v10.log` | 2000→3000 | `/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v3` |
| v11 (下次) | `/tmp/train_v11.log` | 3000→... | `/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v4` (推荐) |

## 4. 评测

### 评测脚本

`/workspace/dreamzero/scripts/eval/batch_eval.py`

功能：对所有 checkpoint 做 Full（视频+动作联合去噪）和 Action-Only（跳过视频去噪）双模式评测。

```bash
cd /workspace/dreamzero && \
python scripts/eval/batch_eval.py \
  --checkpoint_root /data/droid/checkpoints/dreamzero_droid_npu_16gpu_v3 \
  --output_root eval_new \
  --num_samples 10 \
  --seed 42
```

输出结构：
```
eval_new/
├── step_0200/
│   ├── full/ep_00.png ... ep_09.png    ← Full 模式预测图
│   ├── action_only/ep_00.png ...       ← AO 模式预测图
│   └── summary.json                    ← 该 checkpoint 的 MSE 汇总
├── step_0400/...
└── all_summaries.jsonl                 ← 全部 checkpoint 汇总
```

### 注意事项

- **评测必须停止训练**：两种模式都需要独占 NPU（14B 模型加载后无剩余显存）
- **端口冲突**：脚本使用 `MASTER_PORT=29501` 初始化 HCCL，确保无其他进程占用
- 可通过 `MASTER_PORT=29502` 环境变量改用其他端口

## 5. 代码修复部署

当容器重建后，需要重新部署 Fix 1-7。部署方法：本地编辑 → scp 到主机 bind-mount 路径。

```bash
# 1. 从容器拉取当前版本（可选，用于对比）
scp ... base.py /tmp/base.py

# 2. 本地编辑 /tmp/base.py

# 3. 语法检查
python3 -m py_compile /tmp/base.py

# 4. scp 部署到 bind-mount 路径
sshpass -p 'Huawei@123' scp -o StrictHostKeyChecking=no \
  -o ProxyCommand="sshpass -p 'Huawei@123' ssh -o StrictHostKeyChecking=no -W %h:%p root@113.46.41.54" \
  /tmp/base.py root@192.168.106.239:/workspace/dreamzero/groot/vla/experiment/base.py

# 5. 容器内验证
docker exec dreamzero_train grep -c "if model is None" /workspace/dreamzero/groot/vla/experiment/base.py
```

Bind mount 路径：`/workspace/dreamzero/`（主机和容器共享），所以直接 scp 到主机路径即可。

## 6. Checkpoint 管理

### 检查 checkpoint 质量

```bash
docker exec dreamzero_train bash -c '
  D=/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v3/checkpoint-1000
  echo "大小: $(du -sh $D | cut -f1)"     # 应 ~904M
  ls $D/adapter_model.safetensors         # 应存在，~74M
  ls $D/pytorch_model_fsdp.bin 2>/dev/null && echo "BAD 62GB!" || echo "OK"  # 应不存在！
  grep -o "global_step\": [0-9]*" $D/trainer_state.json
'
```

### 清理 62GB 残留文件

旧训练留下的 `pytorch_model_fsdp.bin`（62GB/个）占大量空间：

```bash
docker exec dreamzero_train bash -c '
  for d in /data/droid/checkpoints/dreamzero_droid_npu_16gpu_v2/checkpoint-*/; do
    rm -v "$d/pytorch_model_fsdp.bin" 2>/dev/null
  done
'
```

### 磁盘空间

| 文件系统 | 总容量 | 用途 |
|---------|--------|------|
| `/data` (vgpaas-share) | 1.2TB | DROID 数据集 + checkpoints |
| `/checkpoints` (sda2) | 294GB | 预训练模型（Wan2.1-14B） |

## 7. 已知问题与取舍

### 续训 Loss 曲线断裂

原因：Optimizer 状态在 resume 时被跳过（避免 FSDP OOM）。每次 resume 损失 ~100-200 步训练效率。

### Global step 计数重置

HF Trainer 的 global_step 在 resume 后从 0 重新计数，导致 checkpoint 命名与实际步数不一致。需手动追踪实际步数。

### 评测样本不跨批次一致

不同评测批次（尽管 seed 相同）因数据集加载路径差异，使用不完全相同的固定样本。

## 8. 报告生成

报告生成脚本：`/tmp/gen_report.py`（在本地 `/Users/kevin/code/dreamzero/docs/` 也有副本）

```bash
docker exec dreamzero_train python3 /tmp/gen_report.py
```

输出：`/workspace/dreamzero/final_report.html`（7.2MB，自包含 HTML）

在线查看：`https://versatile-ai.github.io/embodied-ai/dreamzero/final_report.html`

## 9. 快捷命令速查

```bash
# 查看训练状态
docker exec dreamzero_train bash -c 'echo "S=$(grep -oE "[0-9]+/100000" /tmp/train_v10.log|tail -1) E=$(grep -ciE Traceback /tmp/train_v10.log) P=$(ps aux|grep -c [e]xperiment.py)"'

# 查看 NPU
ssh ... root@192.168.106.239 "npu-smi info | grep -E 'NPU|AICore|Memory'"

# 查看最新 checkpoint
docker exec dreamzero_train bash -c 'ls -d /data/droid/checkpoints/dreamzero_droid_npu_16gpu_v3/checkpoint-* | sort -V | tail -5'

# 停止训练
docker exec dreamzero_train bash -c 'pkill -9 -f experiment.py; pkill -9 -f torchrun'

# 查看训练日志
docker exec dreamzero_train tail -50 /tmp/train_v10.log
```

## 10. 文件路径速查

| 内容 | 容器内路径 |
|------|-----------|
| 项目代码 | `/workspace/dreamzero/` |
| 训练脚本 | `/workspace/dreamzero/scripts/train/droid_16gpu.sh` |
| 评测脚本 | `/workspace/dreamzero/scripts/eval/batch_eval.py` |
| DROID 数据 | `/data/droid/` |
| 预训练模型 (Wan2.1-14B) | `/checkpoints/Wan2.1-I2V-14B-480P/` |
| Checkpoint v2 | `/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v2/` |
| Checkpoint v3 | `/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v3/` |
| 评测结果 (旧) | `/workspace/dreamzero/eval/`, `eval_1000/` |
| 评测结果 (v2) | `/workspace/dreamzero/eval_new/` |
| 评测结果 (v3) | `/workspace/dreamzero/eval_new2/` |
| 增强对比图 | `/workspace/dreamzero/eval_enhanced/` |
| 报告生成脚本 | `/tmp/gen_report.py` |
