# DreamZero LoRA 训练评测方案

## 触发时机

每个 checkpoint（每 200 步）自动触发一次评测。checkpoint 位置：

```
/checkpoints/dreamzero_droid_npu_16gpu_v2/checkpoint-{N}/
  adapter_model.safetensors    ← LoRA 权重 (~152 KB)
```

## 评测内容

### 1. Action 预测精度 A/B 对比

| 模式 | 说明 | 推理方法 |
|------|------|----------|
| **Full** | video + action 联合去噪，输出 action | `model.lazy_joint_video_action_causal(inputs)` |
| **Action-Only** | 跳过视频去噪，只输出 action | 同上，但 `video_final_noise=1.0` |

对比指标：MSE（预测 vs GT）、推理时间。

### 2. 可视化产出

每个 checkpoint 产出：

```
eval/step_{N}/
├── full/ep_00.png          ← Full 模式：GT（蓝）vs Pred（红）关节轨迹
├── action_only/ep_00.png   ← Action-Only 模式
├── comparison.png          ← 两种模式并列对比
└── summary.json            ← {"full_mse": 0.xxx, "ao_mse": 0.xxx, ...}
```

全局汇总：

```
eval/
├── trend_mse.png           ← 横轴=step，纵轴=MSE，full vs ao 两条线
└── all_summaries.jsonl     ← 每行一个 checkpoint 的汇总
```

### 3. 核心实验问题

**跳过视频去噪，动作预测精度降低多少？**

如果 AO 模式精度接近 Full，推理时可以省掉视频去噪，大幅加速。

---

## 当前代码路径

### 服务器（192.168.106.239）

```
容器: dreamzero_train (镜像: dreamzero_migrate, --network=host)

训练脚本:
  /workspace/dreamzero/scripts/train/droid_16gpu.sh

评测脚本 (开发中):
  /tmp/eval_ab.py                        ← 最新版，inference 可运行但 NPU 算子崩溃

Model/LoRA 路径:
  /checkpoints/dreamzero_droid_npu_16gpu_v2/checkpoint-{N}/adapter_model.safetensors

数据:
  /data/droid/  (挂载自 /mnt/paas/data/droid)

代码:
  /workspace/dreamzero/groot/
```

### 本地

```
评测脚本:
  /Users/kevin/code/dreamzero/scripts/eval/visual_eval.py   ← 初始版
  /tmp/eval_ab.py                                             ← 最新调试版
  /tmp/eval_v2.py                                             ← 中间版本

训练脚本:
  /tmp/droid_16gpu.sh                                         ← 正确的 fsdp 引号版本

文档:
  /Users/kevin/code/dreamzero/docs/EVAL_PLAN.md               ← 本文件
  /Users/kevin/code/dreamzero/docs/NPU_OOM_REPORT.md          ← OOM 分析报告
```

---

## 评测推进状态

**已验证通过：**
- [x] 模型 + LoRA 权重加载（hydra instantiate + safetensors）
- [x] 数据集管线（DreamTransform 变换后样本获取）
- [x] 输入格式构建（tokenize text、batch dim、pad dims、norm action）
- [x] `model.forward()` 正常返回 `{loss, dynamics_loss, action_loss}`

**已修复 (2026-07-08):**
- [x] NPU 崩溃 `Do has to be positive, but got 0` — 通过 `_reset_model_state()` 在每样本前重置推理状态解决
- [x] `torch.compile` triton 驱动缺失 — 通过 `torch._dynamo.config.disable = True` 全局禁用 compile（eval 单进程模式无 triton 驱动）
- [x] RoPE assertion `action_register_length == num_action_per_block + num_state_per_block` 失败 — state 来自数据集有 4 tokens，但推理期望 1 token；截断 state 到最近 token
- [x] bfloat16 `.numpy()` 转换失败 — `.cpu().float().numpy()` 先转 float32
- [x] **在 239 容器内运行评测确认修复生效** — Full 和 Action-Only 模式均成功
  - Full MSE=36.54, 推理耗时=10.2s
  - Action-Only MSE=62.55, 推理耗时=9.4s

**所有修复均仅在 eval 脚本内，未修改模型代码。**

---

## 环境依赖

评测需要：

```
Python: torch_npu, safetensors, omegaconf, hydra, numpy, matplotlib, decord, pandas
系统: libhccl.so, libascend_hal.so (Ascend CANN 9.0.0)
网络: 无外网需求（模型和数据均在本地）
NPU: 16 × Ascend 910, 需要至少 1 个空闲 NPU（推理占用 ~31 GB）
```

---

## 测试命令

```bash
# 在 239 容器内运行评测
cd /workspace/dreamzero
python3 /tmp/eval_ab.py \
    --checkpoint /checkpoints/dreamzero_droid_npu_16gpu_v2/checkpoint-200 \
    --output_dir eval/step_200 \
    --num_samples 3
```
