# DreamZero NPU 训练 FSDP OOM 问题分析报告

## 概述

DreamZero (Wan2.1-I2V-14B, ~16.5B 参数) 在华为昇腾 910 NPU 服务器 (8卡 FSDP + LoRA) 上训练时，step 0 正常完成，但 **step 1 backward 阶段必现 OOM**。

**环境**:
- 硬件: Ascend 910 × 8, 61.28 GiB HBM/卡
- 软件: PyTorch 2.7.1+cpu, torch_npu, Python 3.11.15
- 分布式策略: FSDP `full_shard auto_wrap`，每个 `CausalWanAttentionBlock` 独立包裹 (41 个 FSDP 模块)
- 训练配置: LoRA (~108.6M 可训练参数), 3-view (176×320) tiled → 352×640, num_frames=33, gradient checkpointing (`use_reentrant=True`)

---

## 1. 根本原因

### 1.1 这是 NPU 平台的 bug，不是应用代码问题

根因在于 `torch_npu` 对 PyTorch autograd 引擎的 C++ 层 `queue_callback` 机制实现不完整，导致 FSDP 的关键清理回调 `_post_backward_final_callback` 在 NPU 上不执行。这不是我们的应用代码改出来的问题。

**证据如下：**

#### 1.1.1 我们没有碰过 FSDP 内部代码

`queue_callback` 是 PyTorch autograd 引擎的 C++ 层机制：

```python
# torch/distributed/fsdp/_runtime_utils.py
Variable._execution_engine.queue_callback(_post_backward_final_callback)
```

我们的所有改动都在应用层：`device.py`（设备抽象）、`base.py`（training_step）、attention 模块（FlashAttention → SDPA 降级）。**没有任何一处触及 FSDP 的 hook 注册、reshard、或 callback 机制**。

#### 1.1.2 NPU 团队自己已知 FSDP 状态管理有问题

在容器内的 `/usr/local/.../fsdp/_runtime_utils.py` 中发现两处 NPU 官方补丁：

```python
# Line 726-728: 
# [NPU PATCH] Relax handle training state check
# — 放宽了 _post_backward_hook 中的训练状态断言

# Line 1251:
# [NPU PATCH] skip prefetch state assertion
# — 跳过 prefetch 状态检查
```

这说明 NPU 团队已经知道 FSDP 训练状态在 NPU 上不按预期流转，但**他们的修复方式只是放宽断言（跳过报错），没有解决根因（`_post_backward_final_callback` 不触发）**。

#### 1.1.3 `queue_callback` 失效是 torch_npu 执行引擎的问题

`Variable._execution_engine.queue_callback()` 依赖 PyTorch 的 C++ autograd 引擎在**整个 backward 图计算完成后**触发回调。在 CUDA 上，execution engine 能正确检测到所有 backward 操作完成并触发回调。`torch_npu` 对接的 NPU 执行引擎没有正确实现这个完成信号，导致回调永远不执行。**这是 C++ 层面的问题，不是 Python 层面能改出来的。**

#### 1.1.4 任何 FSDP 多步训练在 NPU 上都会踩到

这个 bug **不是特定于 DreamZero** 的。只要在 NPU 上用 FSDP `full_shard` 训练超过 1 个 step，理论上都会出现同样的问题：step 0 永远正常（首次注册不受影响），step 1 必崩。

### 1.2 FSDP 参数管理机制详解

FSDP (Fully Sharded Data Parallel) 把 14B 参数切成 8 份，每张 NPU 只持有约 1/8 (~2 GiB)。当某个 block 需要计算时，通过 all-gather 拉取完整参数，计算完成后 reshard 释放。

```
Forward block N:
  all-gather 完整参数 (0.77 GiB) → 计算 → reshard 只保留 1/8

Backward block N:
  all-gather 完整参数 (0.77 GiB) → 计算梯度 → reshard 释放完整参数
```

backward 阶段的 **reshard 是靠注册在 AccumulateGrad 节点上的 `_post_backward_hook` 触发的**。当梯度计算完成，hook 自动调用 `_post_backward_reshard()` 释放 all-gathered 的完整参数。

### 1.3 `_post_backward_final_callback` 的职责

backward 全部完成后，PyTorch 的 autograd 引擎通过 `queue_callback` 触发 `_post_backward_final_callback`，该回调负责：

1. **`_finalize_params()`** — 删除每个 flat_param 上的 `_post_backward_hook_state`（AccumulateGrad 钩子的引用）
2. **`_catch_all_reshard()`** — 清理任何未被 resharded 的参数
3. **`next_iter()`** — 推进 `_exec_order_data._iter` 计数器
4. **训练状态重置** — 将所有 FSDP 模块的 `training_state` 从 `FORWARD_BACKWARD` 重置为 `IDLE`

### 1.4 OOM 连锁反应

当 `_post_backward_final_callback` 不执行时：

```
Step 0 backward 完成
  → _post_backward_hook_state 残留（因为回调没跑，_finalize_params 未执行）

Step 1 forward 开始
  → _register_post_backward_hook 检查 hasattr(flat_param, "_post_backward_hook_state")
  → 返回 True（上一步残留的）
  → 跳过钩子注册

Step 1 backward 开始
  → 使用 step 0 留下的旧 AccumulateGrad 钩子
  → 但旧钩子绑定在 step 0 的计算图上（每个 step 创建新图）
  → 新计算图的 AccumulateGrad 节点上没有钩子
  → _post_backward_reshard() 永远不被调用
  → 每个 block 的 all-gathered 完整参数 (0.77 GiB) + 梯度 (0.77 GiB) 不释放
  → 1.54 GiB/block × 40 blocks = 61.6 GiB → 超出 61.28 GiB HBM → OOM
```

### 1.5 为什么 step 0 正常

Step 0 首次 forward 时，`_post_backward_hook_state` 不存在 → `_register_post_backward_hook` 正常注册新钩子 → 新钩子在 step 0 的 backward 中正常触发 → 参数被正确 reshard → 不泄漏。

### 1.6 泄漏的不是激活值

需要特别指出：**泄漏的不是激活值（activations）**。激活值由 gradient checkpointing 管理，forward 中不保存中间激活，backward 时重新计算，这部分工作正常。

泄漏的是 **FSDP 分片参数的 all-gather 副本**——每个 block 在 backward 中通过 all-gather 拉取完整参数用于梯度计算，计算完成后本应 reshard 释放，但因为 reshard 的 hook 没有触发，这些完整参数副本就一直留在 HBM 中。

---

## 2. 解决方案

### 2.1 修复层面

无法直接修复 `queue_callback`（这需要 torch_npu 团队在 C++ 层修复）。我们在 **应用层** (`BaseTrainer.training_step()`) 手动执行回调本该做的清理工作：

```
正常流程（CUDA）：
  forward → backward → autograd 引擎自动触发 _post_backward_final_callback
                                    ↓
                          清理 hook_state、重置状态、reshard

我们的修复（NPU）：
  training_step 开始前 → 手动清理 hook_state、重置 _iter
  forward → backward → 回调没触发（没关系，下一步开头会手动清理）
  training_step 结束后 → 手动重置 training_state 到 IDLE
```

### 2.2 具体修复代码

修改位置：`groot/vla/experiment/base.py` 的 `BaseTrainer.training_step()`

#### 修复 A：清理 `_post_backward_hook_state`（核心修复）

在 `super().training_step()` 之前执行，删除所有 FSDP 模块的 `_post_backward_hook_state`，迫使 FSDP 在下一次 forward 中重新注册新的 AccumulateGrad 钩子：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

for fsdp_module in FSDP.fsdp_modules(model):
    if hasattr(fsdp_module, '_handle') and fsdp_module._handle is not None:
        fp = fsdp_module._handle.flat_param
        if hasattr(fp, '_post_backward_hook_state'):
            fp._post_backward_hook_state[-1].remove()  # 移除旧钩子
            del fp._post_backward_hook_state
```

#### 修复 B：重置执行顺序

```python
# 重置 _iter 到 0
# 注意：is_first_iter 是只读 @property，底层由 _iter == 0 实现
# 必须直接设置 _iter，不能设 is_first_iter
fsdp_module._exec_order_data._iter = 0
fsdp_module._exec_order_data.handles_post_forward_order.clear()
```

#### 修复 C：backward 后重置训练状态

在 `super().training_step()` 之后执行。因为 `_post_backward_final_callback` 没有运行，training_state 停留在 `BACKWARD_POST` 而不是 `IDLE`，会导致后续的模型保存/参数收集断言失败：

```python
from torch.distributed.fsdp._common_utils import TrainingState, HandleTrainingState

for fsdp_state in FSDP.fsdp_modules(model):
    fsdp_state.training_state = TrainingState.IDLE
    if hasattr(fsdp_state, "_handle") and fsdp_state._handle is not None:
        fsdp_state._handle._training_state = HandleTrainingState.IDLE
        fsdp_state._handle._ran_pre_backward_hook = False
```

---

## 3. 排查历程

### 3.1 版本迭代与内存对比

| 版本 | Step 0 BW/block | Step 1 BW/block | 尝试的策略 | 失败原因 |
|------|:-:|:-:|------|------|
| v33 (基线) | +0.04 GiB | +1.54 GiB | 无修复 | OOM |
| v35 | +0.04 GiB | +1.54 GiB | 重置 `is_first_iter` | 只读属性，静默失败 |
| v36 | +0.04 GiB | +1.54 GiB | `synchronize()` | 不是异步问题 |
| v37 | +0.04 GiB | +1.54 GiB | `set_to_none=False` | 不是梯度累积问题 |
| v39 | +0.04 GiB | +1.54 GiB | hook 清理 + `is_first_iter` 重置 | 同一 try 块，前面异常后全部跳过 |
| **v40 (修复)** | **+0.04 GiB** | **-0.12 GiB** | **`_iter=0` + 分离 try 块 + hook_state 清理** | **成功** |

### 3.2 关键排查发现

#### 3.2.1 `is_first_iter` 只读属性 (v35-v39 静默失败)

`_ExecOrderData.is_first_iter` 是一个只读 `@property`：

```python
# torch/distributed/fsdp/_exec_order_utils.py
class _ExecOrderData:
    def __init__(self):
        self._iter = 0  # 实际存储

    @property
    def is_first_iter(self):
        return self._iter == 0  # 只读属性
```

v35-v39 都尝试直接赋值 `is_first_iter = True`，导致 `AttributeError: property 'is_first_iter' has no setter`。因为在 try/except 块中，异常被静默捕获，导致：
- v39 中 hook 清理代码与 `is_first_iter` 赋值在**同一个 try 块**中
- `is_first_iter` 赋值先执行并抛出异常
- **整个 try 块后续的 hook 清理代码被跳过**
- 从外部看起来像是"hook 清理代码不起作用"

**修复**：v40 使用 `_iter = 0`（直接设置底层属性）+ 每段修复使用独立 try 块。

#### 3.2.2 容器磁盘配额已满

Docker 容器 overlay 文件系统占满 50G 配额：
- `/home/work/models/Qwen3-8B`: ~16G
- `/usr` (预装包): ~19G

**影响**：无法使用 `docker cp`、无法在容器内写临时文件、甚至 `rm -rf` 都可能失败（overlay 需要写 whiteout 文件）。
**绕过**：所有文件操作通过 host-mounted `/workspace/dreamzero/` 路径，设置 `TMPDIR=/workspace/dreamzero/tmp`。

#### 3.2.3 模型保存阶段 OOM (SIGSEGV)

训练完成后，Trainer 的 state_dict 收集流程通过 `_unshard_fsdp_state_params` all-gather 完整参数并调用 `clone()`，在已占用 55+ GiB 的情况下触发 OOM (Signal 11)。

**解决方案**：保存前 `gc.collect()` + `empty_cache()`，或使用 `FullStateDictConfig(offload_to_cpu=True)` 将 state_dict 收集到 CPU。

---

## 4. 验证结果

| 指标 | 值 |
|------|-----|
| 训练步数 | 5/5 完成 |
| 训练时间 | 160.6s (~32s/step) |
| train_loss | 0.607 |
| action_loss | 0.494 |
| dynamics_loss | 0.243 |
| Step 0 BW 内存增长 | +0.04 GiB/block (正常) |
| Step 1 BW 内存增长 | -0.12 GiB/block (参数正确 reshard) |

Step 1 内存为负增长（-0.12 GiB/block）说明参数正在被正确 reshard 释放，符合 FSDP 的预期行为。

---

## 5. 文件修改清单

本次修改涉及两类改动：

### 5.1 FSDP OOM 修复（核心）

| 文件 | 修改内容 |
|------|---------|
| `groot/vla/experiment/base.py` | `training_step()` 添加 FSDP 清理逻辑 (修复 A+B+C) |

### 5.2 NPU 设备抽象层（训练前提）

| 文件 | 修改内容 |
|------|---------|
| `groot/vla/common/utils/device.py` | 新建，设备无关 API (NPU/CUDA 自动检测) |
| `groot/vla/common/utils/__init__.py` | 导出 device 模块 |
| `groot/vla/experiment/experiment.py` | autocast/Event/synchronize → 设备抽象 |
| `groot/vla/experiment/utils.py` | synchronize → 设备抽象 |
| `groot/vla/model/dreamzero/modules/attention.py` | FlashAttention → SDPA 降级 |
| `groot/vla/model/dreamzero/modules/wan2_1_attention.py` | 同上 |
| `groot/vla/model/dreamzero/modules/wan_video_dit.py` | 同上 + `use_reentrant=True` |
| `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` | 同上 |
| `groot/vla/model/dreamzero/modules/wan2_1_submodule.py` | NPU real-valued RoPE (不支持 complex/polar) |
| `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` | 设备抽象 + frozen encoder 处理 |
| `groot/vla/model/dreamzero/modules/wan_video_vae.py` | device='cuda' → DEVICE |
| `groot/vla/model/dreamzero/modules/flow_unipc_multistep_scheduler.py` | 同上 |
| `groot/vla/model/dreamzero/modules/vram_management.py` | mem_get_info → 设备抽象 |
| `groot/vla/common/utils/misc/torch_utils.py` | seed 设置 → 设备抽象 |
| 其他 4 个文件 | 各 1-7 处 cuda → 设备抽象 |

---

## 6. 后续建议

1. **向华为 torch_npu 团队反馈**：`Variable._execution_engine.queue_callback()` 在 NPU 上不触发 FSDP 的 `_post_backward_final_callback`，这是影响所有 FSDP 多步训练的平台级 bug
2. **模型保存修复**：在 checkpoint 保存前释放梯度和优化器状态，或使用 `offload_to_cpu=True`
3. **考虑 `use_reentrant=False`**：PyTorch 官方推荐用于 FSDP + gradient checkpointing，可进一步减少状态管理复杂度
