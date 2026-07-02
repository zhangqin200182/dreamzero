# DreamZero NPU 训练设计文档 — 第二部分：FSDP + LoRA 训练

## 概述

本文档记录了在 **华为昇腾 910 NPU** 上实现 DreamZero（Wan2.1-I2V-14B，~140 亿参数）**8 卡 FSDP + LoRA 训练** 的全部修改、遇到的问题及解决方案。

**服务器**：113.46.41.54，16× Ascend910（每卡 61.27 GiB HBM），320 CPU 核心，2TB 内存，aarch64  
**容器**：`dreamzero_train` — torch 2.7.1 + torch_npu 2.7.1 + CANN 9.0.0 + transformers 4.57.1 + accelerate 1.14.0

---

## 1. 架构设计：14B 模型在 NPU 上的 FSDP + LoRA 方案

### 1.1 为什么选 FSDP 而非 DeepSpeed

Wan2.1-I2V-14B 模型约 140 亿参数（bf16 下约 28 GiB）。每张昇腾 910 NPU 有 61.27 GiB HBM。不做分片的话，模型本身就占用单卡近一半显存，优化器状态、梯度和激活值无法容纳。

**FSDP**（Fully Sharded Data Parallel）将模型参数、梯度和优化器状态分片到所有 NPU 上：
- 使用 `full_shard` + `auto_wrap`，40 个 transformer block 各自成为独立的 FSDP 单元
- 每张 NPU 上模型参数从 ~28 GiB 降至 ~3.5 GiB（28/8）
- 结合 LoRA（仅 ~0.1% 参数可训练），优化器状态开销极小

选择 FSDP 而非 DeepSpeed 的原因：
1. PyTorch 原生集成 — 无第三方依赖与 `torch_npu` 冲突
2. HuggingFace Trainer 内置 FSDP 支持（`training_args.fsdp`）
3. `torch_npu` 通过 `hccl` 后端提供 FSDP 支持

### 1.2 模型架构

```
WANPolicyHead（动作头，~14B 参数）
├── model: PeftModel → LoraModel → CausalWanModel（骨干网络）
│   ├── blocks: ModuleList[40 × CausalWanAttentionBlock]  ← FSDP 包裹每个 block
│   │   ├── CausalWanSelfAttention（LoRA 注入 q, k, v, o）
│   │   ├── WanI2VCrossAttention
│   │   └── ffn（LoRA 注入 ffn.0, ffn.2）
│   ├── state_encoder: CategorySpecificMLP（可训练）
│   ├── action_encoder: MultiEmbodimentActionEncoder（可训练）
│   └── action_decoder: CategorySpecificMLP（可训练）
├── text_encoder: WanTextEncoder（冻结，从 nn.Module 树中注销）
├── image_encoder: WanImageEncoder（冻结，注销）
└── vae: WanVideoVAE（冻结，注销）
```

### 1.3 关键设计决策

1. **FSDP 包裹 `CausalWanAttentionBlock`**（不是 `DiTBlock`）— causal 变体使用不同的 block 类
2. **冻结编码器从 nn.Module 树中注销** — 防止 FSDP 尝试分片/移动它们
3. **冻结编码器延迟设备放置** — 留在 CPU 上，首次前向传播时移至 NPU
4. **统一 bf16 数据类型** — 所有参数（包括 LoRA）在 FSDP 包裹前统一转为 bf16
5. **FSDP 激活时跳过设备放置** — FSDP 自行管理设备分配

---

## 2. 核心代码修改（FSDP 自带 vs 我们的适配）

FSDP 本身提供了完整的分片训练机制（参数分片、all-gather、reduce-scatter、优化器状态分片），但 DreamZero 的模型结构有 3 个特殊点，需要我们手动适配才能让 FSDP 正常工作。

### FSDP 自带的（不需要改代码）

| 能力 | 说明 |
|------|------|
| CPU 加载 → 按 block 分片 → 各取 1/8 → 移到各自 NPU | `full_shard` + `auto_wrap` 标准行为 |
| FlatParameter 创建 | 将每个 FSDP 单元的参数展平为一维张量 |
| 前向传播时 all-gather、反向传播时 reduce-scatter | 自动处理跨卡通信 |
| 参数/梯度/优化器状态分片 | FSDP 核心功能 |

### 我们的 3 处关键适配

#### 适配 1：跳过 Trainer 的整体设备搬移（防止打爆单卡）

**文件**：`groot/vla/experiment/base.py:375`

**问题**：HuggingFace Trainer 在 FSDP wrapping 之前会调用 `_move_model_to_device(model, device)` 把**整个模型**搬到一张卡上。对于 28 GiB 的模型，这会直接打爆单张 61 GiB 的 NPU（加上其他开销）。

**修改**：重写该方法，FSDP 模式下直接跳过，让 FSDP 自己管设备放置：
```python
def _move_model_to_device(self, model, device):
    if self.args.fsdp:
        return  # FSDP 自行处理设备放置，不需要先整体搬移
    super()._move_model_to_device(model, device)
```

#### 适配 2：冻结编码器从 Module 树注销（防止 FSDP 分片不可训练的大模块）

**文件**：`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py:_deregister_frozen_encoders`

**问题**：DreamZero 的动作头包含 3 个冻结编码器（T5 文本编码器 ~45 亿参数、CLIP 图像编码器、VAE），它们不参与训练但仍是 `nn.Module` 子模块。FSDP 会尝试对所有 `nn.Module` 子模块做分片和设备搬移，导致显存溢出。

**修改**：将冻结编码器从 `nn.Module` 树中移除，保留为普通 Python 属性：
```python
def _deregister_frozen_encoders(self):
    _te = self.text_encoder
    _ie = self.image_encoder
    _va = self.vae
    del self._modules['text_encoder']
    del self._modules['image_encoder']
    del self._modules['vae']
    # 保留为普通属性，前向传播时仍可访问，但 FSDP 看不到
    object.__setattr__(self, 'text_encoder', _te)
    object.__setattr__(self, 'image_encoder', _ie)
    object.__setattr__(self, 'vae', _va)
```

编码器在首次前向传播时通过 `_ensure_text_encoder_on_device()` 延迟搬到 NPU。

#### 适配 3：统一 bf16 数据类型（满足 FSDP FlatParameter 要求）

**文件**：`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py:set_trainable_parameters`

**问题**：FSDP 的 `FlatParameter` 要求同一单元内所有参数数据类型一致。但 PEFT/LoRA 默认以 fp32 创建 LoRA 参数，而骨干模型是 bf16，导致混合精度冲突。

**修改**：在 LoRA 注入之后、FSDP wrapping 之前，强制统一所有参数为 bf16：
```python
# set_trainable_parameters() 末尾：
self.to(dtype=torch.bfloat16)        # 统一数据类型
self._deregister_frozen_encoders()    # 注销冻结编码器
```

### 执行顺序

```
from_pretrained() → 加载 14B 权重到 CPU
    ↓
set_trainable_parameters()
    ├── add_lora_to_model()           ← 注入 LoRA（PEFT 创建 fp32 参数）
    ├── self.to(dtype=torch.bfloat16) ← 【适配 3】统一为 bf16
    └── _deregister_frozen_encoders() ← 【适配 2】隐藏 T5/CLIP/VAE
    ↓
Trainer.__init__()
    └── _move_model_to_device() → return  ← 【适配 1】跳过整体搬移
    ↓
accelerator.prepare(model)
    └── FSDP wrapping（框架自带）
         ├── auto_wrap：每个 CausalWanAttentionBlock → FSDP 单元
         └── 分片 → 每张 NPU 只保留 1/8 参数
```

---

## 3. 遇到的问题与解决方案

### 问题 1：FSDP 混合数据类型 ValueError

**报错**：
```
ValueError: Must flatten tensors with uniform dtype but got torch.bfloat16 and torch.float32
```

**根因**：FSDP 的 `FlatParameter` 要求同一 FSDP 单元内所有张量数据类型一致。模型存在混合数据类型：
- `add_lora_to_model()` 将骨干 DiT 转为 bf16
- 但 PEFT 的 `get_peft_model()` 默认以 **fp32** 创建 LoRA 参数（lora_A, lora_B）
- `self.model` 之外的动作编码器/解码器模块也保持 fp32

**解决方案**（`wan_flow_matching_action_tf.py`，`set_trainable_parameters` 方法）：
```python
# 在 LoRA 注入之后、FSDP 包裹之前：
self.to(dtype=torch.bfloat16)  # 将所有参数统一为 bf16
self._deregister_frozen_encoders()
```
这一行确保 FSDP 可见的所有参数具有统一的 bf16 数据类型。

---

### 问题 2：FSDP 创建 FlatParameter 时 OOM

**报错**：
```
RuntimeError: NPU out of memory. Tried to allocate 30.74 GiB (NPU 0; 61.27 GiB total capacity)
```

**根因**：训练命令使用 `training_args.fsdp="full_shard"`，只设置了分片策略但**没有**启用自动包裹。没有 `auto_wrap` 时，**整个模型**（~30 GiB）被视为一个 FSDP 单元。FSDP 创建 `FlatParameter` 时调用 `torch.cat(flat_tensors)` 分配连续缓冲区 — 在已有模型参数基础上再需要 ~30 GiB。

**HF TrainingArguments 如何处理 FSDP 选项**：`fsdp` 字段是**空格分隔的字符串**。`full_shard` 设置分片策略，`auto_wrap` 单独启用自动包裹策略。两者缺一不可：
```python
# transformers/training_args.py line 2019:
if fsdp_option == FSDPOption.AUTO_WRAP:
    os.environ[FSDP_AUTO_WRAP_POLICY] = "TRANSFORMER_BASED_WRAP"
```

**解决方案**（`npu_train.sh`）：
```bash
# 修改前（失败）：
training_args.fsdp=full_shard

# 修改后（成功）：
"training_args.fsdp=full_shard auto_wrap"
```
引号是必要的，因为 Hydra 需要将空格分隔的字符串视为一个完整值。

---

### 问题 3：FSDP Transformer 层类名错误

**报错**：
```
ValueError: Could not find the transformer layer class DiTBlock in the model.
```

**根因**：FSDP 的 auto_wrap_policy 使用 `get_module_class_from_name(model, class_name)` 查找要包裹的 transformer 层类。训练命令指定了 `DiTBlock`（来自基类 `WanModel`），但实际模型使用 `CausalWanModel`，其 block 类型不同：

调试输出显示实际的模块树：
```python
Module tree class names: {'CausalWanAttentionBlock', 'CausalWanSelfAttention',
    'WanI2VCrossAttention', 'CausalWanModel', 'PeftModel', 'LoraModel', ...}
# 注意：树中没有 'DiTBlock'！
```

causal 变体（`CausalWanModel`，`wan_video_dit_action_casual_chunk.py:1254`）使用 `CausalWanAttentionBlock`（1096 行），而非基类的 `DiTBlock`（`wan_video_dit.py:323`）。

**解决方案**（`npu_train.sh`）：
```bash
# 修改前（类名错误）：
training_args.fsdp_transformer_layer_cls_to_wrap=DiTBlock

# 修改后（类名正确）：
training_args.fsdp_transformer_layer_cls_to_wrap=CausalWanAttentionBlock
```

---

### 问题 4：冻结编码器导致 FSDP OOM

**问题**：FSDP 尝试将所有 nn.Module 子模块移至 NPU，包括冻结的文本编码器（T5，~45 亿参数）、图像编码器（CLIP ViT-Huge）和 VAE — 导致显存溢出。

**根因**：FSDP 调用 `accelerator.prepare(model)` 时会移动整个模块树到目标设备。冻结编码器在 FSDP 初始化阶段不需要在 NPU 上。

**解决方案** — 将冻结编码器从 nn.Module 树中注销（`wan_flow_matching_action_tf.py`）：
```python
def _deregister_frozen_encoders(self):
    """将冻结编码器从 nn.Module 树中移除，使 FSDP 不会移动它们。"""
    _te = self.text_encoder
    _ie = self.image_encoder
    _va = self.vae
    del self._modules['text_encoder']
    del self._modules['image_encoder']
    del self._modules['vae']
    object.__setattr__(self, 'text_encoder', _te)
    object.__setattr__(self, 'image_encoder', _ie)
    object.__setattr__(self, 'vae', _va)
```

注销后：
- `self.text_encoder` 作为普通 Python 属性仍可正常使用
- 但对 `nn.Module.parameters()`、`named_modules()` 和 FSDP 不可见
- 编码器留在 CPU 上，首次前向传播时延迟移至 NPU

延迟设备放置：
```python
def _ensure_text_encoder_on_device(self, ref_tensor):
    if not getattr(self, '_text_enc_device_ready', False):
        self.text_encoder.to(device=ref_tensor.device, dtype=torch.bfloat16)
        self.text_encoder.eval()
        self._text_enc_device_ready = True
```

---

### 问题 5：NPU 不支持 TF32

**报错**：
```
ValueError: --tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7
```

**根因**：TF32（TensorFloat-32）是 NVIDIA Ampere+ GPU 的专有特性，昇腾 NPU 不支持。

**解决方案**（`npu_train.sh`）：
```bash
tf32=false  # 原来是：tf32=true
```

---

### 问题 6：Hydra 配置 Schema 缺少键

**报错**：
```
omegaconf.errors.ConfigAttributeError: Key 'fsdp_cpu_ram_efficient_loading' is not in struct
```

**根因**：Hydra 配置 schema（`conf.yaml`）只定义了 `fsdp`、`fsdp_config` 和 `fsdp_transformer_layer_cls_to_wrap`。训练命令尝试设置不在 schema 中的 `fsdp_cpu_ram_efficient_loading` 和 `fsdp_sync_module_states`。

**解决方案**：从训练命令中移除这两个键。对初始训练设置非必要。

---

### 问题 7：FSDP 设备放置冲突

**根因**：HuggingFace Trainer 的 `_move_model_to_device` 在 FSDP 包裹之前被调用，会尝试将整个模型移到单个设备上。使用 FSDP 时，设备管理应由 FSDP 框架处理。

**解决方案**（`base.py`，`BaseTrainer`）：
```python
def _move_model_to_device(self, model, device):
    if self.args.fsdp:
        return  # FSDP 自行处理设备放置
    super()._move_model_to_device(model, device)
```

---

### 问题 8：数据集 LFS 指针未解析

**报错**：
```
pyarrow.lib.ArrowInvalid: Could not open Parquet input source '<Buffer>':
Parquet magic bytes not found in footer.
```

**根因**：`/data/droid` 下的 DROID 数据集是从 HuggingFace 克隆的，但未解析 Git LFS 指针。全部 57,774 个 parquet 文件和 173,322 个视频文件都是 130 字节的 LFS 指针文件，而非实际数据。

**解决方案**：
- 由于服务器无法直接访问 huggingface.co（中国大陆网络），切换为使用中国镜像 `hf-mirror.com`
- 编写了 64 线程并行下载脚本 `scripts/train/fast_download.py`，利用 100 Mbps 带宽并发下载
- 下载目标目录：`/data/droid_mirror`

---

## 4. 代码修改汇总

### 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` | 添加 `self.to(dtype=torch.bfloat16)`、`_deregister_frozen_encoders()`、`_ensure_text_encoder_on_device()`、`_ensure_image_encoder_on_device()` |
| `groot/vla/experiment/base.py` | 添加 `_move_model_to_device` 重写，FSDP 激活时跳过 |
| `groot/vla/common/utils/device.py` | 新文件：NPU/CUDA/CPU 设备抽象层 |
| `groot/vla/common/utils/__init__.py` | 导出 device 模块 |
| `groot/vla/common/utils/misc/torch_utils.py` | 使用设备抽象 |
| `groot/vla/configs/conf.yaml` | 添加 FSDP 配置键 |
| `groot/vla/data/transform/video.py` | cuda → 设备抽象 |
| `groot/vla/experiment/experiment.py` | 使用设备抽象 |
| `groot/vla/experiment/utils.py` | 使用设备抽象的 synchronize |
| `groot/vla/model/dreamzero/modules/attention.py` | 更新 FlashAttention 检测 |
| `groot/vla/model/dreamzero/modules/wan2_1_attention.py` | 更新 FlashAttention 检测 |
| `groot/vla/model/dreamzero/modules/wan_video_dit.py` | 更新 FlashAttention 检测 |
| `groot/vla/model/dreamzero/modules/vram_management.py` | 更新显存查询 |
| `groot/vla/model/dreamzero/modules/wan_video_vae.py` | 更新设备引用 |
| `groot/vla/model/dreamzero/modules/flow_unipc_multistep_scheduler.py` | 更新设备引用 |
| `groot/vla/model/n1_5/sim_policy.py` | 更新缓存清理 |
| `eval_utils/serve_dreamzero_wan22.py` | 更新分布式初始化 |
| `scripts/train/npu_train.sh` | NPU 训练启动脚本 |

### 新增文件

| 文件 | 用途 |
|------|------|
| `groot/vla/common/utils/device.py` | 设备抽象层（~250 行），自动检测 NPU/CUDA/CPU |
| `scripts/train/npu_train.sh` | 8 卡 NPU 训练启动脚本 |
| `scripts/train/fast_download.py` | 64 线程并行数据集下载工具 |

---

## 5. 训练命令

```bash
export DREAMZERO_DEVICE=npu
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=https://hf-mirror.com

torchrun --nproc_per_node 8 --standalone \
    groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/droid_relative \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    training_args.warmup_ratio=0.05 \
    per_device_train_batch_size=1 \
    max_steps=10 \
    bf16=true \
    tf32=false \
    dataloader_num_workers=1 \
    save_strategy=no \
    "training_args.fsdp=full_shard auto_wrap" \
    training_args.fsdp_transformer_layer_cls_to_wrap=CausalWanAttentionBlock \
    droid_data_root=/data/droid_mirror \
    dit_version=/checkpoints/Wan2.1-I2V-14B-480P \
    text_encoder_pretrained_path=/checkpoints/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=/checkpoints/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=/checkpoints/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth \
    tokenizer_path=/checkpoints/umt5-xxl
```

---

## 6. 验证结果

### 已通过 ✅
1. **配置解析**：Hydra + TrainingArguments 在 8 个 rank 上成功创建
2. **设备检测**：`DEVICE: npu | npu:0`，`dist: hccl`，`FA: False`
3. **模型加载**：14B 模型从 7 个 safetensor 分片在 8 个 rank 上加载成功
4. **LoRA 注入**：LoRA 成功添加到 q, k, v, o, ffn.0, ffn.2 目标层
5. **FSDP 类解析**：`CausalWanAttentionBlock` 在模型树中找到，auto_wrap_policy 设置完成
6. **编码器注销**：冻结编码器（T5, CLIP, VAE）从模块树中注销成功
7. **数据类型统一**：所有参数统一为 bf16，满足 FSDP 要求
8. **数据集初始化**：`Initialized dataset droid with EmbodimentTag.OXE_DROID` 在所有 rank 上成功
9. **训练循环启动**：Trainer.train() 调用成功，数据迭代器已启动

### 阻塞中 ⏳
10. **数据加载**：被损坏的数据集阻塞（LFS 指针）。正在通过 hf-mirror.com 下载实际数据。

---

## 7. 待解决问题

### 6.1 数据集下载（关键阻塞项）
- `/data/droid` 中所有 231,096 个文件均为 Git LFS 指针（每个 130 字节）
- 已切换至中国镜像 `hf-mirror.com`，正在通过 64 线程并行下载
- 下载至 `/data/droid_mirror`
- **后续操作**：下载完成后重新启动训练

### 6.2 前向传播验证
- 训练基础设施已验证到数据加载环节
- LoRA + FSDP 在 NPU 上的前向传播尚未测试
- 潜在问题：NPU 算子对因果注意力模式的兼容性
- **后续操作**：数据集就绪后验证

### 6.3 显存优化
- 当前配置 `per_device_train_batch_size=1`，`max_chunk_size=4`
- 梯度检查点尚未启用（可节省 ~40% 激活值显存）
- 如果长序列训练出现 OOM，可能需要 `training_args.gradient_checkpointing=true`
- **后续操作**：前向传播出现 OOM 时启用

### 6.4 弃用警告
- `fsdp_transformer_layer_cls_to_wrap` 在 transformers 4.57.1 中已标记为弃用
- 应迁移至 `fsdp_config` 字典格式：
  ```yaml
  training_args.fsdp_config:
    transformer_layer_cls_to_wrap: CausalWanAttentionBlock
  ```
- **后续操作**：优先级低，当前方式可用

---

## 8. 调用流程图

```
torchrun（8 个进程）
  └─ experiment.py:main()
       └─ VLAExperiment.__init__(cfg)
            ├─ BaseTrainer.__init__()
            │   ├─ TrainingArguments(fsdp="full_shard auto_wrap", ...)
            │   │   └─ 设置环境变量：FSDP_SHARDING_STRATEGY, FSDP_AUTO_WRAP_POLICY
            │   └─ _move_model_to_device() → 跳过（FSDP 激活）
            │
            ├─ WANPolicyHead.__init__()
            │   ├─ CausalWanModel.from_pretrained()  [加载 14B 参数]
            │   ├─ set_trainable_parameters()
            │   │   ├─ add_lora_to_model() → PeftModel 包裹
            │   │   ├─ self.to(dtype=torch.bfloat16)  ← 统一数据类型
            │   │   └─ _deregister_frozen_encoders()  ← 隐藏 T5/CLIP/VAE
            │   └─ _ensure_*_on_device() [前向传播时延迟调用]
            │
            └─ accelerator.prepare(model)
                 └─ FSDP 包裹
                      ├─ auto_wrap：每个 CausalWanAttentionBlock → FSDP 单元
                      ├─ FlatParameter 创建（每个单元每张 NPU ~3.5 GiB）
                      └─ 参数分片到 8 张 NPU
```

---

## 9. 经验总结

1. **HF FSDP 选项是空格分隔的**：`"full_shard auto_wrap"` 而不是 `"full_shard,auto_wrap"`
2. **模块类名必须精确匹配**：`get_module_class_from_name()` 搜索 `named_modules()` — 必须用实际类名（如 `CausalWanAttentionBlock`），不能用基类名
3. **FSDP 要求统一数据类型**：同一 FSDP 单元内所有参数必须数据类型相同。PEFT 默认以 fp32 创建 LoRA 参数 — 需要显式转换
4. **冻结模块需从 nn.Module 树中注销**：使用 `del self._modules[key]` + `object.__setattr__` 保留为普通 Python 属性，对 FSDP 不可见
5. **TF32 是 NVIDIA 专有特性**：在 NPU 上必须禁用
6. **Git LFS 数据集需要显式解析**：`git clone` 只下载指针文件；需要 `git lfs pull` 或 `huggingface-cli download` 获取实际数据
7. **中国大陆服务器需使用镜像**：huggingface.co 不可达，需切换至 `hf-mirror.com` 并设置 `HF_ENDPOINT` 环境变量
