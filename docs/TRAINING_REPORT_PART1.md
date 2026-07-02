# DreamZero NPU 训练技术报告（上篇）

> 服务器：华为昇腾 910 × 16 chips | CANN 9.0.0 | torch_npu 2.7.1  
> 训练方式：LoRA + FSDP，基座冻结  
> 报告日期：2026-06-30

---

## 一、模型架构与数据流动

### 1.1 整体数据流（含 Shape）

```mermaid
flowchart TD
    VIDEO["视频<br/>(B,3,33,176,320)"] --> VAE["VAE Encoder<br/>8× spatial↓, 4× temporal↓"]
    VAE --> LATENT["潜在 z_0<br/>(B,16,9,22,40)<br/>T'=9, H'=22, W'=40"]

    FIRST_FRAME["首帧<br/>(B,3,176,320)"] --> CLIP["CLIP ViT-H/14<br/>31/32 blocks"]
    CLIP --> CLIP_OUT["(B,257,1280)"]
    CLIP_OUT --> CLIP_PROJ["MLPProj(1280→5120)<br/>→ (B,257,5120)"]

    TEXT["T5 token ids<br/>(B,512)"] --> T5["T5-XXL<br/>24层, dim=4096, 64头"]
    T5 --> T5_OUT["(B,512,4096)"]
    T5_OUT --> T5_PROJ["Linear→GELU→Linear<br/>→ (B,512,5120)"]

    CLIP_PROJ --> CTX["cat → context<br/>(B,769,5120)"]
    T5_PROJ --> CTX

    LATENT --> TF["Teacher Forcing 拼接"]
    LATENT -->|"cat + mask(4)"| Y["首帧条件 y<br/>(B,20,9,22,40)"]
    LATENT -->|"Flow Matching 加噪"| NOISY["z_noisy<br/>(B,16,9,22,40)"]

    Y --> CLEAN_X["cat([z_0, y], dim=1)<br/>clean_x (B,36,9,22,40)"]
    Y --> NOISY_X["cat([z_noisy, y], dim=1)<br/>noisy_x (B,36,9,22,40)"]

    CLEAN_X --> PATCH_CLEAN["Patch Embedding<br/>Conv3d(36→5120, (1,2,2))"]
    NOISY_X --> PATCH_NOISY["Patch Embedding<br/>Conv3d(36→5120, (1,2,2))"]
    PATCH_CLEAN --> TOK_CLEAN["(B,1980,5120)"]
    PATCH_NOISY --> TOK_NOISY["(B,1980,5120)"]

    ACTION["动作 a_0<br/>(B,24,7)"] --> ACT_ENC["Action Encoder<br/>Linear×3 → (B,24,5120)"]
    STATE["状态<br/>(B,1,7)"] --> ST_ENC["State Encoder<br/>MLP → (B,1,5120)"]
    ACT_ENC --> REG["cat → register<br/>(B,25,5120)"]
    ST_ENC --> REG

    TOK_CLEAN --> DIT_IN["cat → 完整序列<br/>(B,3985,5120)"]
    TOK_NOISY --> DIT_IN
    REG --> DIT_IN
    CTX --> DIT["DiT ×40层<br/>SelfAttn + CrossAttn + FFN + AdaLN"]

    DIT_IN --> DIT
    DIT --> VIDEO_TOKENS["x[:, :1980]<br/>(B,1980,5120)"]
    DIT --> REG_TOKENS["x[:, 1980:2004]<br/>(B,24,5120)"]

    VIDEO_TOKENS --> HEAD["CausalHead<br/>Linear(5120→64)"]
    HEAD --> UNPATCH["unpatchify<br/>(B,1980,64)→(B,16,9,22,40)"]
    UNPATCH --> V_PRED["video_noise_pred<br/>(B,16,9,22,40)"]

    REG_TOKENS --> ACT_DEC["Action Decoder<br/>MLP(5120→64→7)"]
    ACT_DEC --> A_PRED["action_noise_pred<br/>(B,24,7)"]
```

  #### Flow Matching Loss: 变量来源与数据流

  ```mermaid
  flowchart TD
      subgraph 视频分支
          VAE["VAE.encode(video)"] --> z0_v["z_0 (latents)<br/>(B,16,9,22,40)"]
          RAND_V["torch.randn_like"] --> EPS_V["ε (noise)<br/>(B,16,9,22,40)"]
          z0_v --> FM_V["Flow Matching<br/>z_noisy = (1-σ)·z_0 + σ·ε<br/>target = ε - z_0"]
          EPS_V --> FM_V
          FM_V --> NOISY_V["noisy_latents<br/>(B,16,9,22,40)"]
          FM_V --> TGT_V["training_target<br/>(B,16,9,22,40)"]
          z0_v -->|"clean_x (Teacher Forcing 条件)"| DIT
          NOISY_V --> DIT["DiT Forward<br/>40层, 因果注意力"]
      end

      subgraph 动作分支
          DS_A["dataset actions"] --> A0["a_0 (actions)<br/>(B,24,7)"]
          RAND_A["torch.randn_like"] --> EPS_A["ε_a (noise_action)<br/>(B,24,7)"]
          A0 --> FM_A["Flow Matching<br/>a_noisy=(1-σ')·a_0+σ'·ε_a<br/>target=ε_a-a_0"]
          EPS_A --> FM_A
          FM_A --> NOISY_A["noisy_actions<br/>(B,24,7)"]
          FM_A --> TGT_A["training_target_action<br/>(B,24,7)"]
          NOISY_A --> DIT
      end

      DIT --> V_PRED["video_noise_pred<br/>(B,16,9,22,40)"]
      DIT --> A_PRED["action_noise_pred<br/>(B,24,7)"]

      V_PRED --> L_V["L_video = weighted_MSE(v_pred, target)"]
      TGT_V --> L_V
      A_PRED --> L_A["L_action = weighted_MSE(a_pred, target_a)<br/>× action_mask × has_real_action"]
      TGT_A --> L_A

      L_V --> L_TOTAL["L_total = L_video + L_action"]
      L_A --> L_TOTAL
  ```

  > **关键**: Flow Matching 预测的是"速度场" ε - z_0，不是噪声 ε 也不是原图 z_0。  
  > **训练目标**: 学习从任意加噪状态恢复到干净数据的最优方向。
```

### 1.2 DiT Block 内部 Shape 追踪

#### SelfAttention — Q/K/V + RoPE + 因果掩码

```mermaid
flowchart TD
    X["x: (B,3985,5120)"] --> Q["Q = RMSNorm → Linear → (B,3985,40,128)"]
    X --> K["K = RMSNorm → Linear → (B,3985,40,128)"]
    X --> V["V = Linear → (B,3985,40,128)"]

    Q --> SPLIT["Teacher Forcing 拆分"]
    SPLIT --> QC["q_clean: (B,1980,40,128)"]
    SPLIT --> QNI["q_noisy_img: (B,1980,40,128)"]
    SPLIT --> QNA["q_noisy_act: (B,24,40,128)"]
    SPLIT --> QNS["q_noisy_state: (B,1,40,128)"]

    QC --> ROPE3D["3D RoPE: 时间44+高度42+宽度42=128"]
    QNI --> ROPE3D
    QNA --> ROPE1DA["1D RoPE (freq=10240)"]
    QNS --> ROPE1DS["1D RoPE (freq=1024)"]

    ROPE3D --> ATTN["Causal FlashAttention → SDPA (NPU降级)"]
    ROPE1DA --> ATTN
    ROPE1DS --> ATTN
    ATTN --> OUT["flatten → (B,3985,5120) + Linear 投影"]
```

**因果掩码矩阵:**

| Q ↓ / K → | 干净图 | 加噪图 | 动作寄存器 | 状态寄存器 |
|------------|--------|--------|-----------|-----------|
| 干净图 | ✓ | ✗ | ✗ | ✗ |
| 加噪图 | ✓ (条件) | ✓ (因果) | ✓ (协同) | ✓ |
| 动作 | ✓ (条件) | ✓ (因果) | ✓ (因果) | ✓ |
| 状态 | ✗ | ✗ | ✗ | ✓ (仅自己) |

#### CrossAttention (I2V)

```mermaid
flowchart LR
    Q_TOKEN["Q: 视频 token<br/>(B,3985,40,128)"] --> XA1["CrossAttn₁"]
    CLIP_KV["CLIP KV<br/>(B,257,40,128)"] --> XA1
    XA1 --> OUT1["(B,3985,5120)"]

    Q_TOKEN --> XA2["CrossAttn₂"]
    T5_KV["T5 KV<br/>(B,512,40,128)"] --> XA2
    XA2 --> OUT2["(B,3985,5120)"]

    OUT1 --> SUM["+"] --> RESULT["(B,3985,5120)"]
    OUT2 --> SUM
```

#### FFN + AdaLN — 时间调制

```mermaid
flowchart LR
    T["σ: (B,9) timestep"] --> SIN["sin(2π·t·freq)"]
    SIN --> TIME_EMB["time_embedding<br/>(B,3985,5120)"]
    TIME_EMB --> TIME_PROJ["time_proj<br/>(B,3985,6,5120)"]
    TIME_PROJ --> MOD["6个调制向量:<br/>shift_sa, scale_sa, gate_sa<br/>shift_xa, scale_xa, gate_xa<br/>shift_ffn, scale_ffn, gate_ffn"]

    X_IN["x: (B,3985,5120)"] --> SA["SelfAttn + AdaLN"]
    MOD --> SA
    SA --> XA["CrossAttn + AdaLN"]
    MOD --> XA
    XA --> FFN_BLOCK["FFN(5120→13824→5120) + AdaLN"]
    MOD --> FFN_BLOCK
    FFN_BLOCK --> OUT["(B,3985,5120)"]
```

#### Head: Unpatchify 逆过程

```mermaid
flowchart LR
    TOKENS["video tokens<br/>x[:,:1980]<br/>(B,1980,5120)"] --> HEAD["CausalHead<br/>Linear(5120→64)"]
    HEAD --> FLAT["(B,1980,64)"]
    FLAT --> VIEW["view → (B,9,11,20, 1,2,2, 16)"]
    VIEW --> EINSUM["einsum: bfhwpqrc→bcfphqwr"]
    EINSUM --> RESHAPE["reshape → (B,16,9×1,11×2,20×2)"]
    RESHAPE --> OUTPUT["video_noise_pred<br/>(B,16,9,22,40)"]
```

### 1.3 推理时序列（无 Teacher Forcing）

```
无 Teacher Forcing 时仅一个序列:

  [首帧条件: 220 token] [图像块×4: 各440 token] [动作×4块: 各24] [状态×4块: 各1]

  seq_len = 220 + 4×440 = 1980
  register = 4×24 + 4×1 = 100
  total = 1980 + 100 = 2080

Blockwise Causal 注意力 (num_frame_per_block=2):
  image_blocks = 4,  action_blocks = 4,  state_blocks = 4
  每块 image:   2帧 × 220 = 440 token
  每块 action:  24 token
  每块 state:   1 token

  Tile i 的 image Q 可 attend:
    ✓ 首帧条件 (220)
    ✓ 自己块 image (440)
    ✓ 自己块 action (24) + state (1)
    不可 attend 未来块

  KV Cache:
    每层缓存: (2, B, cache_len, 40, 128)
    max_attention_size = 21 × 220 = 4620 token
    跨注意力缓存: (2, B, 512, 40, 128) [固定 T5 512 token]
```

### 1.4 配置一致性分析 ⚠️

训练配置 `frame_seqlen=880` 与 320×176 分辨率下的实际 `tokens_per_frame=220` **不一致**。

| 参数 | 配置值 | 实际值（320×176 + VAE 16ch） |
|------|--------|---------------------------|
| `frame_seqlen` | 880 | **220** |
| `action_horizon` | 24 | 需与 block 数匹配 |
| `num_frames` | 33 | 9 潜在帧（VAE 4×时间压缩） |

**原因：** 880 是 Wan2.1 在 480×256 分辨率下的 tokens_per_frame：
  - H_lat = 480/8 = 60, H_patch = 60/2 = 30
  - W_lat = 256/8 = 32, W_patch = 32/2 = 16
  - tokens = 30×16 = 480 → 但 880 ≠ 480 仍有差异

DreamZero 的训练脚本可能使用不同于我们配置的分辨率/帧数组合，或在运行时动态计算 `frame_seqlen`。**训练启动后会根据实际 patch embedding 输出验证此参数，若不一致需修正为 220。**

---

## 二、数据集

### 2.1 DROID 数据集

DreamZero-DROID 基于 [DROID 1.0.1](https://droid-dataset.github.io/)，经处理：

| 处理步骤 | 说明 |
|---------|------|
| 格式转换 | RLDS/TFDS → LeRobot v2.0 |
| 空闲帧移除 | Physical Intelligence idle frame detector |
| 语言过滤 | 去除无语言标注的 episode |
| 成功过滤 | 仅保留有非零奖励的 episode |
| 相机选择 | 3 视角：exterior_image_1_left, exterior_image_2_left, wrist_image_left |

**规模：** ~76,000 episodes，131GB，parquet + MP4 格式

### 2.2 数据格式

```
droid_lerobot/
├── data/chunk-000/
│   └── episode_XXXXXX.parquet   # 动作(24维) + 状态 + 语言标注
├── videos/chunk-000/
│   ├── observation.images.exterior_image_1_left/
│   ├── observation.images.exterior_image_2_left/
│   └── observation.images.wrist_image_left/
└── meta/
    ├── info.json, stats.json, modality.json
    ├── embodiment.json, episodes.jsonl, tasks.jsonl
    └── relative_stats_dreamzero.json
```

### 2.3 模态配置

| 模态 | 维度 | Delta Indices | 说明 |
|------|------|---------------|------|
| Video (×3) | 25 帧 | [0..24] | 3 相机，每段 25 帧 |
| State | 7 维 | [0] | joint_position(6) + gripper_position(1) |
| Action | 7 维 | [0..23] | 24 步动作块 |
| Language | 文本 | [0] | 任务指令（最多 3 种表述） |

### 2.4 相对动作

```
relative_action[t] = action[t] - state[anchor]
```

### 2.5 数据增强

```
Video:
  ToTensor:  uint8→float32[0,1]  (T,H,W,3)→(T,3,H,W)
  RandomCrop:  scale=0.95
  Resize:      480×256 (bilinear)
  ColorJitter: brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08
  Normalize:   mean=0.5, std=0.5

State/Action:
  q99 Normalization: clamp to [-1, 1]
```

---

## 三、训练方法

### 3.1 LoRA 微调

| 参数 | 值 |
|------|-----|
| LoRA rank | 4 |
| LoRA alpha | 4 |
| 目标模块 | `q`, `k`, `v`, `o`, `ffn.0`, `ffn.2` |
| 初始化 | Kaiming |
| 可训练参数 | ~83M（基座 14B 的 0.6%） |

**冻结：** DiT 基座（28GB）、T5（11GB）、CLIP（4.5GB）、VAE（0.5GB）  
**训练：** LoRA 适配器 + Action Encoder + State Encoder + Action Decoder

### 3.2 FSDP 配置

| 参数 | 值 |
|------|-----|
| 分片策略 | `full_shard` |
| 自动包装 | `transformer_layer_cls_to_wrap=DiTBlock` |
| CPU 高效加载 | true |
| 同步模块状态 | true |
| 通信后端 | HCCL（device.py 自动拦截 nccl→hccl） |

### 3.3 训练超参数

| 参数 | 值 |
|------|-----|
| 学习率 | 1e-4 |
| 预热比例 | 0.05 |
| 权重衰减 | 1e-5 |
| 每卡 batch size | 1 |
| 优化器 | AdamW（LayerNorm/bias 无衰减） |
| 精度 | bf16 |
| 梯度检查点 | 开启 |
| 视频帧数 | 33 |
| 动作 horizon | 24 |
| 最大步数 | 100,000 |

### 3.4 每卡显存估算（8 卡 FSDP + LoRA）

| 项目 | 占用 |
|------|------|
| DiT 参数分片（28GB÷8） | 3.5GB |
| All-gather 2 层预取（×2） | 1.8GB |
| LoRA 权重 (bf16, ~83M) | 0.2GB |
| LoRA 梯度 (fp32) | 0.4GB |
| LoRA 优化器 (fp32×3) | 1.2GB |
| 激活值（梯度检查点, 33帧） | ~5GB |
| CLIP/T5/VAE（编码后 CPU 卸载） | 0 |
| **总计** | **~12GB** |
| NPU HBM 总容量 | **64GB** |
| 利用率 | **19%** |

---

## 四、训练算法

### 4.1 Flow Matching

**原理：** 学习数据分布与噪声之间的速度场（velocity field）

```
训练:
  σ ~ U(0, 1)                         # 连续噪声级别
  z_noisy = (1-σ)·z_0 + σ·ε           # 线性插值
  target = ε - z_0                     # 速度场目标
  Loss = MSE(v_θ(z_noisy, σ), target)  # 预测速度

推理 (16 步 Euler):
  z_T ~ N(0,I)                         # 纯噪声开始
  for σ ∈ [1.0→0.0]:
    v = DiT(z_i, σ)                    # 预测速度
    z_{i+1} = z_i + v·Δσ               # Euler 前进
```

**调度器参数：** num_train_timesteps=1000, num_inference_steps=16, sigma_shift=5.0

### 4.2 噪声调度

| 模式 | 视频 σ | 动作 σ |
|------|--------|--------|
| 标准 | Uniform(0,1000) | 同步视频 |
| High Noise | Beta(3.0, 1.0) → 偏 σ>0.75 | 同步 |
| Decoupled | Beta(3.0, 1.0) 高噪声 | Uniform(0,1000) |

- `Beta(3.0, 1.0)`：均值 σ=0.75，模型更关注去除强噪声
- `cfg_scale=5.0`：推理时 CFG 引导强度

### 4.3 总损失（变量来源追踪）

```mermaid
flowchart TD
    subgraph 视频分支
        VAE["VAE.encode(video)"] --> Z0_V["z_0<br/>(B,16,9,22,40)"]
        RANDV["torch.randn_like"] --> EPS_V["ε<br/>(B,16,9,22,40)"]
        Z0_V --> FMV["Flow Matching<br/>线747-748"]
        EPS_V --> FMV
        FMV --> Z_NOISY["z_noisy = (1-σ)·z_0 + σ·ε<br/>(B,16,9,22,40)"]
        FMV --> TGT_V["target = ε - z_0<br/>(B,16,9,22,40)"]
        Z0_V -->|"clean_x (Teacher Forcing)"| DIT
        Z_NOISY --> DIT["DiT 40层"]
        DIT --> V_PRED["v_pred: Head→unpatchify<br/>(B,16,9,22,40)"]
        V_PRED --> L_V["L_video = weighted_MSE(v_pred, target)<br/>线787-792"]
        TGT_V --> L_V
    end

    subgraph 动作分支
        DS["dataset"] --> A0["a_0<br/>(B,24,7)"]
        RANDA["torch.randn_like"] --> EPS_A["ε_a<br/>(B,24,7)"]
        A0 --> FMA["Flow Matching<br/>线752-757"]
        EPS_A --> FMA
        FMA --> A_NOISY["a_noisy = (1-σ')·a_0 + σ'·ε_a<br/>(B,24,7)"]
        FMA --> TGT_A["target_a = ε_a - a_0<br/>(B,24,7)"]
        A_NOISY --> DIT
        DIT --> A_PRED["a_pred: Action Decoder<br/>(B,24,7)"]
        A_PRED --> L_A["L_action = weighted_MSE(a_pred, target_a)<br/>× mask × has_real_action<br/>线795-802"]
        TGT_A --> L_A
    end

    L_V --> L_TOTAL["L_total = L_video + L_action<br/>线803"]
    L_A --> L_TOTAL
```

| 变量 | 来源 | 代码行 |
|------|------|--------|
| z_0 | VAE.encode(video) | - |
| ε | torch.randn_like(z_0) | 676 |
| z_noisy | (1-σ)·z_0 + σ·ε | 747 |
| target_v | ε - z_0 | 748 |
| v_pred | DiT→Head→unpatchify | 766-771 |
| a_0 | dataset["action"] | - |
| ε_a | torch.randn_like(a_0) | 705 |
| a_noisy | (1-σ')·a_0 + σ'·ε_a | 752-756 |
| target_a | ε_a - a_0 | 757 |
| a_pred | DiT→Action Decoder | 766 |

> **σ 采样**: Beta(3,1) 偏高频 或 Uniform(0,1000) → scheduler.timesteps → sigma  
> **weight(σ)**: 高斯形加权, 中心 σ 权重最高

---

## 五、NPU 适配

| NVIDIA 算子 | 昇腾 NPU | 方式 |
|------------|---------|------|
| Flash Attention 2/3 | `F.scaled_dot_product_attention` | 内置降级 |
| cuDNN Attention (TE) | 跳过 | try/except |
| SageAttention | 跳过 | try/except |
| DeepSpeed ZeRO-2 | FSDP + HCCL | 框架替换 |
| `torch.cuda.*` | `torch.npu.*` | device.py 分发 |
| `dist.init_process_group("nccl")` | `→ "hccl"` | monkey-patch |

---

*下篇在训练完成后补充：loss 曲线、收敛速度、显存实测占用、推理延迟、闭环评估。*
