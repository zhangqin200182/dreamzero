# DreamZero-Flash: Decoupled Video & Action Denoising

代码位置：`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`

## 1. 为什么要 Decouple

标准 Flow Matching 推理中，视频和动作共享同一个去噪调度，从纯噪声 (σ=1.0) 逐步降到干净 (σ=0.0)。问题：**动作只需要语义信息，不需要视频完全清晰；但视频去噪消耗的计算和动作一样多。**

DreamZero-Flash 的 insight：**视频停在 80% 噪声就够了（VAE decoder 对轻微噪声不敏感），动作照常去到底（关节角度误差直接影响成功率）。** 减少视频去噪的计算需求。

## 2. 配置参数

```python
# 训练阶段
decouple_video_action_noise: bool = False   # video 用 Beta 偏高频，action 用独立 Uniform
video_noise_beta_alpha: float = 3.0         # Beta(3,1) 参数

# 推理阶段
decouple_inference_noise: bool = False      # video sigma 调度停在中间噪声
video_inference_final_noise: float = 0.8    # video 最终噪声水平

# Scheduler
shift = 5                                   # sigma 调度压缩参数
```

## 3. 训练：不是整个样本一个 σ，是每个 token 一个 σ

### 解耦的核心

代码在 `forward()` 方法，行 717-789。

所有 token 拼成一条序列送进同一个 DiT，Self-Attention 时互相对看。解耦靠**加噪时 σ 分布不同**：

```
Video: 每帧独立采样 Beta(3,1)
  f(σ) = 3σ² → 单调递增，65.7% 样本在 σ > 0.7
  均值 0.75，偏向高频噪声

Action: 每个 action token 独立采样 Uniform(0,1)
  f(σ) = 1 → 均匀覆盖所有 σ 水平
  包括低噪区 (σ < 0.3) 有 30% 样本，保证推理走到 σ=0 时有充分训练

State: 从 action timestep 中间隔采样，共享某个 action 的 σ
```

### 为什么不用同一种分布

- Action 要去到 σ=0，低噪区精确度决定最终关节角度 → 需要 Uniform 保证低噪区训练量
- Video 停在 σ=0.8，低噪区训练白费 → Beta(3,1) 把 budget 集中在高噪区
- 交叉组合 (video 高噪, action 低噪) 只在独立采样时出现 → 训练分布覆盖推理分布

### 真实数字（训练脚本参数）

```
输入分辨率和参数:
  视频: 176×320 像素, num_frames=33
  num_frame_per_block=2, num_action_per_block=24, num_state_per_block=1
  action_horizon=24, max_action_dim=32

维度链条:
  原始帧 176×320
    ↓ VAE Encoder: 3 次 spatial stride=2 × 2 次 temporal stride=2
  9 帧 latent, 每帧 22×40 = 880 个 latent 像素
    ↓ DiT patch_embed: Conv3d(kernel=(1,2,2), stride=(1,2,2))
  每帧 11×20 = 220 个 DiT token (dim=5120)

Token 序列长度:
  视频: 9 帧 × 220 tokens/帧 = 1980
  + 首帧条件帧 (latent 0, 带 mask=1 标记): 220
  总视频 = 2200
  + action: 24 tokens
  + state: 1 token
  总序列 ≈ 2225 tokens
```

### 加噪采样

```python
# line 719: 每帧采 1 个 σ, shape=[B, 9]
video_noise_ratio = Beta(3,1).sample([B, 9])

# line 736: 同一 block 的两帧共享 σ
timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]

# line 746-749: 每个 action token 独立采 σ, shape=[B, 24]
timestep_action_id = torch.randint(0, 1000, (B, 24))

# line 2082: 每帧的 1 个 σ 复制到该帧所有 220 个 patch token
timestep = timestep.unsqueeze(-1).expand(B, F, seq_len//F).reshape(B, -1)
```

同一帧内的 220 个 patch token σ 完全相同，保证恢复的画面连贯。只有不同帧之间 σ 才可能不同。action token 是真正每个独立 σ。

## 4. Per-Token Timestep Embedding：如何让同一个 DiT 处理不同 σ

代码在 `wan_video_dit_action_casual_chunk.py` 的 `_forward_train()` 行 2007-2096。

### 拼接 timestep

```python
# video timestep 扩展到每个 token
timestep = timestep.unsqueeze(-1).expand(B, F, seq_len//F).reshape(B, -1)  # [B, 2200]

# 拼接 video + action + state 的 timestep
timestep_state = timestep_action[:, ::stride]
timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)    # [B, 2225]
```

### 时间嵌入 → per-token AdaLN

```python
e = sinusoidal_embedding_1d(freq_dim, timestep.flatten())  # 不同 σ → 不同正嵌入
e0 = time_projection(e)                                     # → [B, L, 6, dim]
```

在每个 `WanAttentionBlock` (wan2_1_submodule.py 行 422-462)：

```python
e = (self.modulation + e).chunk(6, dim=1)  # 6 个调制参数

# Self-Attention
y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], ...)  # scale + shift
x = x + y * e[2]                                              # gate

# FFN
y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])              # scale + shift
x = x + y * e[5]                                              # gate
```

### AdaLN 参数作用

| 参数 | 阶段 | 作用 |
|------|------|------|
| `e[0]` | Self-Attn 前 shift | 偏置，控制进入 attention 时的均值 |
| `e[1]` | Self-Attn 前 scale | 缩放，控制特征差异被放大还是压扁 |
| `e[2]` | Self-Attn 后 gate | 门控，控制 attention 输出保留多少 |
| `e[3]` | FFN 前 shift | 偏置，控制激活函数的输入区间 |
| `e[4]` | FFN 前 scale | 缩放，控制特征差异 |
| `e[5]` | FFN 后 gate | 门控，控制 FFN 输出保留多少 |

LayerNorm 把所有 token 归一化到相同分布 → 抹平了 σ 差异 → AdaLN 通过 per-token 的 6 个数把 σ 信息重新注入回去。**每个 token 因为 σ 不同，被 AdaLN 施加不同的 scale/shift/gate，同一个 block 对高噪 token 和低噪 token 行为不同。**

## 5. Loss：分开算，共享梯度

代码在 forward() 行 814-840：

```python
# Video velocity loss (前 2200 个 token 的输出 → video head)
dynamics_loss = MSE(video_noise_pred, training_target_video)  # target = ε - z_0

# Action velocity loss (后 24 个 token 的输出 → action decoder)
action_loss = MSE(action_noise_pred, training_target_action)

loss = dynamics_loss + action_loss  # 共享 DiT 参数，梯度流经同一批 block
```

## 6. 推理：两个 Scheduler，Video Sigma Rescale

代码在 `lazy_joint_video_action()` 行 1010-1375。

### 创建两个独立的 scheduler

```python
# Line 1236-1247: 架构相同，都用 UniPC + shift=5
sample_scheduler = FlowUniPCMultistepScheduler(...)        # video
sample_scheduler_action = FlowUniPCMultistepScheduler(...)  # action

sample_scheduler.set_timesteps(16, shift=5)
sample_scheduler_action.set_timesteps(16, shift=5)
```

### Rescale video sigmas

```python
# Line 1252-1257: video sigma 调度从 [1.0→0.0] 映射到 [1.0→0.8]
video_final_noise = 0.8
sigma_max = sample_scheduler.sigmas[0].item()  # = 1.0
sample_scheduler.sigmas = sigmas * (1.0 - 0.8) / 1.0 + 0.8
# = sigmas * 0.2 + 0.8
# [1.0, 0.93, ..., 0.0] → [1.0, 0.986, ..., 0.8]
```

action sigmas 不变，保持 `[1.0, ..., 0.0]`。

### 16 步去噪循环

```python
for index, current_timestep in enumerate(timesteps):
    # 一次 DiT forward，所有 token 一起预测
    flow_pred, flow_pred_action = model(noisy_input, timestep=video_t,
                                         action=noisy_action, timestep_action=action_t)

    # Video: 用 rescaled scheduler 更新 (走到 0.8 停)
    noisy_input = sample_scheduler.step(flow_pred, timestep=video_t, sample=noisy_input)

    # Action: 用标准 scheduler 更新 (走到 0.0)
    noisy_input_action = sample_scheduler_action.step(flow_pred_action, timestep=action_t,
                                                       sample=noisy_input_action)
```

推理时每个 step 内部所有 token 共享同一个 σ（video 一轮一个值，action 一轮一个值），不像训练时每帧独立随机。

### Compute Skipping（可选）

代码在 `should_run_model()` 行 980-1008：

```python
# Dynamic 模式：连续两步预测的 cosine similarity
sim = cosine_similarity(flow_pred[t], flow_pred[t-1])
if sim > 0.95: skip 4 steps    # 几乎不变，跳过
elif sim > 0.93: skip 2 steps

# 跳过时直接复用上一步的 flow_pred
```

## 7. Scheduler 设计

### 训练：FlowMatchScheduler (flow_match_scheduler.py)

```python
scheduler = FlowMatchScheduler(
    shift=5,         # sigma 调度压缩
    sigma_min=0.0,   # 推理可到 0
    extra_one_step=True
)
```

Flow Matching 公式（video 和 action 相同）：
```
z_noisy = (1 - σ) · z_0 + σ · ε
target  = ε - z_0           ← 预测速度场 (velocity field)，不是噪声或原图
```

σ 从各自的分布采样（video: Beta(3,1), action: Uniform），除此之外公式完全一致。

训练没有 "步" 的概念——随机采一个 σ，一次 forward 预测速度场，算 loss。

### 推理：FlowUniPCMultistepScheduler (flow_unipc_multistep_scheduler.py)

sigma 调度变换（行 169）：

```python
sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
```

| shift | 效果 |
|-------|------|
| 1 | 等距 σ 序列，每步 Δσ ≈ 0.07 |
| 5 | 高噪区 Δσ 小（约 0.015），低噪区 Δσ 大（约 0.27） |

Flow Matching 在高噪区速度场方向多变，需要小步频繁修正；低噪区方向确定，可以大步跨越。shift=5 把步骤密集分配在高噪区，适合 video（全在高噪区操作）。action 低噪区虽步大，但此时预测已准，不影响精度。

### shift 和 rescale 的关系

两个独立变换，串行执行：

1. `set_timesteps` 先做 shift → 改变 σ 在步骤上的分配密度
2. 再对 video 做 rescale → 把 [1.0, 0.0] 压到 [1.0, 0.8]

rescale 只改值域，不改变 shift 造成的密度分布。video 只用上半段的密集区。

## 8. 完整数据流总结

```
训练 (一次 forward):
┌─────────────────────────────────────────────────────┐
│ 视频 (33 帧, 176×320)                                │
│   → VAE encode → 9 帧 latent (22×40)                │
│   → patch_embed → 每帧 220 tokens                   │
│                                                     │
│ 每帧采 σ ~ Beta(3,1) (偏高频)                       │
│   → 加到对应帧的 latent 上                          │
│                                                     │
│ 动作 (24 个, 每个 32 维)                             │
│   → ActionEncoder → 24 × 5120 tokens                │
│                                                     │
│ 每个 action 采 σ ~ Uniform (全覆盖)                 │
│   → 加到对应 action token 上                        │
│                                                     │
│ 拼接: [video_tokens; action_tokens; state_token]     │
│ Per-token timestep embedding → AdaLN                │
│ DiT forward → video head + action decoder           │
│ loss = MSE(v_pred, ε-z_0) + MSE(a_pred, ε-a_0)      │
└─────────────────────────────────────────────────────┘

推理 (16 次 forward):
┌─────────────────────────────────────────────────────┐
│ video sigma schedule:  [1.0 → 0.8] (rescale 后)     │
│ action sigma schedule: [1.0 → 0.0] (标准)            │
│                                                     │
│ for i in 0..15:                                     │
│   DiT forward (所有 token, per-token σ)             │
│   video:  z = z + (σ_video[i+1] - σ_video[i]) · v   │
│   action: a = a + (σ_action[i+1] - σ_action[i]) · v  │
│                                                     │
│ video 停在 0.8, action 走到 0                        │
└─────────────────────────────────────────────────────┘
```

## 9. 关键代码位置索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `wan_flow_matching_action_tf.py` | 111-126 | Decouple 配置定义 |
| 同上 | 180 | FlowMatchScheduler 初始化 (shift=5) |
| 同上 | 717-721 | Video σ ~ Beta(3,1) 采样 |
| 同上 | 744-750 | Action σ ~ Uniform 独立采样 |
| 同上 | 736 | Block 内帧共享 σ |
| 同上 | 780-791 | 分别加噪 + 计算 target |
| 同上 | 800-812 | DiT forward 调用 |
| 同上 | 814-840 | Loss 计算（分开算 + 合成） |
| 同上 | 1236-1247 | 推理两个 scheduler 创建 |
| 同上 | 1252-1259 | Video sigma rescale |
| 同上 | 1266-1342 | 16 步去噪循环 |
| 同上 | 980-1008 | Compute skipping |
| `flow_match_scheduler.py` | 7-92 | 训练 scheduler（加噪+target+weight） |
| `flow_unipc_multistep_scheduler.py` | 131-184 | 推理 scheduler（set_timesteps + shift） |
| `wan_video_dit_action_casual_chunk.py` | 66-99 | ActionEncoder |
| 同上 | 1382-1392 | Action/State encoder 初始化 |
| 同上 | 2007-2096 | `_forward_train` (per-token timestep 拼接) |
| 同上 | 1740-1833 | `_forward_blocks` (AdaLN 调制) |
| `wan2_1_submodule.py` | 382-462 | `WanAttentionBlock` (6 参数 AdaLN) |
| `dreamzero_cotrain.py` | 475-536 | 数据预处理 (action_horizon, padding) |
