# DreamZero: Autoregressive Chunk-wise Architecture

代码位置：
- 主循环：`groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
- DiT 实现：`groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`
- Attention Block：`groot/vla/model/dreamzero/modules/wan2_1_submodule.py`

## 1. 核心概念

模型每次只生成 `num_frame_per_block` 帧（默认 2 帧），通过自回归方式逐块生成完整视频。

```
num_frames = 33
num_frame_per_block = 2
→ 需要 33/2 ≈ 16 次自回归调用（含首帧条件帧初始化）
```

每次 chunk 的生成是独立的 16 步去噪过程，chunk 之间通过 KV Cache 传递历史信息。

### 完整参数配置

```yaml
# vla.yaml / wan_flow_matching_action_tf.yaml
num_frames: 33
action_horizon: 24
num_frame_per_block: 2
num_action_per_block: 24
num_state_per_block: 1
frame_seqlen: 880
max_chunk_size: 4
local_attn_size: -1      # -1 表示全局 attention，或设为 N 限制窗口
sink_size: 0
```

### 序列结构

```
每个 chunk 的 token 排布:

┌──────────┬──────────────────────┬────────────┬───────────┐
│ 首帧条件  │  Video Block         │  Action    │  State    │
│ 220 tok  │  2帧×220=440 tok     │  24 tok    │  1 tok    │
│          │                      │            │           │
│ 仅条件帧  │  当前块(从噪声恢复)   │  当前块    │  当前块   │
└──────────┴──────────────────────┴────────────┴───────────┘

序列长度 = 220 + 440 + 24 + 1 = 685 tokens（单 chunk）
```

训练时所有 chunk 合并成一条长序列，推理时每次生成一个 chunk，通过 KV Cache 串起历史。

## 2. 训练：Teacher Forcing（一次前向）

代码：`_forward_train` 行 2007-2096。

### 原理

一次前向处理全部帧，但 **Causal Block-wise Attention** 确保每个 block 只能看到当前及之前的信息，模拟推理时的自回归条件。

```
训练时的注意力掩码:

           能看→
   ↓被看   │首帧│Blk0│Blk1│Blk2│...│Action│State│
  首帧     │ █  │    │    │    │   │      │     │   只能看自己
  Block 0  │ █  │ █  │    │    │   │  █   │  █  │   看首帧+当前block+当前action+当前state
  Block 1  │ █  │ █  │ █  │    │   │  █   │  █  │   看首帧+当前及之前的block
  Block 2  │ █  │ █  │ █  │ █  │   │  █   │  █  │
  Action   │    │ █  │ █  │ █  │   │  █   │  █  │   看之前的block+当前block+当前state
  State    │    │    │    │    │   │      │  █  │   只能看自己
```

### 关键约束

```python
# _forward_train 行 2063-2090
# 1. action encoder 把 action + state 编码后拼接
x = torch.cat([x, action_register], dim=1)  # video + action + state 合并序列

# 2. video timestep 扩展到每个 token
timestep = timestep.unsqueeze(-1).expand(B, F, seq_len//F).reshape(B, -1)

# 3. action timestep 也拼入
timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

# 4. per-token time embedding → AdaLN 调制
e = sinusoidal_embedding_1d(freq_dim, timestep.flatten())
e0 = time_projection(e)
e0 = e0.unflatten(dim=2, sizes=(6, self.dim))  # → [B, L, 6, dim]
```

### Block-wise Causal Attention 实现

代码：`_blockwise_causal_flash_attn` 行 316-482。

```python
def _blockwise_causal_flash_attn(self, q, k, v, frame_seqlen, ...):
    # 首帧: 只能看自己（条件帧）
    # Image Block i:
    #   看: 首帧 + 当前及之前所有 image blocks + 当前 action block + 当前 state block
    #   不看: 未来的 image blocks
    # Action Block i:
    #   看: 首帧 + 之前的 image blocks + 当前 image block + 当前 state block
    # State Block i:
    #   看: 只自己（条件 token）
    
    # local_attn_size 限制窗口:
    if self.local_attn_size != -1:
        image_kv_start = max(image_blocks_start, block_end - local_attn_size * frame_seqlen)
        # 只看最近 N 帧，旧帧 K/V 不可见
```

### 时间维度的 Block 约束

```python
# wan_flow_matching_action_tf.py 行 736
timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]
# 同一 block 的 2 帧共享同一个 timestep → 保证 block 内时序一致性
```

## 3. 推理：真正的自回归生成

代码：`lazy_joint_video_action` 行 1010-1375。

### 3.1 状态管理

```python
# 行 202: 全局状态
self.current_start_frame = 0     # 当前生成位置的帧索引
self.kv_cache1: KVCacheType      # KV Cache (cond + neg)
self.kv_cache_neg: KVCacheType
self.crossattn_cache             # Cross-Attention Cache
self.crossattn_cache_neg

# 行 509-532: KV Cache 结构
# 40 layers × 每 layer [K, V] × [2, B, 0, num_heads, head_dim]
# 初始 seq_len=0，随自回归逐步增长
kv_cache1 = [
    torch.zeros([2, B, 0, 40, 128]),  # Layer 0's K/V
    torch.zeros([2, B, 0, 40, 128]),  # Layer 1's K/V
    ...                                 # 40 layers
]
```

### 3.2 完整推理流程

```
┌── 输入 ──────────────────────────────────────────┐
│ videos: [B, 33, 3, 176, 320]                      │
│ language: "pick up the red block"                 │
└───────────────────────────────────────────────────┘
                     │
    ┌────────────────┴────────────────┐
    │ T5 文本编码 / CLIP 图像编码      │  一次，所有 chunk 共享
    │ VAE 编码首帧 → ys, clip_feas, image │
    └────────────────┬────────────────┘
                     │
current_start_frame = 0
    │
    ├─── Phase 1: 首帧初始化 ───────────────────────
    │     t=0 进入 KV Cache

    def _run_diffusion_steps(image, timestep=0, update_kv_cache=True)
    │   向每个 DiT Block 存入 220 个干净 K/V tokens
    │   current_start_frame += 1 → 1

    ├─── Phase 1.5: 非首 chunk 的历史帧缓存 ─────────
    ├───

    当 current_start_frame > 1:
      取上一 chunk 生成的干净 latent (2 帧)
     写入 KV Cache (t=0, update_kv_cache=True)
     一次性，无去噪

    ├─── Phase 2: Chunk 级别迭代去噪 ───────────────
    │
    │  noise = randn([B, 16, 2, 22, 40])  ← 仅 2 帧！
    │  noise_action = randn([B, 24, 32])
    │
    │  ┌─ 16 步去噪循环 ──────────────────────────┐
    │  │ for step 0..15:                          │
    │  │   predictions = model(                    │
    │  │     noisy_input,                          │
    │  │     action=noisy_input_action,             │
    │  │     kv_cache=... ,                        │
    │  │     current_start_frame=chunk 起始帧号,    │
    │  │     update_kv_cache=False  ← 全程不写！   │
    │  │   )                                       │
    │  │   video step(rescaled sigma schedule)     │
    │  │   action step(standard sigma schedule)    │
    │  └──────────────────────────────────────────┘
    │
    │  去噪完毕: noisy_input → clean_latent (2 帧)
    │  current_start_frame += 2

    ├─── 重复 Phase 2，直到生成全部 16 chunk ────

    └─── Phase 3: 拼接与解码 ───────────────────────
    拼接所有 chunk latent:
      clean_latents = [chunk_0, chunk_1, ..., chunk_15]
    完整 latent: [B, 16, 33, 22, 40]

    VAE Decode:
      latent_tokens → 时间 + 空间上采样
      [B, 16, 33, 22, 40] → [B, 3, 33, 176, 320]

    Action Decoder:
      action_noise_pred[24 tokens] → 24 个关节角度 [B, 24, 32]
```

### 3.3 去噪过程中 KV Cache 是只读的

```python
# 行 1306-1309: 16 步去噪循环内
kv_cache_metadata=dict(
    start_frame=self.current_start_frame,
    update_kv_cache=False,    # ← 不写 KV cache
)

# 去噪完成后才写入
# 行 1204-1227: 单独一次 t=0 forward
self._run_diffusion_steps(
    noisy_input=clean_result,
    timestep=0,               # t=0，不做预测
    kv_cache_metadata=dict(update_kv_cache=True)  # ← 此时写入
)
```

### 3.4 CFG 双 KV Cache

```python
# 行 1171-1176: cond 和 uncond 各有独立 KV Cache
kv_caches = self._get_caches([self.kv_cache1, self.kv_cache_neg])
crossattn_caches = self._get_caches([self.crossattn_cache, self.crossattn_cache_neg])

# _run_diffusion_steps 行 889-946:
# 对 cond 和 uncond 分别 forward，使用各自的 cache
for index, prompt_emb in enumerate(context):
    kv_cache = kv_caches[index]         # cond → kv_cache1, uncond → kv_cache_neg
    crossattn_cache = crossattn_caches[index]
    obs_noise_pred, action_noise_pred, updated_kv = model(
        noisy_input, ..., kv_cache=kv_cache, crossattn_cache=crossattn_cache, ...
    )
    predictions.append((obs_noise_pred, action_noise_pred))

# CFG 组合:
flow_pred = flow_pred_uncond + cfg_scale * (flow_pred_cond - flow_pred_uncond)
```

### 3.5 local_attn_size 滑动窗口

```python
# 行 1079-1081: 超出窗口自动重置
elif self.current_start_frame >= self.model.local_attn_size:
    self.current_start_frame = 0  # 重置！新序列开始

# wan_video_dit_action_casual_chunk.py:
# 每层 Self-Attention 只保留最近 local_attn_size 帧的 KV
image_kv_start = max(image_blocks_start, 
                     block_end - self.local_attn_size * frame_seqlen)
```

`local_attn_size=-1`（默认）时使用全局 attention——所有历史帧的 K/V 保留在 cache 中。显存随帧数线性增长。设限制后可支持无限长度生成。

## 4. Chunk-wise 带来的好处

### 4.1 显存

| | 全部帧 | Chunk-wise (2 帧) |
|---|---|---|
| 每步 activation | ~6 GB | ~360 MB |
| KV cache | — | ~5.6 GB (33 帧) |
| 总计 | > 显存容量 | ~6 GB 可控 |

KV cache 存的是压缩后的 K/V（head_dim=128），不是原始 activation（dim=5120），约 **25 倍压缩**。

### 4.2 训练推理 Attention 一致性

```
训练 (Teacher Forcing):
  Block_i 只能 attend Block_0..Block_i
  不能 attend Block_{i+1}..Block_n （因果遮罩）

推理 (Autoregressive):
  Chunk_i 只能 attend 首帧 + 已生成的 Chunk_0..Chunk_{i-1}

两者 Attention 语义完全一致 → 无 train-inference mismatch
```

### 4.3 无限长度生成

```
训练的 33 帧模型 → 推理出 100 帧视频

local_attn_size 滑动窗口:
  只保留最近 N 帧的 KV cache
  旧帧被逐步滚动出去
  显存不随总帧数线性增长
```

### 4.4 与 Decoupled Denoising 协同

```
每个 chunk 的 video 停在 σ=0.8:
  → 前序 chunk 有足够结构信息（不需完全清晰）
  → 后序 chunk 可以及早开始
  → 因果链传播效率更高

如果 video 要去到 σ=0.0:
  → 前序 chunk 必须跑满所有步
  → 后序 chunk 必须等前序完全去噪完成
  → 因果依赖更严格
```

### 4.5 潜在的流水线并行

当前实现是串行的（去噪完成再写 cache），但架构允许流水线化：

```
当前 (串行):
  Chunk 0: 16步去噪 → 写干净KV → Chunk 1: 16步去噪 → ...

可能 (流水线):
  Chunk 0, step 0: forward → 写 KV (σ=1.0 K/V)
  Chunk 0, step 1: forward → 更新 KV (σ=0.985 K/V)
  Chunk 1, step 0: forward → 读 Chunk 0 的 σ=0.985 KV
  ...
```

训练时独立采样的各种 σ 组合天然覆盖了这个场景（前序帧的各种噪声水平），但当前实现选择串行以保证工程简单性。

## 5. 为什么不并行化当前 Chunk 的去噪步骤

### 因果依赖链

```
Chunk_{i} 的条件:
  └─ 首帧 (永久 cache)
  └─ Chunk_{i-1} 的干净 latent K/V  ← 必须等待上一 chunk 去噪完成

KV cache 里写的是干净 K/V（t=0 写入），不是中间步骤的 K/V。
当前实现: 去噪过程 update_kv_cache=False → 中间 K/V 不持久化
去噪完成后: 单次 t=0 forward + update_kv_cache=True → 干净 K/V 写入
```

### 为什么不用中间 K/V

- 训练时 Teacher Forcing 用干净帧的 K/V 作为条件
- 训练-推理一致性优先
- 避免管理多噪声水平的并发 KV cache 的复杂度

## 6. 代码位置索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `wan_flow_matching_action_tf.py` | 174 | `num_frame_per_block` 初始化 |
| 同上 | 202 | `current_start_frame = 0` 状态 |
| 同上 | 509-532 | `_create_kv_caches` KV cache 结构 |
| 同上 | 534-546 | `_create_crossattn_caches` |
| 同上 | 889-946 | `_run_diffusion_steps` (CFG 双 pass + cache 管理) |
| 同上 | 948-978 | `_exchange_predictions` (多卡 CFG 交换) |
| 同上 | 980-1008 | `should_run_model` compute skipping |
| 同上 | 1010-1375 | `lazy_joint_video_action` 推理主循环 |
| 同上 | 1079-1081 | `local_attn_size` 溢出重置 |
| 同上 | 1102-1105 | 首帧 CLIP/VAE 编码 (仅一次) |
| 同上 | 1141-1142 | chunk 级噪声初始化 |
| 同上 | 1153-1200 | 首帧写入 KV cache |
| 同上 | 1204-1227 | 后续 chunk 写入 KV cache |
| 同上 | 1266-1342 | 16 步去噪循环 |
| `wan_video_dit_action_casual_chunk.py` | 66-99 | `MultiEmbodimentActionEncoder` |
| 同上 | 198-232 | `CausalWanSelfAttention` 初始化 |
| 同上 | 198-222 | `num_frame_per_block` / `local_attn_size` / `sink_size` |
| 同上 | 220 | `max_attention_size` 计算 |
| 同上 | 243-482 | `_visualize_attention_mask` (因果掩码可视化) |
| 同上 | 316-482 | `_blockwise_causal_flash_attn` (核心注意力实现) |
| 同上 | 1443-1570 | `_prepare_blockwise_causal_attn_mask` |
| 同上 | 1740-1833 | `_forward_blocks` (推理 KV cache 读写) |
| 同上 | 1917-1962 | `_forward_inference` (推理 DiT forward) |
| 同上 | 2007-2096 | `_forward_train` (Teacher Forcing) |
| 同上 | 2173-2181 | `forward` 分发 (有 cache→推理, 无→训练) |
| `wan2_1_submodule.py` | 382-462 | `WanAttentionBlock` (AdaLN 6 参数调制) |
