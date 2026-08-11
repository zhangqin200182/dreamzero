# Attention 层对齐范式：代码级 Review

> 基于对 DreamZero、FastWAM、π₀ (openpi)、COSMOS 3 四个代码库的逐层分析
> 2026-08-11

---

## 摘要

本文是对 `RESEARCH_DIRECTION.md` 提出的"Attention 层对齐作为独立跨模态对齐范式"的代码级 review。通过分析四个架构的实际实现，本文：
- 验证了文档对四个架构的归类（全部正确）
- 发现 Expert 分离程度构成一个**连续谱**，而非二元分类——这对实验结论的泛化有直接影响
- 指出两个核心主张的证据缺口：强耦合在 Expert 分离架构中未经验证
- 提出了路线图的优先修正和关键实验
- 对 DreamZero 实验 B 的反直觉结果给出了基于代码的解释

---

## 一、四个架构的 Joint Self-Attention 机制确认

### 1.1 机制高度一致

四个架构在 Attention 层进行跨模态对齐的方式完全相同：

```
每个 Layer:
  Expert A ──→ Q_a, K_a, V_a (各自权重) ──┐
                                            ├── cat([Q, K, V]) → Joint Flash Attention → split → 各自 o_proj → 各自 FFN
  Expert B ──→ Q_b, K_b, V_b (各自权重) ──┘
```

对齐发生在 Q·K^T 内积（一个无参数操作），Attention 结束后各 Expert 回到独立表示空间。

### 1.2 实现差异汇总

| | DreamZero | FastWAM | π₀ | COSMOS 3 |
|---|---|---|---|---|
| **代码位置** | `wan_video_dit_action_casual_chunk.py` | `mot.py:447-556` | `gemma.py:157-249` | `transformer_cosmos3.py:49-128` |
| **Q/K/V 权重** | 完全共享 | 完全独立（`_name()` 命名隔离） | 完全独立（`_name()` 命名隔离） | 完全独立（`add_q_proj` 等独立投影） |
| **FFN 权重** | 完全共享 | 完全独立 | 完全独立 | 完全独立（含 gen-only MoE） |
| **交叉层数** | 全部 40 层 | 全部 30 层 | 全部 18 层 | 全部 36 层 |
| **Q/K/V head dim** | 相同（共享权重） | 相同（24×128=3072，强制对齐） | 相同（8×256，width 不同但 head_dim 对齐） | 相同 |
| **各 Expert width** | 相同（5120） | Video 3072, Action 1024 | LLM 2048, Action 1024 | 不同（具体值 per checkpoint） |
| **Position Encoding** | 共享 RoPE | 各自 RoPE（Video 3D, Action 1D） | 共享 RoPE（统一 position id cumsum） | 统一 3D mRoPE + temporal modality margin |
| **Attention Mask** | 全 causal（chunk-wise） | action→video 仅第一帧（uncond） | block-causal（prefix 双向，suffix→all） | text=causal, gen=full（`two_way_attention`） |
| **推理 KV Cache** | 无分离 | `prefill_video_cache` + `forward_action_with_video_cache` | prefix LLM K/V 缓存，10 步只用 Action Expert | Reasoner K/V 缓存，gen_only 复用 |

### 1.3 四个架构独立收敛的证据强度

FastWAM（清华/上海 AI Lab, 2026.3）、COSMOS 3（NVIDIA, 2026.5）、DreamZero（NVIDIA GEAR Lab, 2025）、π₀（Physical Intelligence, 2025）——四个独立团队、四种不同主干（Wan2.1、Wan2.2、Gemma2B+PaliGemma、Qwen3-VL+Nemotron），不约而同走向了同一个机制。文档的核心观察成立。

---

## 二、关键发现：Expert 分离程度构成连续谱

这在 RESEARCH_DIRECTION.md 中被忽略了。四个架构并非同质的"Expert 分离"，而是落在一条连续谱上：

```
全共享 ←──────────────────────────────────────────────→ 全分离（不同 attention mode）

DreamZero            FastWAM                π₀                    COSMOS 3
  │                     │                    │                       │
  Q/K/V/FFN 全共享      Q/K/V/FFN 全独立     Q/K/V/FFN 全独立       Q/K/V/FFN 全独立
  仅 token 类型区分      但 Q/K/V 维度强制     但 width 不同           und/gen 不同 attention
                         对齐 (3072=3072)     仅 head_dim=256 对齐    mode (causal vs full)
                         各自独立 RoPE         各自 width: 2048/1024   gen 有独立 MoE router
```

**这对论证策略的影响**：

- DreamZero 是全共享架构。你的所有因果实验（A-E）在 DreamZero 上进行，它处于这条谱的**最左端**（耦合最紧）。
- 文档试图将 DreamZero 的结论推广到整条谱，但**谱的两端行为可能完全不同**。
- 具体来说：在 DreamZero 中验证的"训练强耦合"来自**共享参数**（Q/K/V/FFN 完全相同），而非 Attention 机制的 Q·K^T 内积本身。

---

## 三、"训练强耦合"的梯度路径分析（核心 gap）

### 3.1 不同架构中 video loss → action 的梯度路径

**DreamZero（全共享权重）**：
```
video_loss → ∂/∂(shared_QKV_weight) ← action_loss
```
梯度通过**完全相同的参数**传播。这是"共享参数耦合"，不是"Attention 耦合"。实验 D/E 验证的是这一条路径。

**FastWAM（独立 Q/K/V/FFN）**：
```
video_loss → ∂L/∂(video_attn_output) → [通过 softmax 中的 action_K 贡献] → ∂L/∂(action_K_weight)
```
梯度仅通过 attention score 矩阵中 video_Q · action_K^T 这一项传播。这是一条**间接且可能很弱**的梯度路径。

**π₀（独立 Q/K/V/FFN + 不同 width）**：
```
LLM_loss → ∂L/∂(LLM_attn_output) → [通过 softmax 中的 action_K 贡献] → ∂L/∂(action_K_weight)
```
耦合更弱——两边的表示空间维度不同（2048 vs 1024）。

**COSMOS 3（独立路径 + 不同 attention mode）**：
```
gen_loss → ∂L/∂(gen_attn_output) → [gen→und cross-attention 中使用 und_K] → ∂L/∂(und_K_proj)
```
在 `three_way_attention` 中显式拆分为 gen→gen + gen→und。耦合仅存在于 gen→und 的 cross-attention 路径中。

### 3.2 结论

"训练强耦合是 Attention 对齐的通用属性"这个主张，**在 Expert 分离架构中未经验证**。在 DreamZero 上观察到的强耦合是共享参数的产物，不是 Attention 机制的固有属性。

**这是整个研究方向最高优先级的待验证假设。**

---

## 四、"推理弱耦合"在 Expert 分离架构中的表现

### 4.1 FastWAM：主动弱耦合

FastWAM 的推理弱耦合比 DreamZero 更激进——它**根本不生成未来视频**：

- `infer_action`（部署模式）：只编码当前帧，跳过 video denoising
- `prefill_video_cache`：一次性编码第一帧 → 缓存 30 层 video K/V
- `forward_action_with_video_cache`：每 denoising step 只 forward action expert，读缓存 video K/V
- `infer_joint`（仅用于 PSNR 评测）：才同时生成 video + action

这证明在 Expert 分离架构中，弱耦合不仅是成立的，而且是**主动设计选择**——分离 inference 是他们的核心加速策略。

### 4.2 π₀：多频率 KV Cache 分离

`pi0.py:233-278` 的 `sample_actions` 展示了完全相同的策略：
- Phase 1：一次 LLM forward（2B），缓存 18 层 prefix K/V
- Phase 2：10 步 Euler denoising，每步只 forward Action Expert（300M），读缓存 LLM K/V

没有输入缓存 → 退出 → 下次再进来的策略，动作推理精度无损。

### 4.3 COSMOS 3：Reasoner K/V 复用

`_make_inference_text_kv_cache` 在所有 denoising step 共享 text K/V cache。gen tower 的 `gen_only=True` 模式跳过 und computation。

### 4.4 小结

**推理弱耦合是四个架构共同验证的属性，这个结论是稳固的。** 而且 Expert 分离架构中的弱耦合比 DreamZero 更强（主动跳过不需要的模态生成），而非更弱。

---

## 五、实验 B 反直觉结果的代码级解释

### 5.1 现象回顾

在 DreamZero 实验中，将 video latent 替换为 random noise 或 zero 后，action MSE 下降了 13%（从 21.61 降至 18.69）。替换全黑/全白第一帧后，action 仅退化 12%。

### 5.2 基于代码的分析

DreamZero 使用**独立的 per-token noise schedule**：
- Video: `Beta(3, 1)` 分布采样 sigma（偏向低噪声）
- Action: `Uniform` 分布采样 sigma

在 joint denoising 过程中，video 的 noisy latent 在 Shared Q/K/V attention 中作为 key/value 存在。由于 sigma 分布不同，video token 的噪声水平波动范围与 action token 不同。这导致：

1. Video 的 denoising 路径在 full model 中产生了一个**对 action 有害的表示偏移**（因为共享 Q/K/V 权重需要同时服务于两种不同的 denoising 动态）
2. 当你用 random/zero latent 替换 video 时，意外消除了这种干扰
3. 替换第一帧（CLIP 编码）无影响，说明 CLIP 特征本身对 action 的贡献方式不同于 denoising 过程中的 video latent

### 5.3 推论

如果这个解释成立，意味着：**在共享 DiT 架构中，video denoising 对 action 是 harmful（而非 neutral）的。** 这对你的理论框架有一个重要补充——Attention 对齐的推理分离不是可选的优化，而是**防止模态间有害干扰的必要设计**。

**建议实验**：在 FastWAM 或 π₀ 上做同样的实验（random/zero latent 替换），看 action 精度是否变化。如果 Expert 分离架构中不存在这种"有害干扰"，这将成为 Attention 对齐优于全共享架构的强证据。

---

## 六、对路线图的修正建议

### 6.1 实验优先级重新排序

| 优先级 | 实验 | 验证内容 | 风险 |
|--------|------|---------|------|
| **P0** | π₀ 梯度传播实验（平行 DreamZero D/E） | "训练强耦合"是否泛化到 Expert 分离架构 | 如果不存在，理论框架需要重大修正 |
| **P1** | π₀ prefix 扰动实验（平行 DreamZero A/C） | "推理弱耦合"是否泛化 | 低风险——FastWAM/π₀ 已有主动弱耦合证据 |
| **P2** | π₀ 上做 random latent 替换（平行 DreamZero B） | Expert 分离架构中是否存在"有害干扰" | 如果不存在有害干扰，这是范式 B 的重要优势 |
| **P3** | FastWAM 梯度传播实验 | 纯 DiT Expert 分离的中间验证点 | 中风险 |

### 6.2 Phase 3 的实际架构风险

从 π₀ (AR LLM + Diffusion Action) 加 Video DiT Expert 时，面临 **generation paradigm 冲突**：

- π₀ 的 LLM 是 autoregressive（causal attention）
- Video DiT 是 diffusion（full/bidirectional attention）
- 两者的 attention mask 需求不同

COSMOS 3 用 `two_way_attention` / `three_way_attention` 解决了这个问题（und=causal, gen=full）。如果不想复制这个复杂度：

**简化方案**：Video Expert 只在 prefix 阶段做一次编码（如 π₀ 中的 SigLIP），不参与 diffusion rollout。这本质上是 FastWAM 的 `prefill_video_cache` 策略——只编码当前帧，不生成未来视频。第一轮实验只需验证：加 Video Expert 的额外训练是否改善 action，而非验证视频生成质量。

### 6.3 论文定位策略

基于代码分析，建议将论文定位调整为两个层次：

**层次 1（已验证，可立即撰写）**：
- Attention 层对齐是一种独立范式（四个架构的代码级证据）
- 推理弱耦合是其通用属性（四个架构的推理策略一致验证）
- 多频率分离推理是自然推论（π₀ prefix cache, FastWAM video cache, COSMOS 3 text cache）

**层次 2（待验证）**：
- 训练强耦合的泛化边界（P0 实验决定）
- Expert 分离 vs 全共享的优劣对比（P2 实验决定）
- 三专家架构的统一（Phase 3）

---

## 七、补充发现

### 7.1 FastWAM 的 Action Expert 初始化策略

`scripts/preprocess_action_dit_backbone.py` 揭示了一个重要细节：Action Expert 的权重是从 Video DiT 权重通过线性插值初始化的（`alpha = sqrt(d_src/d_dst)`）。这意味着 **Action Expert 的初始表示空间是 Video Expert 表示空间的低维投影**。

这个初始化策略与"Attention 对齐"有微妙关系——它确保了训练初期两个 Expert 的 Q/K/V 投影落在相近的表示空间，使 Q·K^T 内积从一开始就有意义。如果 Expert 是随机初始化的，Attention 对齐可能需要更长的 warm-up 才能建立有效的跨模态连接。

### 7.2 π₀ 的 adaRMS 调制

`gemma.py:112-131` 中的 `RMSNorm` 支持自适应模式：timestep MLP 产生 scale/shift/gate，对 Action Expert 的每一层 RMSNorm 进行调制（`use_adarms=[False, True]`——LLM 为 False，Action 为 True）。这是一种**独立的 per-Expert 时间条件注入**，确保 Action Expert 的扩散动态不影响 LLM Expert 的表示。

这种设计进一步证明：Expert 分离不仅体现在 Attention 层，还体现在 Normalization 层——各 Expert 的时间条件完全独立。

### 7.3 COSMOS 3 的独立噪声调度

COSMOS 3 的每个模态（vision/action/sound）拥有**独立的 timestep 分布和 noise schedule**，与 DreamZero 的 per-token sigma 采样设计理念一致。这进一步支持了一个潜在论点：**Attention 对齐天然支持多频率/多噪声水平的独立调度**，而统一隐空间（JEPA）无法做到这一点。

---

## 八、结论

RESEARCH_DIRECTION.md 提出的核心主张——Attention 层对齐是独立于 JEPA 的跨模态对齐范式——在代码层面得到了确认。四个独立架构的一致性实现提供了很强的收敛性证据。

但从代码到论文，需要填补的最大 gap 是：**"训练强耦合"在 Expert 分离架构中的泛化验证。** 这决定了你的理论框架是"Attention 对齐同时提供推理分离和训练耦合优势"还是"Attention 对齐主要提供推理分离优势，训练优势仅在共享参数架构中成立"。

建议优先完成 P0 实验（π₀ 梯度传播），根据结果决定论文的最终定位。
