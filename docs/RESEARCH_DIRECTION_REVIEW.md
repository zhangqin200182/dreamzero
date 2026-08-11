# Attention 层对齐范式：代码级 Review（v2）

> 基于对 DreamZero、FastWAM、π₀ (openpi)、COSMOS 3 四个代码库的逐层分析
> 2026-08-11

---

## 摘要

本文是对 `RESEARCH_DIRECTION.md` 提出的"Attention 层对齐作为独立跨模态对齐范式"的代码级 review。通过分析四个架构的实际实现，本文：
- 验证了文档对四个架构的归类（全部正确，机制高度一致）
- 发现 Expert 分离程度构成一个**连续谱**——这对实验结论的泛化有直接影响
- 指出文档中一个关键的**逻辑不对称**：弱耦合从共享架构到分离架构的推断是成立的，但强耦合的同样推断不成立
- 指出文档的核心概念混淆：Q·K^T 执行的是"信息路由"而非"表示对齐"
- 指出 JEPA 作为对比对象的适配性问题
- 对实验 B 的反直觉结果给出代码级解释
- 提出路线图修正建议

---

## 一、四个架构的代码级确认

### 1.1 Joint Self-Attention 机制一致性（确认）

四个架构的跨模态交互机制完全一致：

```
每个 Layer:
  Expert A ──→ Q_a, K_a, V_a (各自权重) ──┐
                                           ├── cat([Q, K, V]) → Joint Flash Attention → split → 各自 o_proj → 各自 FFN
  Expert B ──→ Q_b, K_b, V_b (各自权重) ──┘
```

对齐发生在 Q·K^T 内积（无参数操作），Attention 结束后各 Expert 回到独立表示空间。**文档对此的归类是正确的。**

### 1.2 实现差异总表

| | DreamZero | FastWAM | π₀ | COSMOS 3 |
|---|---|---|---|---|
| **代码位置** | `wan_video_dit_action_casual_chunk.py` | `mot.py:447-556` | `gemma.py:157-249` | `transformer_cosmos3.py:49-128` |
| **Q/K/V 权重** | 完全共享 | 完全独立 | 完全独立 | 完全独立 |
| **FFN 权重** | 完全共享 | 完全独立 | 完全独立 | 完全独立（含 gen-only MoE） |
| **交叉层数** | 全部 40 层 | 全部 30 层 | 全部 18 层 | 全部 36 层 |
| **head_dim 对齐** | 天然相同 | 强制相同（24×128=3072） | 强制相同（8×256） | 强制相同 |
| **各 Expert width** | 相同（5120） | Video 3072, Action 1024 | LLM 2048, Action 1024 | 不同 |
| **Position Encoding** | 共享 RoPE | 各自 RoPE（Video 3D, Action 1D） | 共享 RoPE | 统一 3D mRoPE + temporal margin |
| **Attention Mask** | 全 causal | action→video 仅第一帧 | block-causal | text=causal, gen=full |
| **推理 KV Cache** | 无 | `prefill_video_cache` + `forward_action_with_video_cache` | prefix LLM K/V 缓存 + 10 步只用 Action Expert | Reasoner K/V 缓存，gen_only 复用 |

### 1.3 收敛性证据的强度

四个独立团队（GEAR Lab / 清华&上海AI Lab / NVIDIA / Physical Intelligence）、四种不同主干（Wan2.1 / Wan2.2 / Gemma2B+PaliGemma / Qwen3-VL）、三个不同时间点（2025 / 2026.3 / 2026.5），不约而同走向同一机制。**这不是巧合——文档的核心观察成立。**

---

## 二、Expert 分离程度是连续谱（文档未涉及）

### 2.1 连续谱模型

四个架构并非同质的"Expert 分离"，而是分布在一条连续谱上：

```
全共享 ←────────────────────────────────────────────────→ 全分离（不同 attention mode）

DreamZero            FastWAM                π₀                    COSMOS 3
  │                     │                    │                       │
  Q/K/V/FFN 全共享      Q/K/V/FFN 全独立     Q/K/V/FFN 全独立       Q/K/V/FFN 全独立
  仅 token 类型区分      但 dim 强制对齐        width 不同             und/gen 不同 attention
                         (3072=3072)          仅 head_dim=256 对齐    mode (causal vs full)
                         RoPE 各自独立         共享 RoPE              gen 有独立 MoE router
```

### 2.2 这对文档论证策略的影响

文档第 100 行给出的逻辑是：
> "选择 DreamZero 的原因：它是唯一没有 Expert 分离的全共享 DiT 架构...如果在此最'紧'的架构中仍观察到弱耦合，则 Expert 分离的架构中更应成立。"

这个逻辑对**弱耦合**成立：共享参数下输出质量都不传递，分离参数下更不可能传递。但文档隐含地将同样的逻辑用在了**强耦合**上——在第 151 行直接得出"Video loss 的梯度通过共享 Attention 权重传播到 action"的统一结论——这只是共享参数的产物，对 Expert 分离架构完全不适用。

**核心逻辑不对称**：

| | 弱耦合（推理） | 强耦合（训练） |
|---|---|---|
| DreamZero 观察到 | ✓（video 输出不影响 action） | ✓（video loss → action loss 下降） |
| 泛化到 Expert 分离的推断 | **成立**：共享参数下都不传递质量，分离更不传递 | **不成立**：共享参数的梯度路径 ≠ Attention 的梯度路径 |

---

## 三、梯度路径分析：强耦合能否泛化？（文档未区分）

### 3.1 四种架构中 video loss → action 的梯度路径

**DreamZero（全共享权重）**：
```
video_loss → ∂/∂(shared_QKV_weight) → 直接影响 action 的 forward
```
梯度通过**完全相同的参数**传播。实验 D/E 验证的是"共享参数耦合"，不是"Attention 耦合"。

**FastWAM（独立 Q/K/V/FFN）**：
```
video_loss → ∂L/∂(video_attn_output) → [仅通过 softmax 中 video_Q · action_K^T] → ∂L/∂(action_K_weight)
```
这是一条间接且可能极弱的梯度路径——梯度不经过 action expert 的参数，只通过 attention score 矩阵中的交叉项。

**π₀（独立 Q/K/V/FFN + 不同 width）**：
```
LLM_loss → ∂L/∂(LLM_attn_output) → [仅通过 softmax 中 LLM_Q · action_K^T] → ∂L/∂(action_K_weight)
```
耦合更弱——两边表示空间维度不同（2048 vs 1024），仅在 256-dim head 空间交互。

**COSMOS 3（独立路径 + 不同 attention mode）**：
```
gen_loss → ∂L/∂(gen_attn_output) → [仅 gen→und cross-attn 中的 und_K] → ∂L/∂(und_K_proj)
```
在 `three_way_attention` 中显式拆分，耦合仅存在于 gen→und 的 cross-attention 路径。

### 3.2 结论

**"训练强耦合是 Attention 对齐的通用属性"是当前框架中最大且最关键的待验证假设。** 在 DreamZero 上观察到的强耦合来自共享参数，该结论能否推广到 Expert 分离架构，完全取决于间接梯度路径的强度——这个强度目前是未知的。

**这是 P0 优先实验。**

---

## 四、概念精确性：两个需要修正的问题

### 4.1 "对齐" vs "路由"——核心概念的混淆

文档将 Q·K^T 描述为跨模态"对齐"机制。但从代码实现来看，Q·K^T 实际执行的是**信息路由**（information routing），而非表示对齐（representation alignment）：

- **对齐**意味着将两种表示拉到同一个空间、建立对应关系（如 CLIP 的对比学习使 image 和 text embedding 可互换）
- **路由**意味着根据相似度有选择地传递信息（如 softmax(QK^T) 使每个 token 从其他 token 聚合信息）

在 Attention 中，各 Expert 的表示在 Attention 前后保持在各自的独立空间中，Q·K^T 只是决定"从哪个 Expert 取多少信息"。称为"路由"比"对齐"更精确。

**建议**：文档的核心术语可以从"Attention 层对齐"改为"Attention 层跨模态路由"，或者至少在引言中区分这两个概念。这会让理论框架更精准，也避免与 JEPA 的"对齐"概念产生范畴混淆——JEPA 做的是表示对齐，Attention 做的是信息路由，它们在逻辑层次上是正交的，不是直接竞争的。

### 4.2 JEPA 作为对比对象的适配性

文档将 JEPA 设为对立范式。但 JEPA 的核心贡献是**同模态自监督学习**（从图像的一部分预测另一部分），跨模态扩展（multi-modal JEPA）在 JEPA 文献中并非主线。

更准确的对比对象应该是**共享隐空间方法**这个更宽泛的类别，包括：
- CLIP-style 对比学习（image-text shared space）
- 多模态 LLM 的 projection layer（将 vision/audio 投影到 LLM token space）
- JEPA 的 shared latent（同模态或跨模态）

将对比对象从"JEPA"扩大为"共享隐空间方法"，论证会更稳健，也不会陷入"JEPA 到底是不是做跨模态"的争议。

---

## 五、实验 B 反直觉结果：代码级解释

### 5.1 现象

Random/zero latent 替换 video latent 后，action MSE 反而**下降 13%**（21.61 → 18.69）。

### 5.2 基于代码的解释

DreamZero 使用**独立 per-token noise schedule**——Video: `Beta(3,1)`（偏低压噪声），Action: `Uniform`。两种噪声分布差异意味着共享的 Q/K/V 权重需要同时服务两种不同的 denoising 动态。Video 的 noisy latent 在 joint attention 中对 action token 产生了**有害的表示偏移**——共享权重被两种冲突的降噪目标拉扯。

Random/zero latent 消除了这种冲突，所以 action 精度反而提升。替换第一帧（CLIP 编码）几乎无影响，因为 CLIP 特征是冻结的，不参与 denoising 动态。

### 5.3 推论

在共享 DiT 架构中，video denoising 对 action 是 **harmful（而非 neutral）** 的。这意味着 Attention 层的 Expert 分离不仅是"可选的优化"，而是**防止模态间有害干扰的必要设计**。

**验证实验**：在 Expert 分离架构（FastWAM 或 π₀）中做同样的 random latent 替换。如果不存在 harmful 效应，这本身就是范式 B 相对于全共享架构的重要优势证据。

---

## 六、路线图修正建议

### 6.1 实验优先级重新排序

| 优先级 | 实验 | 验证问题 | 风险 |
|--------|------|---------|------|
| **P0** | π₀ 梯度传播实验（平行 DreamZero D/E） | 强耦合是否泛化到 Expert 分离架构？ | **高**——如果不存在，框架需重大修正 |
| **P1** | π₀ prefix 扰动实验（平行 DreamZero A/C） | 弱耦合是否泛化？ | 低——已有主动弱耦合代码证据 |
| **P2** | Expert 分离架构的 random latent 替换 | Expert 分离中是否存在 harmful 干扰？ | 低——预期无害 |
| **P3** | FastWAM 梯度传播实验 | 纯 DiT Expert 分离的中间验证 | 中 |

### 6.2 Phase 3 实际架构风险

从 π₀ (AR LLM + Diffusion Action) 加 Video DiT Expert 时，面临 generation paradigm 冲突：
- LLM：autoregressive，causal attention
- Video DiT：diffusion，bidirectional attention

COSMOS 3 用 `two_way_attention` / `three_way_attention` 解决。简化方案：Video Expert 只做 prefix 编码（类似 π₀ 中的 SigLIP / FastWAM 的 `prefill_video_cache`），不参与 diffusion rollout。

### 6.3 Phase 4 可实现性

COSMOS 3 代码中**不存在 GRPO 管线**——目前仅有 SFT post-training（`action/posttrain_config/*`）。从零构建 GRPO + forward dynamics rollout + reasoner-based reward 的工程量远超 8-12 周。建议 Phase 4 的范围明确为"在 COSMOS 3 上做 SFT-based 前向动态训练"，GRPO 作为 Phase 5+ 的扩展。

### 6.4 论文定位策略

**已验证（立即可写）**：
- Attention 层跨模态路由是一种独立于共享隐空间的范式（四个架构代码证据）
- 推理弱耦合是其通用属性（四个架构推理策略一致，FastWAM/π₀ 主动采用）
- 多频率分离推理是自然推论

**待验证（决定论文上限）**：
- 训练强耦合的泛化边界（P0 决定）
- Expert 分离 vs 全共享的优劣对比（P2 决定）
- 三专家架构可行性（Phase 3 决定）

---

## 七、补充代码发现

### 7.1 FastWAM 的 Action Expert 初始化

`scripts/preprocess_action_dit_backbone.py`：Action Expert 权重从 Video DiT 通过线性插值初始化（`alpha = sqrt(d_src/d_dst)`）。初始表示空间是 Video Expert 空间的低维投影，确保训练初期 Q·K^T 内积有意义。如果 Expert 随机初始化，Attention 路由可能需要更长 warm-up。

### 7.2 π₀ 的 adaRMS：独立时间条件

`gemma.py:112-131`：Action Expert 每层 RMSNorm 接受 timestep MLP 调制（`use_adarms=[False, True]`，LLM=False, Action=True）。扩散时间条件仅影响 Action Expert 的表示，不影响 LLM。**Expert 分离不仅体现在 Attention 层，也体现在 Normalization 层。**

### 7.3 独立噪声调度是通用设计

COSMOS 3 每个模态拥有独立 timestep 分布和 noise schedule，与 DreamZero 的 per-token sigma 采样一致。Attention 路由天然支持多频率/多噪声水平的独立调度——这是共享隐空间方法做不到的。

---

## 八、综合评估

### 文档的优势

1. **核心观察敏锐且正确**：四个独立架构的代码级一致性提供了非常强的收敛性证据
2. **理论框架简洁**：弱耦合 + 强耦合两个属性抓住了 Attention 路由的本质
3. **路线图有层次**：从验证到原型到应用的递进清晰

### 需要修正或注意的问题

1. **强耦合泛化未验证（关键）**：DreamZero 的强耦合来自共享参数，不是来自 Attention 机制。P0 实验优先
2. **"对齐" vs "路由"概念混淆**：建议使用"Attention 层跨模态路由"以更精确描述机制
3. **JEPA 对比适配性**：建议扩大为"共享隐空间方法"类别
4. **Phase 4 时间估计偏低**：COSMOS 3 无 GRPO 管线，从零构建工程量大于 8-12 周
5. **Expert 分离程度未被建模**：文档隐含地将四个架构视为同质，但它们落在连续谱上
6. **实验 B 的 harmful 效应未被重视**：这可能是最锋利的论据——证明 Expert 分离是必要设计而非可选优化

### 一句话总结

**方向正确，核心观察可靠，但"强耦合泛化"是当前理论框架的阿喀琉斯之踵——它决定了论文的上限。** 优先在 π₀ 上验证，根据结果决定最终定位。
