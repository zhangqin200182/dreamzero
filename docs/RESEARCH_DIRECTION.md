# 跨模态对齐的两种范式：从 JEPA 到 Attention 层对齐

> 基于对 DreamZero、FastWAM、COSMOS 3、π₀ 和 JEPA 的深入分析
> 2026-08-11

---

## 摘要

未来多模态模型架构存在两种跨模态对齐的范式。

**范式 A（JEPA）**：通过 Encoder 将不同模态压缩到一个统一的隐空间，在此空间中进行对齐和预测。不同模态的表示在 Encoder 输出处被强制拉到同一向量空间。

**范式 B（Attention 层对齐）**：不同模态保留独立的 Expert（各自的编码器、Q/K/V 投影、FFN），仅在 Attention 层通过 Q·K^T 内积建立跨模态连接。对齐发生在每一层的 Joint Self-Attention 中，不留存到下一层。

范式 B 已经被 π₀（LLM + Action Expert）、FastWAM（Video + Action Expert）、COSMOS 3（多模态 MoT）和 DreamZero（全共享 DiT）各自独立实现并验证。但尚未被明确识别为一种独立于 JEPA 的跨模态对齐范式。

我们通过 7 组因果实验在 DreamZero 上系统验证了范式 B 的关键属性：**推理弱耦合**（一个模态的输出质量不影响另一模态的预测精度）和**训练强耦合**（一个模态的损失函数梯度可以通过共享 Attention 传播到另一模态）。这两个属性在范式 A（统一隐空间）中不成立——统一空间中的表示彼此交织，推理时无法分离，训练时 Encoder 承受全部对齐压力。

基于此，本文提出：**Attention 层对齐是比统一隐空间更优的跨模态对齐方案**，并从架构设计、训练策略、推理部署三个层面给出未来研究路线。

---

## 一、背景：跨模态对齐的两种思路

### 1.1 范式 A：统一隐空间（JEPA 路线）

```
JEPA (Joint Embedding Predictive Architecture):

  modality A ──→ Encoder_A ──→ shared_latent ──→ Predictor ──→ shared_latent  ←── Encoder_B ←── modality B
  (高维原始数据)   (压缩投影)     (统一向量空间)   (MLP/Transf)   (同空间预测)      (压缩投影)     (另一部分数据)
```

核心思想：不同模态（或同一模态的不同部分）被各自的 Encoder 压缩/扩展到同一个向量空间，在此空间中通过 Predictor 网络进行预测。I-JEPA（图像）、V-JEPA（视频）、Multi-modal JEPA 都遵循此范式。

**属性**：
- 对齐靠 Encoder 的参数完成（Encoder 承受全部跨模态对齐压力）
- 一旦进入统一空间，所有表示彼此交织——推理时无法分离
- Encoder 的信息压缩带来信息损失（如 video 56320-dim → 5120-dim）
- 统一的向量空间便于插值、可视化和迁移

### 1.2 范式 B：Attention 层对齐（VLA → WAM 路线）

```
Expert-Interleaved Self-Attention:

  modality A ──→ Expert A ──→ Q_a, K_a, V_a ──┐  ┌──→ o_proj_A → FFN_A → next layer
                                                 ├──┤
  modality B ──→ Expert B ──→ Q_b, K_b, V_b ──┘  └──→ o_proj_B → FFN_B → next layer
                                          │
                                    Joint Flash Attention
                                   (Q·K^T 内积, 无参数)
```

核心思想：不同模态保留独立的 Expert（各自的编码方式、各自的隐空间维度、各自的 Q/K/V/FFN 权重），仅在 Attention 层通过拼接 Q/K/V 和一次联合 Flash Attention 建立跨模态连接。对齐发生在 Attention 中——一个 Q·K^T 内积——不留存到 Attention 之外。

**属性**：
- 对齐靠 Attention 的 Q·K^T 内积完成（无参数跨模态路由）
- Attention 结束后，各 Expert 回到各自的表示空间——推理时可以分离
- 各模态保留独有表示空间，无信息损失
- 可以利用各 Expert 的预训练权重（LLM 的 PaliGemma、Video 的 Wan2.2 等）

### 1.3 范式 B 已被独立实现但未被识别

以下四个架构，尽管主干类型不同（DiT / LLM / MoT），都在 Attention 层进行跨模态对齐：

| | DreamZero | FastWAM | COSMOS 3 | π₀ |
|---|---|---|---|---|
| **Expert A** | 无分离 | Video DiT (30层) | AR Reasoner | LLM Gemma 2B (18层) |
| **Expert B** | 无分离 | Action DiT (30层) | DiT Generator | Action 300M (18层) |
| **对齐机制** | Token 混排 Attention | Joint Self-Attention | Mode Switch Attention | Joint Self-Attention |
| **主干来源** | Wan2.1 视频生成 | Wan2.2 视频生成 | 自研多模态预训练 | PaliGemma VLM |
| **论文归属** | GEAR Lab | 清华/上海AI Lab | NVIDIA | Physical Intelligence |
| **时间** | 2025 | 2026.3 | 2026.5 | 2025 |

**四个独立团队、四种不同主干、四个不同时间点，不约而同地走向了同一种跨模态对齐机制。** 这不是巧合——Attention 层对齐具有架构层面的必然优势。

---

## 二、核心问题：Attention 层对齐的耦合属性

虽然范式 B 已被广泛使用，但一个根本问题尚未被回答：**在 Attention 层对齐的框架下，不同模态之间的耦合关系是怎样的？**

具体而言：

| 问题 | 如果答案为"是" | 如果答案为"否" |
|------|--------------|--------------|
| 推理时，模态 A 的输出质量是否影响模态 B 的预测？ | 两者强耦合——质量一损俱损 | 两者弱耦合——质量各自独立 |
| 训练时，模态 A 的梯度是否通过共享 Attention 传播到模态 B？ | 训练强耦合——联合训练互相受益 | 训练不耦合——各模态各自优化 |

**这两个问题的答案决定了范式 B 和范式 A 的优劣对比。**

如果在 Attention 层对齐下，推理是弱耦合的（可以分离执行），训练是强耦合的（可以互相受益），那么它同时拥有范式 A 的训练优势（跨模态信号传递）和范式 A 不具备的推理优势（独立执行、多频率分离）。这将使 Attention 层对齐成为客观上更优的架构范式。

---

## 三、实验验证：DreamZero 上的因果分析

为回答上述问题，我们在 DreamZero 上进行了系统验证。选择 DreamZero 的原因：它是唯一**没有 Expert 分离**的全共享 DiT 架构——video 和 action 使用完全相同的 Q/K/V/FFN 权重。如果在此最"紧"的架构中仍观察到弱耦合，则 Expert 分离的架构中更应成立。

### 3.1 推理时：弱耦合成立

**实验 A：flow_pred 噪声注入。** 在 16 步去噪的每一步，往视频 flow_pred 注入 σ=0~1.0 的高斯噪声。

| 噪声 σ | Action MSE | vs Full |
|--------|-----------|---------|
| 0.0 (Full) | 24.78 | baseline |
| 0.05 ~ 1.0 | 24.65 ~ 24.82 | ±0.1 |
| AO (skip video) | 41.31 | +16.53 |

**实验 B：video latents 替换。** 每步用纯噪声/全零取代 video latent。

| 条件 | Action MSE | vs Full |
|------|-----------|---------|
| Full | 21.61 | baseline |
| Random latents | **18.69** | **-13%** |
| Zero latents | **18.75** | **-13%** |
| AO | 49.17 | +127% |

**实验 C：第一帧语义扰动。** 模型仅使用第一帧（CLIP 编码）。双 checkpoint 验证。

| 扰动类型 | 对 Action 的影响 |
|---------|----------------|
| 换另一个 episode | **0% 退化** |
| 旋转 180°、遮掉一半 | **0% 退化** |
| 全黑/全白 | ~12% 退化 |
| 随机噪声/非真实图像 | ~20% 退化 |
| AO（关掉视频） | **100% 退化** |

**统一结论**：视频输出质量不影响动作精度。视频的贡献是二值的——ON（video tokens 在 attention 中在场）或 OFF。输出值无关。

### 3.2 训练时：强耦合成立

**实验 D：单步梯度传播。** 仅 backward video_loss，更新 LoRA 一步。

```
action_loss BEFORE:  5.65
action_loss AFTER:   2.33
Δ:                  -3.32 (-58.8%)
```

**实验 E：100 步持续优化。** 100 步只优化 video loss，固定验证集评测，与等量联合训练对比。

| | 纯视频优化 | 联合训练 (video+action) |
|---|---|---|
| **基线** | 1.059 | 2.169 |
| **100 步后** | 0.596 (**-43.7%**) | 0.316 (**-85.5%**) |
| **收敛模式** | 慢启动（70 步后加速） | 快饱和（10 步到最优） |

**统一结论**：Video loss 的梯度通过共享 Attention 权重传播到 action。纯视频优化无需任何 action 标注或梯度，即可将 action loss 减半。

### 3.3 这两种属性在范式 A 中不成立

```
                     范式 A (统一隐空间)              范式 B (Attention 对齐)
                  ┌────────────────────┐          ┌────────────────────┐
  推理时           │  表示彼此交织        │          │  Attention 结束即分离   │
  弱耦合           │  无法独立推理        │          │  支持多频率独立执行      │
                  │                    │          │                       │
  训练时           │  Encoder 承受全部     │          │  40层分布式对齐          │
  强耦合           │  对齐压力（难训练）     │          │  每层学一点（易训练）      │
                  │                    │          │  可利用预训练权重        │
                  └────────────────────┘          └────────────────────┘
```

---

## 四、范式 B 的应用：两个架构方向

### 4.1 利用弱耦合：多频率分离推理

LLM Expert、Video Expert、Action Expert 以不同频率独立执行：

```
语义层（LLM）：            ★               ★               ★
  "任务是什么？"             ↑ ~1 Hz（缓存 K/V，任务切换时刷新）

视觉层（Video）：         ★─★─★─★─★─★─★─★─★─★─★─★
  "画面变成什么样？"         ↑ ~10 Hz（缓存 K/V，N 个 action step 更新一次）

动作层（Action）：       ★★★★★★★★★★★★★★★★★★★★★★★★★★★★
  "具体关节位置？"          ↑ ~50 Hz（每步仅 forward Action Expert，O(1) 读缓存 K/V）
```

已有证据：π₀ prefix KV cache、FastWAM video KV cache 均验证了分离推理的精度无损。

### 4.2 利用强耦合：潜空间想象 RL

在范式 B 中，Video Expert 充当世界模型，LLM Expert 充当语义裁判，Action Expert 是被训练的策略：

```
1. LLM: "wipe the countertop" → 任务分解 → K_l 缓存（一次）
2. Action: N 条不同噪声种子的去噪轨迹 → N 组 (action, video_pred)
3. Video: autoregressive rollout 预测多步未来（世界模型）
4. LLM 裁判: "任务完成了吗？" → reward（语义级别的成功检测）
5. GRPO: advantage → ∂L/∂Expert_weights → 改善 Action
```

已有证据：我们的梯度传播实验（video loss → action），COSMOS 3 的 Forward Dynamics autoregressive rollout。

### 4.3 三专家多频率架构（最终形态）

将 LLM Expert、Video Expert、Action Expert 统一在同一组 Joint Self-Attention 中：

```
Layer i (共 N 层):

  LLM Expert (AR/causal):    Video Expert (DiT/diff):    Action Expert (DiT/diff):
    Q_l, K_l, V_l (2048)       Q_v, K_v, V_v (3072)        Q_a, K_a, V_a (1024)
         │                            │                           │
         └────────────────────────────┼───────────────────────────┘
                                      ↓
                   Joint Flash Attention (cat Q, K, V)
                                      ↓
              各自 o_proj → residual → FFN → next layer
```

**与现有架构的关系**：

```
π₀ (LLM+Action)                FastWAM (Video+Action)         COSMOS 3 (多模态 MoT)
       ↓ +Video Expert                ↓ +LLM Expert                 ↓ +频率分离 + RL
       └────────────────── 三专家多频率 ──────────────────┘
```

**最小可行实现**：从 π₀ (LLM + Action, 已开源, 已有 PyTorch port) 出发，加 Video DiT Expert（Wan2.2-5B 或冻结主干），LoRA fine-tune 双 Expert（LLM 冻结）。第一轮实验只需验证：三专家推理分离后精度不变，且额外视频训练能改善 action。

---

## 五、为什么 Attention 对齐是更优的范式

| | 范式 A: 统一隐空间 | 范式 B: Attention 对齐 |
|---|---|---|
| **对齐机制** | Encoder 参数压缩 | Q·K^T 内积（无参数） |
| **信息损失** | 有（到统一维度） | 无（各保留独立空间） |
| **预训练利用** | 难（Encoder 随机初始化） | 易（LLM/Video DiT 预训练权重） |
| **推理分离** | 不可（表示彼此交织） | **可以**（弱耦合） |
| **训练耦合** | Encoder 承受全部压力 | **分布式**（40 层，每层学一点） |
| **模态扩展** | 需重训 Encoder | 加 Expert + token |
| **多频率执行** | 不支持 | **天然支持**（KV cache 缓存） |
| **想象 RL** | 需额外组件 | **内置**（Video Expert = 世界模型） |

**不是 JEPA 不好，而是在多模态机器人控制的场景下，Attention 对齐是更适配的架构选择。**

---

## 六、研究路线图

```
Phase 1（已完成）: 先导验证
  ✅ DreamZero 7 组因果实验（弱耦合 + 强耦合）
  ✅ 四个架构深度分析（DreamZero / FastWAM / COSMOS 3 / π₀）
  ✅ JEPA 范式对比
  ✅ 统一理论框架建立

Phase 2（4-6 周）: 弱耦合通用性验证 + 推理加速
  - π₀ prefix 扰动实验（平行 DreamZero 实验 A/C）
  - π₀/FastWAM 多频率分离推理实现
  - 验证：弱耦合是 Attention 对齐的通用属性，与主干类型无关

Phase 3（6-8 周）: 三专家架构原型
  - 从 π₀ (LLM + Action) 出发，加 Video DiT Expert
  - LoRA fine-tune，验证三专家推理分离
  - 产出：三专家多频率推理 demo

Phase 4（8-12 周）: 想象 RL 训练
  - COSMOS 3 上实现 GRPO 训练管线
  - Reasoner-based reward + FD rollout
  - 对比：SFT vs SFT+RL 的 action 精度

Phase 5（综合）: 论文撰写
  - 整合 Phase 2-4 的实验结果
  - 定位：提出 Attention 层对齐作为独立范式，系统验证其属性
```

---

## 七、总结

> 跨模态对齐有两种范式：JEPA 的统一隐空间（Encoder 压缩对齐），和 VLA→WAM 的 Attention 层对齐（Expert 独立编码 + Q·K^T 无参数路由）。后者已被四个独立架构不约而同地实现，但从未被识别为一种独立的对齐范式。我们通过因果实验揭示了其关键属性——推理弱耦合（可分离执行）和训练强耦合（梯度互通）——并论证了它作为未来多模态架构方向的优势。
