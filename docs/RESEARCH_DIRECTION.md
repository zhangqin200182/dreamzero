# 跨模态交互的两种范式：共享隐空间与 Attention 层路由

> 基于对 DreamZero、FastWAM、COSMOS 3、π₀ 和 JEPA 的深入分析
> 2026-08-11

---

## 摘要

未来多模态模型架构存在两种跨模态信息传递的范式。

**范式 A（共享隐空间）**：通过 Encoder 将不同模态压缩到一个统一的隐空间——CLIP 的对比学习、多模态 LLM 的 projection layer、JEPA 的 shared latent——都在此范畴。不同模态的表示在 Encoder 输出处被强制拉到同一向量空间。这是**表示对齐**（representation alignment）。

**范式 B（Attention 层跨模态路由）**：不同模态保留独立的 Expert（各自的编码器、Q/K/V 投影、FFN），仅在 Attention 层通过 Q·K^T 内积进行信息路由。Attention 结束后各 Expert 回到独立表示空间。这是**信息路由**（information routing），不是表示对齐——各模态的表示从未被"拉到一起"，只是通过 softmax(QK^T) 有选择地传递信息。

"对齐"和"路由"是两个正交的逻辑层次：JEPA/CLIP 通过对比学习或潜空间预测使不同模态的表示可互换，Attention 通过内积权重使不同模态的 token 互相传递信息。**前者改变表示空间，后者不改变。**

范式 B 已经被 π₀、FastWAM、COSMOS 3 和 DreamZero 四个独立架构不约而同地实现。但尚未被明确识别为一种独立于共享隐空间的跨模态信息传递范式。**本文的核心贡献是识别并系统化这一范式。**

我们通过 7 组因果实验在 DreamZero 上揭示了范式 B 的关键属性：**推理弱耦合**（一个模态的输出质量不影响另一模态的预测精度）和**训练强耦合**（在共享参数架构中，一个模态的损失函数梯度可以通过共享 Attention 传播到另一模态）。这两个属性在范式 A（共享隐空间）中不成立——统一空间中的表示彼此交织，推理时无法分离；训练时 Encoder 承受全部交互压力。

基于这些属性，范式 B 在多模态机器人控制场景中具备独特的架构优势（多频率分离推理、预训练复用、潜空间想象 RL）。本文从架构设计、训练策略、推理部署三个层面给出验证路线图，并根据 P0 实验的结果选择论文最终定位——范式识别或范式优越性。

---

## 一、背景：跨模态交互的两种思路

### 1.1 范式 A：共享隐空间（表示对齐）

```
共享隐空间方法（CLIP / 多模态 LLM projection / JEPA）：

  modality A ──→ Encoder_A ──→ shared_latent ──→ 在此空间中进行下游任务
  modality B ──→ Encoder_B ──→ shared_latent
  (不同模态)     (压缩/投影)    (统一向量空间)
```

核心思想：通过 Encoder 将不同模态压缩/扩展到同一个向量空间，在此空间中表示可以直接比较（CLIP 的 cosine similarity）、直接拼接（多模态 LLM 的 token concat）或进行预测（JEPA 的潜空间预测）。

**关键属性**：
- Encoder 承受全部跨模态对齐压力（通过对比学习、重建损失或预测损失训练）
- 一旦进入统一空间，所有表示彼此交织——推理时**无法分离**
- 信息压缩带来信息损失（如 video 56320-dim → 5120-dim）
- 好处：统一空间便于跨模态检索、零样本迁移、表示可视化

### 1.2 范式 B：Attention 层跨模态路由（信息路由）

```
Expert-Interleaved Self-Attention:

  modality A ──→ Expert A ──→ Q_a, K_a, V_a ──┐  ┌──→ o_proj_A → FFN_A → next layer
                                                 ├──┤
  modality B ──→ Expert B ──→ Q_b, K_b, V_b ──┘  └──→ o_proj_B → FFN_B → next layer
                                          │
                                    Joint Flash Attention
                                   (Q·K^T 内积, 无参数)
```

核心思想：不同模态保留独立的 Expert（各自的编码方式、各自的隐空间维度、各自的 Q/K/V/FFN 权重），仅在 Attention 层通过 Q·K^T 内积进行**信息路由**——softmax(QK^T) 决定每个 token 从其他 Expert 获取多少信息。这不是表示对齐，各模态的表示在 Attention 前后始终留在各自空间中。

**关键属性**：
- 路由靠 Attention 的 Q·K^T 内积完成（无参数）
- Attention 结束后，各 Expert 回到独立表示空间——推理时**可以分离执行**
- 各模态保留独有表示空间，**无信息损失**
- 可以利用各 Expert 的**预训练权重**（LLM 的 PaliGemma、Video 的 Wan2.2 等）

**对齐 vs 路由的本质区别**：

| | 表示对齐（范式 A） | 信息路由（范式 B） |
|---|---|---|
| **目标** | 使不同模态的表示可互换 | 使不同模态的 token 可互相传递信息 |
| **机制** | Encoder 参数压缩/投影 | Attention Q·K^T 内积 |
| **表示空间** | 改变（压缩到统一维度） | 不改变（各自独立空间） |
| **逻辑层次** | 表示层 | 交互层 |
| **关系** | 两者**正交**——可以同时使用 | COSMOS 3 的 3D mRoPE = 表示对齐 + 信息路由

### 1.3 范式 B 已被独立实现但未被识别

以下四个架构，尽管主干类型不同（DiT / LLM / MoT），都在 Attention 层进行跨模态信息路由：

| | DreamZero | FastWAM | COSMOS 3 | π₀ |
|---|---|---|---|---|
| **Expert A** | 无分离 | Video DiT (30层) | AR Reasoner | LLM Gemma 2B (18层) |
| **Expert B** | 无分离 | Action DiT (30层) | DiT Generator | Action 300M (18层) |
| **路由机制** | Token 混排 Attention | Joint Self-Attention | Mode Switch Attention | Joint Self-Attention |
| **主干来源** | Wan2.1 视频生成 | Wan2.2 视频生成 | 自研多模态预训练 | PaliGemma VLM |
| **论文归属** | GEAR Lab | 清华/上海AI Lab | NVIDIA | Physical Intelligence |
| **时间** | 2025 | 2026.3 | 2026.5 | 2025 |

**四个独立团队、四种不同主干、四个不同时间点，不约而同地走向了同一种跨模态路由机制。** 这不是巧合——Attention 层路由具有架构层面的必然优势。

---

## 二、核心问题：Attention 层路由的耦合属性

虽然范式 B 已被广泛使用，但一个根本问题尚未被回答：**在 Attention 层路由的框架下，不同模态之间的耦合关系是怎样的？**

具体而言：

| 问题 | 如果答案为"是" | 如果答案为"否" |
|------|--------------|--------------|
| 推理时，模态 A 的输出质量是否影响模态 B 的预测？ | 两者强耦合——质量一损俱损 | 两者弱耦合——质量各自独立 |
| 训练时，模态 A 的梯度是否通过共享 Attention 传播到模态 B？ | 训练强耦合——联合训练互相受益 | 训练不耦合——各模态各自优化 |

**这两个问题的答案决定了范式 B 相比范式 A 的差异化价值。**

如果在 Attention 层路由下，推理是弱耦合的（可以分离执行），且训练强耦合在特定架构条件下成立，那么它将同时具备范式 A 的训练优势（跨模态信号传递）和范式 A 不具备的推理优势（独立执行、多频率分离）。这将使 Attention 层路由成为多模态机器人控制中极具吸引力的架构选择。

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

### 3.3 关键 nuance：Expert 分离的连续谱与逻辑不对称

四个架构并非同质的"Expert 分离"，而是分布在一条连续谱上：

```
全共享 ←────────────────────────────────────────────────→ 全分离

DreamZero         FastWAM            π₀                COSMOS 3
Q/K/V/FFN 全共享   Q/K/V/FFN 完全独立  Q/K/V/FFN 完全独立  独立 + 不同 attention mode
仅 token 类型区分   但 head_dim 强制对齐  width 不同          (causal vs full)
                  (3072=3072)       仅 head_dim=256 对齐   gen 有独立 MoE router
```

**逻辑不对称（review v2 指出）**：文档从 DreamZero（最左端）推断整个谱的行为。这个逻辑对**弱耦合**成立（共享参数下输出质量都不传递 → 分离参数下更不传递）。但对**强耦合**不成立——DreamZero 的强耦合来自共享参数（Q/K/V/FFN 完全相同），在 Expert 分离架构中，梯度路径完全不同：

```
DreamZero:  video_loss → ∂/∂(shared_QKV) → 直接影响 action（实验 D/E 验证）
FastWAM:    video_loss → ∂/∂(video_QKV) → [仅通过 softmax 中的 video_Q·action_K^T] → ∂/∂(action_K)
π₀:         LLM_loss →  [仅通过 softmax 中的 LLM_Q·action_K^T] → ∂/∂(action_K)
```

在分离架构中，梯度仅通过 attention score 矩阵中的跨模态项传播——这是一条**间接且可能极弱的路径**。

**结论**：推理弱耦合是已验证的通用属性，训练强耦合的泛化边界是当前框架中**最高优先级的待验证假设**。

### 3.3.1 表示层面的连续谱：Video Expert 内部表示的结构化程度

除了架构层面的 Expert 分离程度，还存在一个正交的设计维度：**Video Expert 内部表示的结构化程度**。

```
表示层面的连续谱（以 Video Expert 为例）：

  无结构向量 ──────────────→ 结构化特征图 ──────────────→ 显式几何表示

  Wan2.1 VAE latent      ViT feature tokens           3D Gaussian Splatting
  (16×H×W, compressed)   (patch embeddings)           (means + cov + features)
  DreamZero / FastWAM                                  (可能的扩展方向)
```

3D Gaussian Splatting（3DGS）不改变 Attention 路由机制——它改变的是 Video Expert **内部**的表示格式。两者的关系是正交的：Expert 分离程度决定"谁跟谁交互"（架构轴），内部表示格式决定"交互时携带什么信息"（表示轴）。3DGS 填的是"Expert 分离架构 + 结构化几何表示"这一组合格——连续谱的右下角。

### 3.4 实验 B 的重新解释：有害干扰假说

Random/zero latent 替换 → action 改善 13%。代码级解释（`RESEARCH_DIRECTION_REVIEW.md` v2）：DreamZero 使用独立 per-token noise schedule——Video: `Beta(3,1)`（偏低压噪声），Action: `Uniform`。共享的 Q/K/V 权重需要同时服务两种不同的 denoising 动态 → 两种冲突的降噪目标在 attention 中相互干扰 → video latent 对 action token 产生有害表示偏移。Random/zero latent 意外消除了这种冲突。

**如果这个解释成立，Expert 分离不是可选的优化，而是防止模态间有害干扰的必要架构设计。** 这可能比"弱耦合推理加速"更强——它不依赖"训练强耦合是否泛化"的实验结果，仅从推理时行为就能证明 Expert 分离的架构优势。

**P2 验证实验**：在 Expert 分离架构（FastWAM/π₀）中做相同替换。预测：不会看到 action 改善，因为本身没有有害干扰需要消除。

### 3.5 这两种属性在共享隐空间中均不成立

```
                     范式 A (共享隐空间)              范式 B (Attention 路由)
                  ┌────────────────────┐          ┌────────────────────┐
  推理时           │  表示彼此交织        │          │  Attention 结束即分离   │
  弱耦合           │  无法独立推理        │          │  支持多频率独立执行      │
                  │                    │          │                       │
  训练时           │  Encoder 承受全部     │          │  40层分布式路由          │
  强耦合           │  交互压力（难训练）     │          │  每层学一点（易训练）      │
                  │                    │          │  可利用预训练权重        │
                  └────────────────────┘          └────────────────────┘
```

---

## 三-附、与标准 Behavior Cloning 的基线对比

在讨论范式 B 的应用之前，需要先回答一个更基础的问题：**相比机器人学习的实际默认基线——标准 Behavior Cloning——Attention 路由有什么优势？**

### 标准 BC 的结构

```
标准 BC 架构：
  Vision Encoder (ResNet/ViT) ──→ concat(language, state) ──→ Action Decoder (MLP/Transformer)
```

标准 BC 的特征：
- 所有模态的表示在 Encoder 输出处一次性 concat
- 没有迭代的跨模态交互（单 pass 前馈）
- 没有 Expert 概念——所有模态共享同一套表示压缩逻辑
- Vision Encoder 通常是轻量的（ResNet-50 或 SigLIP），无生成能力

**标准 BC 在本质上更接近范式 A（共享隐空间）**，区别在于 BC 的"隐空间"很小（特征向量 concat），且没有 JEPA 的自监督预训练目标。在机器人学习社区，BC 才是真正的工程基线。

### 从 BC 到 Attention 路由的进化谱系

```
BC → 共享 Encoder → JEPA → DreamZero 全共享 DiT → FastWAM Expert 分离 → π₀ LLM+Action → COSMOS 3 全分离
│        (范式 A 实用版)    (范式 A 理论版)    (范式 B, 最左端)     (范式 B, 中间)     (范式 B, 右)   (范式 B, 最右端)
│
└── 简单，但所有模态共享同一表示压缩，单 pass 无法迭代推理
```

### 范式 B 的三个优势直接对治 BC 局限

| BC 局限 | 范式 B 的解决方案 |
|---------|-----------------|
| 单 pass 无迭代推理 | 40 层 Joint Self-Attention，每层重新路由信息，不同层可关注不同粒度的跨模态交互 |
| 统一表示压缩导致信息损失 | 各 Expert 保留独立表示空间，无强制维度压缩，Video Expert 天然支持生成 |
| 无法复用预训练权重 | LLM Expert（PaliGemma）、Video Expert（Wan2.2）可直接加载预训练权重 |
| 所有模态同频率执行 | 多频率分离推理：慢模态（LLM、Video）缓存 K/V，快模态（Action）轻量执行 |

**对本研究的启示**：范式 B 的 motivation 不应只是"我们与 JEPA 不同"，而应是"它解决了 BC 的实际瓶颈（无迭代推理、无法复用预训练、无法多频率执行）"。BC 的局限恰好是 Attention 路由的优势，这个 motivation 直接面向机器人学习社区的核心关注点。

---

---

## 四、范式 B 的应用：架构方向与扩展性

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

### 4.4 扩展性：Expert 内部表示的可替换性

Attention 路由框架的一个重要特性是：**各 Expert 的内部表示可以自由替换，而不影响跨模态路由机制**。路由仅依赖 Q·K^T 内积，对 Expert 内部的表示格式无约束。

以 Video Expert 为例，其内部 latent 可以从无结构向量逐步升级为结构化几何表示：

| 表示类型 | 具体形式 | 优势 | 当前状态 |
|---------|---------|------|---------|
| 无结构向量 | Wan2.1 VAE latent（16×H×W） | 成熟、压缩率高 | DreamZero / FastWAM 在用 |
| 结构化特征图 | ViT patch embeddings | 保留空间结构、便于定位 | 可探索 |
| 显式几何表示 | 3D Gaussian Splatting（means + cov + features） | 精确物理约束、3D 一致性 | 同事提案中 |

三种表示在 Attention 路由中的行为完全一致——对外暴露的都是 K/V token 序列。差异仅在 Video Expert 内部的编码器实现。

**这与范式 A（共享隐空间）形成关键对比**：在共享隐空间中，更换任一模态的内部表示都需要重新训练 Encoder，因为 Encoder 的输出必须映射到统一空间。而在 Attention 路由中，只要 Expert 输出的 token 序列格式不变，内部表示可以任意替换。这使得 3D Gaussian 等方向可以自然地整合进框架——不要求修改路由机制，只需替换 Video Expert 的内部编码器。

这一特性也为 4.3 节的三专家架构提供了灵活性：Video Expert 的内部表示可以根据任务需求（精度 vs 效率）在三种类型之间切换，而不影响 LLM → Action 的已有路由路径。

---

## 五、范式 B 的架构特性对比

| | 范式 A: 共享隐空间 | 范式 B: Attention 路由 |
|---|---|---|
| **交互机制** | Encoder 参数压缩 | Q·K^T 内积（无参数路由） |
| **信息损失** | 有（压缩到统一维度） | 无（各保留独立空间） |
| **预训练利用** | 难（Encoder 随机初始化） | 易（LLM/Video DiT 预训练权重） |
| **推理分离** | 不可（表示彼此交织） | **已验证**（弱耦合，四个架构共同确认） |
| **训练耦合** | Encoder 承受全部压力 | **待验证泛化**（DreamZero 共享参数中成立，P0 决定是否泛化到 Expert 分离架构） |
| **模态扩展** | 需重训 Encoder | 加 Expert + token |
| **多频率执行** | 不支持 | **天然支持**（KV cache 缓存） |
| **潜空间 RL** | 需额外组件 | **内置**（Video Expert = 世界模型） |

**注意**：训练耦合一行的"待验证泛化"是关键约束。在 DreamZero（全共享参数架构）中，训练强耦合已通过实验 D/E 确认。但这一属性是否泛化到 Expert 分离架构（FastWAM、π₀、COSMOS 3），取决于 P0 实验。如果 P0 不成立，"训练强耦合"将重新定义为共享参数架构的特性，而非 Attention 路由范式的通用属性。

**从已确认的属性看**，范式 B 在多模态机器人控制场景中具备独特的工程优势（推理分离、预训练复用、多频率执行）。这些优势独立于 P0 的结果，足以支撑范式 B 作为一种有前景的架构方向。

---

## 七、研究路线图（修正版）

### 7.1 实验优先级

> 基于 `RESEARCH_DIRECTION_REVIEW.md` 的代码级分析，重新排序实验优先级。

| 优先级 | 实验 | 验证内容 | 当前状态 | 风险 |
|--------|------|---------|---------|------|
| **P0** | π₀ 梯度传播（平行 DreamZero 实验 D/E） | "训练强耦合"是否泛化到 Expert 分离架构 | **未做** | **高**：如果不成立，理论框架需要重大修正 |
| **P1** | π₀ prefix 扰动（平行 DreamZero 实验 A/C） | "推理弱耦合"是否泛化到 LLM+Action 架构 | **未做** | 低：FastWAM/π₀ 已有主动弱耦合证据 |
| **P2** | π₀/FastWAM random latent 替换（平行 DreamZero 实验 B） | Expert 分离架构中是否存在"有害干扰" | **未做** | 中：如果不存在有害干扰，是范式 B 的重要优势 |
| **P3** | FastWAM 梯度传播 | 纯 DiT Expert 分离的中间验证点 | **未做** | 中 |

### 7.2 分阶段计划

```
Phase 1（已完成）: 先导验证 + 理论建立
  ✅ DreamZero 7 组因果实验
  ✅ 四个架构代码级分析
  ✅ JEPA 范式对比
  ✅ 跨模态交互范式文档
  ✅ review 反馈整合

Phase 2（P0/P1/P2, 3-4 周）: 关键假设验证
  - P0: π₀ 梯度传播实验——决定"训练强耦合"的泛化边界
  - P1: π₀ prefix 扰动——确认"推理弱耦合"泛化
  - P2: π₀ random latent 替换——验证"有害干扰"假说
  - 根据结果决定论文最终定位

Phase 3（6-8 周）: 三专家架构原型
  - 从 π₀ (LLM + Action) 出发，加 Video DiT Expert
  - 简化方案（基于 review 建议）：Video Expert 仅做 prefix 编码（如 π₀ 的 SigLIP），
    不参与 diffusion rollout。本质是 FastWAM 的 prefill_video_cache 策略
  - 需注意 generation paradigm 冲突（AR LLM vs Diffusion Video）
    参考 COSMOS 3 的 two_way_attention 解决方案

Phase 4（12-16 周）: 想象 RL 训练
  - COSMOS 3 上实现 GRPO 训练管线（从零构建——目前仅有 SFT 配方，需预留充分工程时间）
  - Reasoner-based reward + FD rollout

Phase 5（综合）: 论文撰写
  - 层次 1（已验证，可立即撰写）：Attention 路由作为独立范式 + 推理弱耦合通用性
  - 层次 2（待 P0 验证）：训练强耦合边界 + Expert 分离 vs 全共享优劣
```

### 7.3 论文分层策略

基于 review 建议，将论文定位分为两个可独立发表的层次：

**层次 1（已可撰写）**：Attention 层路由是一种独立的跨模态交互范式
- 四个架构的代码级收敛性证据
- 推理弱耦合是其通用属性（π₀ prefix cache, FastWAM video cache, COSMOS 3 text cache 共同验证）
- 多频率分离推理是自然推论
- **贡献级别**：识别+系统化——将已有但未被连接的独立发现统一为范式

**层次 2（待 P0 验证）**：范式 B 的训练耦合属性与 Expert 分离的优势
- P0 决定"训练强耦合"是否泛化
- P2 决定"有害干扰"假说是否成立（Expert 分离的必要性）
- 三专家架构的统一
- **贡献级别**：机制分析+架构创新——发现新属性并提出新架构

---

## 八、总结

> 跨模态交互有两种范式：共享隐空间（Encoder 压缩对齐，如 JEPA、CLIP），和 Attention 层路由（Expert 独立编码 + Q·K^T 无参数信息路由）。后者已被四个独立架构不约而同地实现，但从未被识别为一种独立的交互范式。我们通过因果实验揭示了其关键属性——推理弱耦合（可分离执行，四个架构共同确认）和训练强耦合（在共享参数架构中已确认，泛化性由 P0 实验决定）。基于已确认的弱耦合属性，范式 B 在多频率分离推理、预训练复用、潜空间 RL 等方面具备独特的架构潜力。
