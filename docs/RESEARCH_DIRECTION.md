# 基于 Attention 路由的下一代机器人世界模型架构

> 基于对 JEPA、DreamZero、FastWAM、COSMOS 3、π₀ 的代码级分析
> 2026-08-11

---

## 摘要

**COSMOS 3 证明了训练应该在一起（共享参数 = 语义/物理/视觉知识迁移到 Action）。FastWAM 证明了推理应该分开（Expert 分离 = 多频率 + 无有害干扰 + 4× 加速）。DreamZero 实验揭示了推理弱耦合和训练强耦合是 Attention 路由的通用属性。三者共同指向同一个目标架构：训练时共享 Attention 实现知识迁移，推理时 KV cache 分离实现多频率执行。**

本文梳理了从现有架构中识别 Attention 路由范式、揭示其耦合属性、推导目标架构、设计先导实验的完整逻辑链。

---

## 一、起点：两个关键观察

### 1.1 COSMOS 3 的全共享架构：训练在一起的价值

COSMOS 3 的 AR Reasoner 和 DiT Generator 使用**同一组 Transformer blocks、同一组 Q/K/V/FFN 权重**，仅 attention mask 不同（causal vs full）。

```
                    COSMOS 3 全共享架构的训练知识迁移

  AR 训练阶段                                 DiT Action 生成阶段
  ┌─────────────────────────┐              ┌─────────────────────────┐
  │ "pick up the blue cup"  │              │ Policy 模式:              │
  │   → 蓝色杯子的视觉特征    │   共享权重    │ 处理同一段指令文本          │
  │   → "抓取"的运动语义      │──────────→  │ 用的是同一套 W_Q/W_K/W_V   │
  │   → 物体间空间关系        │  Q/K/V/FFN  │ 生成动作序列                │
  │   → 物理合理性            │              │ [gripper_open,             │
  │   → 时序因果关系          │              │  move_to(x,y,z),           │
  │                         │              │  gripper_close]            │
  └─────────────────────────┘              └─────────────────────────┘
         ↑                                              │
         │          DiT→AR 反向迁移                      │
         └──────────────────────────────────────────────┘
         Generator 在 latent space 中学到的连续动态
         → 增强 Reasoner 的时序理解和物理判断
```

这个设计不是偶然的——全共享参数在训练中产生四个层面的正迁移：

1. **语言理解 → 动作指令落地**。AR 模式的指令理解能力通过**同一套 Q 投影矩阵**直接用于 Action 生成。Policy 推理时，文本指令经过同一套 Q 投影矩阵编码为 Key/Value，这些 representation 在 AR 训练中已经被"教"会了什么是杯子、什么是蓝色、什么是放置。DiT 去噪时的 cross-attention 可以直接利用这些语义信息约束动作生成。没有共享权重的话，Generator 的 cross-attention 需要从零学语言理解。

2. **物理推理 → 动作合理性约束**。Reasoner 明确训练了物理世界推理（Physical Plausibility Analysis, VideoPhy-2 SFT）：模型学会了物体不穿墙、重力向下、因果关系、物体恒存性。共享的 FFN 层将同样的"物理常识"编码在权重中，Policy 生成时自动避免不合理的 action。

3. **视觉理解 → 更好的状态表征**。Reasoner 模式训练了丰富的视觉理解（captioning、temporal localization、grounding）。Policy 的输入是同一张图像、经过同一个视觉塔、同一个 Transformer 编码。如果模型已在 AR 模式学会"这是一个机械臂，处于抓取姿态，目标物体在 (x, y)"，DiT 模式去噪 action tokens 时不需要从头学。

4. **Action CoT → 推理与生成的桥梁**。Reasoner 的 Action CoT 任务——"给出 pick up the flower 的 2D 轨迹"→ 模型输出 `<think> I will move gripper to [713,680]...</think>` 后跟轨迹坐标——和 Generator Policy 的视觉→动作预测在本质上是同一件事。共享权重意味着语言化的动作推理可以直接加速连续动作空间的生成。

**迁移是双向的**：DiT 训练同样帮助 AR 推理。Generator 训练期间，模型在 latent space 中学会了视频帧之间的连续动态——物体如何运动、场景如何变化。这种"动态直觉"编码在共享权重中，可以增强 Reasoner 的时序事件理解、下一动作预测和物理合理性判断。AR→DiT 和 DiT→AR 的正向循环是全共享架构设计的一个完整论证。

**配方证据**：Policy-DROID 训练不是从零开始的 Generator-only 模型，而是从完整的 Cosmos3-Nano omni-checkpoint 启动。如果 AR 能力对 Policy 没用，NVIDIA 完全可以从 Generator 子集启动。

**一句话**：COSMOS 3 的核心赌注——同一组参数在"理解世界"和"生成世界"之间产生**双向正迁移**——已被验证。AR 训练给了 Generator 语义理解、物理常识、视觉表征；DiT 训练给了 Reasoner 动态直觉。**对 Action 生成来说，AR 带来的语言理解和物理推理能力直接降低了 Policy 学习所需的样本量和训练时间。** 这是全共享架构存在的根本原因，也是我们目标架构必须继承的能力。

### 1.2 FastWAM 的 Expert 分离：推理分开的价值

FastWAM（Wan2.2-5B, 清华/上海 AI Lab, 2026.3）将 Video Expert（3072 dim, 30 层）和 Action Expert（1024 dim, 30 层）分离为独立的 Q/K/V/FFN，仅通过 Joint Self-Attention 在每层进行跨模态路由：

```
                FastWAM 双 Expert 分离架构

  训练时（共享 Attention）:              推理时（KV cache 分离）:
  ┌───────────────────────┐            ┌───────────────────────┐
  │ Video    Action       │            │ Video (1 次)           │
  │ Q_v,K_v  Q_a,K_a     │            │ forward → K/V 缓存     │
  │   └──┬───┘            │            │          ↓             │
  │   Joint Attn          │            │ Action (N 步)          │
  │   split → FFN_v/FFN_a │            │ 每步 forward Action     │
  └───────────────────────┘            │ + O(1) 读缓存 K/V      │
                                       └───────────────────────┘
```

推理时分离执行：
- `prefill_video_cache`：Video tokens 一次 forward → 缓存 30 层 K/V
- `forward_action_with_video_cache`：Action 每步去噪仅 forward Action Expert + O(1) 读取 video K/V

论文消融结论：
- 推理时去掉视频生成，精度几乎不变（LIBERO 97.6 vs 98.0，RoboTwin 91.8 vs 91.3）
- 训练时去掉视频联合建模，精度暴跌（LIBERO -4.1，RoboTwin -8.0）
- 延迟 4.3× 降低（190ms vs 810ms）

**FastWAM 证明了：测试时不需要生成未来视频，但训练时必须联合建模视频和动作。这是"推理弱耦合 + 训练强耦合"的最早独立证据。**

---

## 二、洞察：Attention 路由是统一的底层范式

### 2.1 四个架构，同一个机制

| | DreamZero | FastWAM | COSMOS 3 | π₀ |
|---|---|---|---|---|
| **Expert A** | 无分离 | Video DiT (30层) | AR Reasoner | LLM Gemma 2B (18层) |
| **Expert B** | 无分离 | Action DiT (30层) | DiT Generator | Action 300M (18层) |
| **跨模态机制** | Token 混排 Attention | Joint Self-Attention | Mode Switch Attention | Joint Self-Attention |
| **主干类型** | DiT | DiT | MoT（AR+DiT） | LLM（AR） |
| **团队/时间** | GEAR Lab / 2025 | 清华&上海AI Lab / 2026.3 | NVIDIA / 2026.5 | Physical Intelligence / 2025 |

四个独立团队、四种不同主干（Wan2.1 / Wan2.2 / Gemma2B+PaliGemma / 自研 MoT）、三个不同时间点，不约而同走向了同一个底层机制：**Attention 层跨模态路由**——不同模态的 Expert 在 Attention 层通过 Q·K^T 内积进行无参数信息路由，Attention 结束后各回各自表示空间。

这与 JEPA 的**潜空间对齐**（Encoder 将不同模态压缩到统一隐空间）是两种正交的跨模态交互范式。JEPA 做的是表示对齐——使不同模态可互换。Attention 做的是信息路由——使不同模态可互相传递信息。两者在逻辑层次上是正交的，COSMOS 3 的 3D mRoPE 同时使用了两种。

### 2.2 Expert 分离程度：连续谱而非二元

四个架构并非同质的"分离"，而是分布在一条连续谱上：

```
全共享 ←──────────────────────────────────────────────────→ 全分离

COSMOS 3          DreamZero        FastWAM            π₀
同一组 blocks      全共享 DiT        独立 Q/K/V/FFN     独立 Q/K/V/FFN
AR+DiT 共享 Q/K/V  仅 token 区分      2 Expert, 30 层     2 Expert, 18 层
仅 mask 不同                                               不同 width (2048 vs 1024)
```

**关键发现（基于代码级分析）**：DreamZero 位于谱的最左端（全共享 DiT）。我们的因果实验在此进行。将 DreamZero 的结论推广到整条谱时，"推理弱耦合"可以安全泛化（共享参数下输出质量都不传递 → 分离参数下更不传递），但"训练强耦合"不能——DreamZero 的强耦合来自共享参数（Q/K/V/FFN 完全相同），Expert 分离架构中梯度路径完全不同（仅通过 Q·K^T 中的跨模态项，间接且可能极弱）。**这是 P0 实验的根源。**

---

## 三、DreamZero 实验：Attention 路由的耦合属性

DreamZero 位于谱的最左端（全共享 DiT，video 和 action 使用完全相同的 Q/K/V/FFN 权重）。在这个耦合最紧的架构中测试，对 Expert 分离架构有最强的泛化约束。

### 3.1 推理弱耦合（实验 A-C，多维度验证）

**实验 A：flow_pred 噪声注入。** 在 16 步去噪的每一步，往视频 flow_pred 注入 σ=0~1.0 的高斯噪声。

| 噪声 σ | Action MSE | vs Full |
|--------|-----------|---------|
| 0.0 (Full) | 24.78 | 基线 |
| 0.05 | 24.73 | -0.05 |
| 0.1 | 24.82 | +0.04 |
| 0.3 | 24.73 | -0.05 |
| 0.5 | 24.81 | +0.03 |
| 1.0 | 24.65 | -0.13 |
| AO (skip video) | 41.31 | **+16.53 (+67%)** |

σ=0~1.0，Action MSE 波动 ±0.15 以内。完全关掉视频去噪（AO），Action +67%。**视频去噪对 Action 的贡献是二值的：ON 或 OFF。输出值无关。**

**实验 B：video latents 替换。** 每步用纯噪声/全零取代 video latent。

| 条件 | Action MSE | vs Full |
|------|-----------|---------|
| Full | 21.61 | 基线 |
| Random latents | **18.69** | **-13%**（反直觉） |
| Zero latents | **18.75** | **-13%**（反直觉） |
| AO | 49.17 | +127% |

**实验 C：第一帧语义扰动。** 模型仅使用第一帧（CLIP 编码），双 checkpoint 验证。

| 扰动类型 | ckpt-200 | ckpt-1000 |
|---------|---------|-----------|
| 换另一个 episode | +1% | 0% |
| 旋转 180° | — | -1% |
| 遮掉一半（左/右） | — | -1~3% |
| 全黑 | +12% | +14% |
| 全白 | +13% | — |
| 随机噪声 | +22% | +21% |
| 正弦波纹理（非照片） | — | +19% |
| AO（关掉视频） | +100% | +100% |

**统一结论**：视频输出质量与动作精度无因果关联。token 在场 = ON（保留 80-100% 的 AO-Full gap 收益），token 不在场 = OFF（退化 +100%）。CLIP 只需要"像真实照片"的通用锚点——换 episode、旋转、遮挡影响均为 0。**视频内容语义完全无关。**

### 3.2 训练强耦合（实验 D-E，在共享参数架构中验证）

**实验 D：单步梯度传播。** 从 checkpoint-1000 出发，取 4 个 batch，仅 backward video_loss，更新 LoRA 权重一步。改变集中在 FFN 层（`ffn.0`, `ffn.2`）——共享 DiT 中 video 和 action 共用的组件。

```
action_loss BEFORE:  5.65 → AFTER: 2.33
Δ: -3.32 (-58.8%)
```

**实验 E：100 步持续优化（含联合训练对照组）。** 100 步纯 video loss（组 A）vs 100 步联合训练（组 B，video+action loss）。固定验证集，每 10 步评测。

| | 组 A: 纯视频优化 | 组 B: 联合训练 |
|---|---|---|
| **Step 0（基线）** | 1.059 | 2.169 |
| **Step 10** | 1.184 (+11.8%) | 0.318 (-85.3%) |
| **Step 30** | 1.320 (+24.6%) | 0.355 (-83.6%) |
| **Step 50** | 1.035 (-2.3%) | 0.308 (-85.8%) |
| **Step 70** | 1.210 (+14.3%) | 0.317 (-85.4%) |
| **Step 90** | 0.559 (-47.3%) | 0.309 (-85.8%) |
| **Step 100（最终）** | **0.596 (-43.7%)** | **0.316 (-85.5%)** |

**关键观察**：组 B（联合训练）10 步到最优，之后 90 步饱和。组 A（纯视频优化）前 70 步震荡，最后 30 步加速改善。纯视频优化无需任何 Action 标注或梯度即可将 Action loss 减半。

**关键约束**：此结论在 DreamZero 的**全共享参数架构**中获得。在 Expert 分离架构（FastWAM）中，梯度路径完全不同——仅通过 Joint Attention 中 softmax(QK^T) 的跨模态项传播，不经过另一 Expert 的 Q/K/V/FFN 参数。**这是整个研究方向的 P0 优先级待验证假设（FastWAM F5）。**

### 3.3 有害干扰假说（实验 B 的代码级解释）

实验 B：全共享架构中 random latent → action 改善 13%。代码级原因：DreamZero 的 Video 使用 `Beta(3,1)` 分布采样 timestep，Action 使用 `Uniform` 分布。共享 Q/K/V 权重需要同时服务两种不同去噪动态 → 相互冲突 → video 对 action 产生有害表示偏移。Random latent 意外消除了这种冲突。

AO 模式的进一步证据：AO 本质是 sigma 在 16 步中全部卡在 1.0——16 次重复第一步，video K/V 完全静态。Fresh random 每步不同 K/V 保持 attention 多样性。

**如果假说成立，Expert 分离不是可选优化——是防止模态间有害干扰的必要架构设计。** Expert 分离架构中预期不存在此问题（各 Expert 独立 noise schedule，FastWAM F6 验证）。

---

## 四、目标架构：基于 Attention 路由的全模态多频率世界模型

### 4.1 架构设计

三 Expert（LLM + Video + Action），每层通过 Joint Self-Attention 路由信息，各自独立 Q/K/V/FFN。

```
              三 Expert 分离架构 = COSMOS 3 的能力 + FastWAM 的效率

  训练时（共享 Attention = 知识迁移）:    推理时（KV cache 分离 = 多频率）:
  ┌────────────────────────────┐       ┌────────────────────────────┐
  │ LLM    Video    Action     │       │ LLM (1 次, ~1Hz)           │
  │ Q_l,K_l Q_v,K_v Q_a,K_a   │       │ forward → K_l 缓存          │
  │   └──────┬──────┘          │       │          ↓                 │
  │    Joint Flash Attention   │       │ Video (1 次, ~10Hz)        │
  │    split → FFN_l/v/a       │       │ forward → K_v 缓存          │
  │                            │       │          ↓                 │
  │ AR→Action 知识迁移         │       │ Action (N 步, ~50Hz)       │
  │ 语义/物理/视觉/CoT         │       │ 每步 Action Expert forward  │
  │ + DiT→AR 动态直觉          │       │ + O(1) 读 K_l + K_v        │
  └────────────────────────────┘       └────────────────────────────┘
```

```
Layer i:

  LLM Expert (AR, causal)    Video Expert (DiT, diff)    Action Expert (DiT, diff)
  Q_l, K_l, V_l ──────────── Q_v, K_v, V_v ──────────── Q_a, K_a, V_a
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ↓
                         Joint Flash Attention
                                    ↓
                    各自 o_proj → residual → FFN → next layer

训练: 三 Expert 在共享 Attention 中联合训练 → AR 知识迁移到 Action（同 COSMOS 3 机制）
推理: Phase 1 LLM 单独 forward (causal, 1 次) → 缓存 K_l
      Phase 2 Video 单独 forward (full attention, ~10Hz) → 缓存 K_v
      Phase 3 Action 每步 forward + 读 K_l, K_v (~50Hz)
      Block-causal mask（π₀ 已验证）：不存在 AR vs DiT 冲突
```

```
    三个架构的对比

    COSMOS 3                    FastWAM                    三 Expert (目标)
    全共享参数                    双 Expert 分离                三 Expert 分离
    ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
    │ AR ←→ DiT       │         │ Video ←→ Action │         │ LLM ←→ Video    │
    │ 同一组 Q/K/V/FFN │         │ 独立 Q/K/V/FFN   │         │   ←→ Action     │
    │                  │         │                  │         │ 独立 Q/K/V/FFN   │
    │ ✓ 知识迁移       │         │ ✓ 推理分离       │         │ ✓ 知识迁移       │
    │ ✗ 推理分离       │         │ ✓ 多频率         │         │ ✓ 推理分离       │
    │ ✗ 多频率         │         │ ✗ 语义层         │         │ ✓ 多频率         │
    │ ✗ 无害干扰       │         │ ✓ 无害干扰       │         │ ✓ 无害干扰       │
    └─────────────────┘         └─────────────────┘         └─────────────────┘
```

### 4.2 四个递进目标

**目标 1：构建 Expert 分离架构。** 从 FastWAM（已有 Video + Action Expert 分离 + KV cache 基础设施）出发，演化到三 Expert。渐进策略：先加轻量 LLM（冻结 T5/CLIP），目标 1-3 验证后，如果 F5 成立再升级完整 AR LLM。LLM 升级是一个 gate decision。

**目标 2：全模态能力不丢失。** 继承 COSMOS 3 的全部功能——Policy（LLM 理解指令 + Video/Action 联合去噪）、Forward Dynamics（首帧 + Action → 未来视频）、Inverse Dynamics（视频 → Action）、Reasoner（LLM 独立推理）——每种模式只用到需要的 Expert，无关 Expert 不参与，消除有害干扰。

**目标 3：多频率执行。** 三 Expert 以各自独立频率运行：

```
        多频率执行金字塔

  LLM Expert           ★               ★               ★
  语义理解+任务规划      ↑ ~1 Hz（K_l 缓存复用，任务切换时刷新）

  Video Expert         ★─★─★─★─★─★─★─★─★─★─★─★
  视觉预测+世界模拟       ↑ ~10 Hz（K_v 缓存复用，画面变化时刷新）

  Action Expert       ★★★★★★★★★★★★★★★★★★★★★★★★★★★★
  实时动作执行           ↑ ~50 Hz（每步仅 Action Expert forward + O(1) 读 K_l, K_v）
```

AR LLM 是多频率的关键收益——不是"跳过计算"，而是提供 T5/CLIP 无法实现的任务规划、空间推理、语义泛化。

**目标 4：想象 RL 闭环。** Video Expert = 世界模型（FD rollout），LLM Expert = 语义裁判（Reasoner 判断任务完成），Action Expert = 被训练策略。闭环：LLM 理解任务 → Action 采样 N 条轨迹 → Video rollout → LLM 裁判打分 → GRPO 更新 Action。不需要真机、仿真器、GT action 标注。已有证据：DreamZero 实验 E（纯视频优化 -43.7%），待验证：FastWAM F5（Expert 分离中的强耦合）。

### 4.3 与现有架构的对比

| | COSMOS 3 | FastWAM | π₀ | **三 Expert** |
|---|---|---|---|---|
| **训练知识迁移** | ✓（共享参数） | 未知（F5） | ✓（共享 Attention） | **✓（共享 Attention）** |
| **推理分离** | ✗（全共享，绑一起） | ✓（Video KV cache） | ✓（LLM KV cache） | **✓（LLM + Video KV cache）** |
| **有害干扰** | 有（全共享） | 预期无（F6） | 预期无 | **预期无（Expert 分离）** |
| **多频率执行** | ✗ | ✓（2 Expert） | ✓（2 Expert） | **✓（3 Expert, 语义→视觉→动作）** |
| **RL 闭环** | 基础设施有，无 GRPO | 有 FD，缺语义 reward | 无世界模型 | **完整闭环** |

### 4.4 Expert 分离程度的选择

三 Expert 位于谱的右端。核心论证：COSMOS 3 的训练迁移通过共享 Attention（而非共享参数）保留——不需要全共享。全共享的代价（性能/有害干扰）通过 Expert 分离消除。最优位置 = 共享 Attention + Expert 分离。

---

## 五、目标架构的优势：解决什么问题

### 5.1 继承 COSMOS 3 的训练迁移，避免其代价

COSMOS 3 的全共享架构证明了训练知识迁移的价值（第二节），但付出了四条代价：

| COSMOS 3 的优势（我们继承） | COSMOS 3 的代价（我们解决） |
|---|---|
| AR→Action 语义迁移（共享参数） | **推理性能受限**：所有模态必须一起 forward（无法分离） |
| 物理常识约束 Action 生成 | **有害干扰**：共享权重服务多种噪声分布 → 表示冲突（DreamZero 实验 B 在 COSMOS 3 中同样存在） |
| 视觉理解迁移到 State Representation | **频率无法分离**：LLM/Video/Action 绑在一起执行（不能各自按需运行） |
| Action CoT 加速连续动作生成 | **模态扩展代价高**：新模态需重新适应共享权重 |

**我们的方案**：用**共享 Attention**（而非共享参数）实现训练迁移。AR LLM 和 Video/Action Expert 在 Joint Attention 中联合训练——语义知识通过 Q·K^T 路由传递到 Action，无需共享 Q/K/V/FFN 权重。推理时各 Expert 通过 KV cache 独立执行。

**关键区别**：共享 Attention 保留了 COSMOS 3 的"LLM 和 Action 在同一个交互空间中"的优势（训练迁移的必要条件），但不需要它们共享参数（推理分离的必要条件）。这是全共享架构做不到的——全共享一旦分开了参数，训练迁移就断了。

### 5.2 超越 FastWAM：从双 Expert 到三 Expert

FastWAM 的 Video + Action 双 Expert 架构解决了推理性能和有害干扰问题（第四节），但有一个关键缺失：**没有语义层**。

| FastWAM 的优势（我们继承） | FastWAM 的局限（我们补充） |
|---|---|
| Expert 分离 → 推理弱耦合（已验证） | **无语义推理**：文本仅通过 T5 cross-attention 注入——T5 只能做"相似性编码"，无法做任务规划、条件推理、空间理解 |
| Video KV cache → 多频率执行 | **无世界模型闭环**：有 FD rollout 但缺语义 reward——无法区分"看起来像 wiping"和"真的在 wiping the right spot" |
| DiT-DiT 分离 → 无害干扰（预期） | **无 AR LLM 的 CoT 能力**：不能将语言指令拆解为子任务序列 |

**我们的方案**：加 AR LLM Expert——不仅是为了多频率的语义层，更是为了目标 4（RL 闭环）的语义 reward。没有 AR LLM 的世界模型闭环是"盲的"——只能靠视频重建质量判断好坏，不知道任务是否真的完成了。

### 5.3 超越 π₀：从 VLA 到 WAM

π₀ 的 LLM + Action 双 Expert 架构有语义理解和推理，但缺世界模型：

| π₀ 的优势（我们参考） | π₀ 的局限（我们补充） |
|---|---|
| Block-causal mask → AR + DiT 和谐共存 | **无视频生成**：不能做 rollout 想象 → 无法做 RL 闭环 → 只能做 behavior cloning |
| Prefix KV cache → LLM 1 次 + Action 10 步 | **无 Forward Dynamics**：不知道 action 执行后的未来状态 |
| 已验证的 LLM→Action 训练耦合 | **无法做 Inverse Dynamics**：不能从视频反推动作 |

**我们的方案**：加 Video Expert——让 π₀ 从 VLA（只能模仿）变成 WAM（可以想象）。Video Expert 的 Forward Dynamics 提供世界模型能力，使 RL 闭环成为可能。

### 5.4 独特的想象 RL 闭环：解决机器人数据与训练成本难题

前面三节分别论证了三 Expert 架构相比 COSMOS 3、FastWAM、π₀ 各自补了什么短板。但这不是拼凑——三个 Expert 合在一起，产生了一个**任何双 Expert 架构都不具备的完整能力：在想象中自我训练**。

**机器人学习的核心瓶颈**：

| 瓶颈 | 根源 |
|------|------|
| **数据稀缺** | GT action 标注必须在真机上收集。DROID 历时 18 个月、7 个机构才收集 76K episodes。新场景、新机器人、新任务 → 几乎必然没有标注数据 |
| **真机成本** | 真机运行有物理风险（碰撞损坏）、时间成本（实时执行不可加速）、人力成本（需要人工 reset 和 supervision） |
| **仿真器构建复杂** | 高保真仿真环境需要大量工程工作（物理建模、资产创建、domain randomization），且始终存在 sim-to-real gap |
| **闭环改进困难** | 标准 behavior cloning 只能模仿已有的 action 分布，无法自我改进。模型的上限 ≤ 训练数据的质量上限 |

**三 Expert 架构如何解决**：

```
              想象 RL 闭环

  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  1. LLM Expert: "wipe the countertop"               │
  │     → 语义理解 → 任务分解 → K_l 缓存（1 次）          │
  │                         │                           │
  │                         ▼                           │
  │  2. Action Expert: N 条不同噪声种子                  │
  │     → N 组 (action_chunk, video_pred)               │
  │     → 每步仅 Action Expert forward (~50Hz)           │
  │                         │                           │
  │                         ▼                           │
  │  3. Video Expert: autoregressive rollout            │
  │     → forward dynamics: frame_t + action → frame_{t+1}│
  │     → 生成多步未来视频（世界模型）                    │
  │                         │                           │
  │                         ▼                           │
  │  4. LLM Expert (裁判模式):                          │
  │     观看 rollout 视频 → "任务完成了吗？"              │
  │     → reward ∈ {yes, no} 或连续评分                 │
  │                         │                           │
  │                         ▼                           │
  │  5. GRPO: advantage → ∂L/∂Expert_weights            │
  │     → 改善 Action（通过 Joint Attention 梯度传递）   │
  │     → 回到步骤 1，用改善后的 policy 重新采样         │
  │                                                     │
  └─────────────────────────────────────────────────────┘
```

**为什么能解决瓶颈**：

| 瓶颈 | 现有方案 | 三 Expert 方案 |
|------|---------|---------------|
| **数据稀缺（GT action）** | behavior cloning：必须有 (video, action) 配对 | **不需要 GT action**——reward 来自 LLM 对生成视频的语义判断。我们的实验 E 已验证：纯视频优化即可改善 action 43.7% |
| **真机成本** | 实时运行、有碰撞风险、需人工 reset | **零真机成本**——所有 rollout 在 GPU 潜空间中完成（Video Expert 的 FD 模式），不需要物理执行 |
| **仿真器 gap** | 需手工构建物理环境、域随机化 | **不需要仿真器**——Video Expert 从 DiT 预训练中获得物理直觉（物体运动、遮挡、重力），LLM 从 AR 训练中获得物理常识（穿模不合理、悬空不合理） |
| **无法自我改进** | BC 上限 = 数据质量 | **可以自我改进**——GRPO 的 reward 信号推动 policy 持续优化。模型的上限 = reward 函数的判别能力 + 世界模型的想象力 |
| **新场景迁移** | 需要收集新场景的配对数据 | **只要有视频就可以改善**——只有 video 没有 action 的数据可以通过纯视频 loss（实验 E 已验证）或 GRPO reward 驱动改善 |

**数据与成本的数量级对比**：

| | 传统方案（真机 BC） | 三 Expert 想象 RL |
|---|---|---|
| **收集 1000 个 episode 的 action 标注** | 数周真机运行 | 0（不需要） |
| **生成 1000 个训练样本** | ~数小时真机执行 | ~数小时 GPU 推理（可批量并行） |
| **在新机器人上部署** | 需要收集该机器人的标注数据 | 已有视频即可开始改善 |
| **持续在线学习** | 需要在真机上持续运行 | 不需要真机，可以在服务器上持续训练 |

**一句话**：三 Expert 架构的想象 RL 闭环 = 世界模型（Video）+ 语义裁判（LLM）+ 可训练策略（Action）。FP&A 三要素齐全——不需要真机、不需要仿真器、不需要 GT action。这在现有架构中没有任何一个能做到。

### 5.5 三 Expert 架构的独特价值总结

| 问题 | 现有方案 | 三 Expert 方案 |
|------|---------|---------------|
| 训练时 LLM 知识如何到 Action？ | COSMOS 3：共享参数（但带来代价） | **共享 Attention——保留迁移，避免代价** |
| 推理时如何避免全共享代价？ | FastWAM/π₀：Expert 分离（但缺语义/缺世界模型） | **三 Expert KV cache 分离——同时有语义 + 世界模型** |
| 如何实现想象 RL？ | 无人做到 | **LLM reward + Video rollout + Action GRPO——完整闭环** |
| 如何防止模态间有害干扰？ | 全共享架构中存在（实验 B） | **Expert 独立噪声调度——消除共享权重的冲突** |
| 如何实现多频率执行？ | FastWAM：2 层 | **3 层——语义(~1Hz)→视觉(~10Hz)→动作(~50Hz)** |

**一句话总结**：三 Expert 架构 = COSMOS 3 的训练迁移 + FastWAM 的推理效率 + π₀ 的语义理解 + 前两者都没有的世界模型 RL 闭环能力。它不是三个架构的简单拼接——共享 Attention 作为统一的跨模态路由机制，是这一切得以同时成立的前提。

---

---

## 六、研究路线图

**整个研究方向的本质是两个 P0 技术突破**：

| P0 突破 | 平台 | 核心问题 | 决定了 |
|---------|------|---------|--------|
| **突破 1** | FastWAM + AR LLM | Expert 分离架构中加入 AR LLM 是否带来真正的语义优势？ | 三 Expert 架构是否值得（vs 双 Expert + 外部 LLM） |
| **突破 2** | COSMOS 3 GRPO | 能否在 WAM 上打通想象 RL 闭环？ | 想象训练是否工程可行 |

其余实验（F1-F4 多频率、F6 有害干扰、P3b Reasoner 评测）都是支撑性的——它们提供证据和消融，但不改变方向的可行性。只有这两个突破是 gate decision。

### 6.1 平台分工

四种架构在路线图中各司其职：

| 平台 | 谱位置 | 角色 | 关键实验 | 决定了 |
|------|--------|------|---------|--------|
| **DreamZero** | 最左（全共享） | 全共享属性基线 | 7 组因果实验（✅ 已完成） | 耦合机制发现 + 实验方法论 |
| **FastWAM** | 偏右（DiT-DiT 分离） | **突破 1 平台** | F7, F8（语义）, F5（梯度耦合 P0）, F1-F4（多频率支撑） | 三 Expert 架构的价值 |
| **COSMOS 3** | 最左（全共享双模） | **突破 2 平台** | P3a（GRPO）, P3b（Reasoner）, P3c（vs SFT） | 想象 RL 的可行性 |
| **π₀** | 最右（LLM-DiT 分离） | 架构参考 | Block-causal mask, prefix KV cache | 三 Expert 的语义层设计参考 |

### 6.2 先导实验详细设计（Phase 2）

#### 6.2.1 FastWAM 先导实验（4-6 周）

FastWAM 已有 Expert 分离 + video KV cache + action 独立去噪——不需新架构即可直接验证。

**计算多频率验证（F1-F4）**：

| 实验 | 方法 | 验证问题 | 成功标准 |
|------|------|---------|---------|
| **F1 ★★★** | 分离推理 vs Full joint → Action MSE | 多频率精度是否等同于联合推理？ | Action MSE 差异 < 3% |
| **F2 ★★★** | 固定 video K/V → 连续 50 步 Action → Action MSE 曲线 | K/V 缓存的有效期？ | 曲线不平滑上升（无累积退化） |
| **F3 ★★** | 每 5 步 vs 每 10 步 vs 从不刷新 video K/V | 最优刷新策略？ | 输出最优刷新频率 |
| **F4 ★★** | 相同 DROID 任务，FastWAM 分离 vs π₀ prefix cache | DiT-DiT vs LLM-DiT 多频率效率 | Action MSE + 延迟对比 |

**耦合属性验证（F5-F6）**：

| 实验 | 方法 | 验证问题 | 重要性 |
|------|------|---------|--------|
| **F5 ★★★** | 仅 backward video loss，更新 LoRA 一步 → Action 变化（平行 DreamZero 实验 D） | **DiT-DiT Expert 分离中强耦合是否成立？** | **P0——决定目标 4（RL 闭环）的可行性** |
| **F6 ★★** | video latent 替换为 random/zero → Action 变化（平行 DreamZero 实验 B） | Expert 分离中是否存在有害干扰？ | 验证有害干扰假说 |

**语义多频率验证（F7-F8）**：

| 实验 | 方法 | 验证问题 | 重要性 |
|------|------|---------|--------|
| **F7 ★★★** | FastWAM + 冻结 AR LLM（Gemma 2B）→ 三 Expert block-causal 推理（LLM 1 次 → Video 1 次 → Action 30 步） | 三层多频率是否可行？精度 vs Full joint？ | 三 Expert 架构的最小可行验证 |
| **F8 ★★★** | 共享 Attention 训练 vs LLM 冻结仅推理注入 vs T5 基线 → Action 精度 | AR LLM 的训练迁移是否优于独立训练 + 文本注入？ | 决定 AR LLM 是否需要参与训练（vs 仅推理注入） |

#### 6.2.2 COSMOS 3 先导实验（6-8 周，可与 FastWAM 并行）

COSMOS 3 是目前唯一具备完整闭环基础设施的平台（Policy + Forward Dynamics autoregressive rollout + Reasoner VLM）。

| 实验 | 方法 | 验证问题 | 成功标准 |
|------|------|---------|---------|
| **P3a ★★★** | 在 COSMOS 3 DROID Policy 上实现最小 GRPO 循环：N=4 采样，Reasoner reward（"Did the robot successfully [task]?"），GRPO loss → LoRA 更新，100 episodes，200 步 | GRPO 在 WAM 上的工程可行性？训练稳定性？ | Loss 收敛，无 NaN/梯度爆炸 |
| **P3b ★★★** | 100 个 DROID rollout 视频 → Reasoner 判断"任务完成" → 与 GT action 的 success label 对比 | Reasoner 做机器人任务成功检测的准确率？ | 准确率 > 70%（如低于，需探索替代 reward 方案） |
| **P3c ★★** | 200 步 GRPO vs 200 步 FM SFT → Action MSE | 全共享中 GRPO 能否超越 SFT？ | GRPO Action MSE ≤ SFT（平行确认 DreamZero 实验 E） |

#### 6.2.3 平台间依赖关系

```
FastWAM F1-F4 ──→ 目标 3（多频率）验证 ──┐
                                          ├──→ Phase 3 三 Expert 原型
FastWAM F5 ──→ 目标 4（RL 闭环）可行性 ──┤
                                          │
COSMOS 3 P3a-c ──→ RL 工程可行性 ────────┘

F1-F4 不依赖 F5（只依赖弱耦合，已确认成立）
F5 和 P3a 可以并行推进（不互依赖）
Phase 3 需要 F5 成立（否则 Expert 分离中的训练迁移无法保证）
Phase 4 需要 F5 + P3a 均成立
```

### 6.3 完整时间线

```
Phase 1（已完成）✅
  DreamZero 7 组因果实验 → 推理弱耦合 + 训练强耦合 + 有害干扰假说
  四个架构代码级分析 → 连续谱 + 全共享代价
  Attention 路由范式识别 → 理论框架建立

Phase 2（4-8 周，两个突破并行推进）:

  突破 1 - FastWAM + AR LLM（4-6 周）:
    Week 1-2: F1-F4（计算多频率支撑实验）
    Week 3-4: F5-F6（耦合属性 + 有害干扰）
    Week 4-6: F7-F8（语义多频率——突破 1 的核心）

  突破 2 - COSMOS 3 GRPO（6-8 周）:
    Week 1-3: P3a（GRPO 最小循环搭建——突破 2 的核心）
    Week 3-5: P3b（Reasoner reward 评测）
    Week 5-8: P3c（GRPO vs SFT 对比）

Phase 3（取决于突破 1，6-8 周）: 三 Expert 原型
  若突破 1 成立（AR LLM 有语义优势）:
    目标 1: FastWAM + AR LLM → 三 Expert 架构搭建
    目标 2: 四种模式验证（Policy/FD/ID/Reasoner）
    目标 3: 多频率执行精度 + 延迟评测
  若突破 1 不成立:
    直接进入 Phase 4 论文撰写

Phase 4（取决于两个突破的结果）: 想象 RL 原型 + 论文
  双突破均成立: COSMOS 3 GRPO + 三 Expert → 完整目标 4 验证 → 第一档论文
  仅突破 1 成立: 架构论证 → 第二档论文
  仅突破 2 成立: 训练方法论证 → 第三档论文
  均不成立: 属性论证 → 第四档论文
```

### 6.4 论文策略

论文定位取决于两个突破的结果：

```
突破 1 (FastWAM + AR LLM)     突破 2 (COSMOS 3 GRPO)      论文定位
─────────────────────────────────────────────────────────────────────
       ✓ 成立                       ✓ 成立                完整版：范式 + 架构 + 想象 RL
       ✓ 成立                       ✗ 不成立              架构版：范式 + 架构 + RL 作为 future work
       ✗ 不成立                     ✓ 成立                发现版：范式 + 想象 RL（在全共享中）
       ✗ 不成立                     ✗ 不成立              属性版：范式识别 + 耦合属性 + Expert 分离优势
```

**第一档（双突破均成立）**：完整论证
- Attention 路由范式识别 + 推理弱耦合的跨平台验证
- 三 Expert 分离架构：继承 COSMOS 3 迁移 + 实现 FastWAM 效率 + 加入 π₀ 语义层
- AR LLM 语义优势的消融验证（F8）
- 想象 RL 闭环验证（P3a-c + Phase 4）
- **贡献级别**：范式识别 + 架构创新 + 训练方法创新

**第二档（仅突破 1 成立）**：架构论证
- Attention 路由范式识别 + 耦合属性
- 三 Expert 架构 + AR LLM 语义优势 + 多频率验证
- 想象 RL 作为 future work
- **贡献级别**：范式识别 + 架构创新

**第三档（仅突破 2 成立）**：训练方法论证
- Attention 路由范式识别
- 全共享架构中的想象 RL 闭环验证
- Expert 分离作为未来架构方向
- **贡献级别**：范式识别 + 训练方法创新

**第四档（均不成立）**：属性论证
- Attention 路由范式识别
- 推理弱耦合 + Expert 分离优势（有害干扰假说）
- **贡献级别**：范式识别 + 属性发现

---

## 七、总结

> COSMOS 3 证明了训练应该在一起。FastWAM 证明了推理应该分开。DreamZero 实验揭示了 Attention 路由的耦合属性。三者共同指向目标架构：基于 Attention 路由的全模态多频率世界模型。下一步需要突破两个关键技术：(1) FastWAM 加入 AR LLM 是否带来真正的语义优势？(2) COSMOS 3 能否打通想象 RL 闭环？这两个问题的答案决定了整个方向的可行性。
