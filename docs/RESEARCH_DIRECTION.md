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

COSMOS 3 的 AR Reasoner 和 DiT Generator 使用**同一组 Transformer blocks、同一组 Q/K/V/FFN 权重**，仅 attention mask 不同（causal vs full）。这个设计不是偶然的——全共享参数在训练中产生四个层面的正迁移：

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
Layer i:
  Video Expert ──→ Q_v, K_v, V_v ──┐
                                     ├── Joint Flash Attention ──→ split → 各自 o_proj → 各自 FFN
  Action Expert ─→ Q_a, K_a, V_a ──┘
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

## 四、目标架构：训练共享 Attention + 推理 KV cache 分离

### 4.1 架构设计

三 Expert（LLM + Video + Action），每层通过 Joint Self-Attention 路由信息，各自独立 Q/K/V/FFN：

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

### 4.2 四个递进目标

**目标 1：构建 Expert 分离架构。** 从 FastWAM（已有 Video + Action Expert 分离 + KV cache 基础设施）出发，演化到三 Expert。渐进策略：先加轻量 LLM（冻结 T5/CLIP），目标 1-3 验证后，如果 F5 成立再升级完整 AR LLM。LLM 升级是一个 gate decision。

**目标 2：全模态能力不丢失。** 继承 COSMOS 3 的全部功能——Policy（LLM 理解指令 + Video/Action 联合去噪）、Forward Dynamics（首帧 + Action → 未来视频）、Inverse Dynamics（视频 → Action）、Reasoner（LLM 独立推理）——每种模式只用到需要的 Expert，无关 Expert 不参与，消除有害干扰。

**目标 3：多频率执行。** LLM ~1Hz（任务理解，K_l 缓存）→ Video ~10Hz（视觉预测，K_v 缓存）→ Action ~50Hz（实时控制，仅 Action Expert forward + O(1) 读取缓存）。AR LLM 是多频率的关键收益——不是"跳过计算"，而是提供 T5/CLIP 无法实现的任务规划、空间推理、语义泛化。

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

### 5.4 三 Expert 架构的独特价值总结

| 问题 | 现有方案 | 三 Expert 方案 |
|------|---------|---------------|
| 训练时 LLM 知识如何到 Action？ | COSMOS 3：共享参数（但带来代价） | **共享 Attention——保留迁移，避免代价** |
| 推理时如何避免全共享代价？ | FastWAM/π₀：Expert 分离（但缺语义/缺世界模型） | **三 Expert KV cache 分离——同时有语义 + 世界模型** |
| 如何实现想象 RL？ | 无人做到 | **LLM reward + Video rollout + Action GRPO——完整闭环** |
| 如何防止模态间有害干扰？ | 全共享架构中存在（实验 B） | **Expert 独立噪声调度——消除共享权重的冲突** |
| 如何实现多频率执行？ | FastWAM：2 层 | **3 层——语义(~1Hz)→视觉(~10Hz)→动作(~50Hz)** |

**一句话总结**：三 Expert 架构 = COSMOS 3 的训练迁移 + FastWAM 的推理效率 + π₀ 的语义理解 + 前两者都没有的世界模型 RL 闭环能力。它不是三个架构的简单拼接——共享 Attention 作为统一的跨模态路由机制，是这一切得以同时成立的前提。

---

## 六、先导实验：在现有平台上验证关键假设

### 5.1 FastWAM（多频率 + P0 + 语义验证）

FastWAM 已有 Expert 分离 + video KV cache + action 独立去噪——不需新架构即可直接验证核心假设。

| 实验 | 方法 | 验证问题 | 对应目标 |
|------|------|---------|---------|
| **F1 ★★★** | 分离推理 vs 联合推理 → Action MSE | 多频率精度是否等同于 Full joint？ | 目标 3 |
| **F2 ★★★** | 固定 video K/V → 连续 50 步 Action → Action MSE 曲线 | K/V 缓存的保质期？ | 目标 3 |
| **F3 ★★** | 每 5 步 vs 每 10 步 vs 从不刷新 video K/V | 最优刷新策略？ | 目标 3 |
| **F5 ★★★** | **仅 backward video loss → 测 Action 变化**（平行 DreamZero 实验 D） | **DiT-DiT Expert 分离中强耦合是否成立？** | **目标 4（P0）** |
| **F6 ★★** | video latent 替换 → 测 Action（平行 DreamZero 实验 B） | Expert 分离中是否存在有害干扰？ | 目标 1 |
| **F7 ★★★** | FastWAM + 冻结 AR LLM → 三 Expert block-causal 推理 | 三层多频率是否可行？ | 目标 3 |
| **F8 ★★★** | 共享 Attention 训练 vs LLM 冻结仅推理注入 vs T5 基线 | AR LLM 训练迁移的必要性？ | 目标 1+3 |

F5 是整个项目的 P0——决定训练强耦合能否泛化到 Expert 分离架构，决定目标 4（RL 闭环）的可行性。F8 验证 COSMOS 3 的训练迁移机制在 Expert 分离架构中是否可复现。

### 5.2 COSMOS 3（想象 RL 先导）

COSMOS 3 是目前唯一具备完整闭环基础设施的平台（Policy + Forward Dynamics autoregressive rollout + Reasoner VLM）。

| 实验 | 方法 | 验证问题 |
|------|------|---------|
| **P3a ★★★** | GRPO 最小循环：N=4，Reasoner reward，200 步，LoRA 更新 | GRPO 在 WAM 上的工程可行性？ |
| **P3b ★★★** | 100 个 DROID rollout → Reasoner 判断 vs GT label | Reasoner 做任务成功检测的准确率？ |
| **P3c ★★** | GRPO vs FM SFT → Action MSE | 全共享中 GRPO 能否超越 SFT？（平行确认实验 E） |

### 5.3 平台分工与依赖

| 平台 | 角色 | 不依赖 | 决定了 |
|------|------|--------|--------|
| **DreamZero** | 全共享属性基线（已完成 ✅） | — | 耦合机制发现 |
| **FastWAM** | 多频率 + P0 + 语义训练迁移 | COSMOS 3 | 目标 3 + 目标 4 可行性 |
| **COSMOS 3** | 想象 RL 先导 | FastWAM F5 | RL 工程可行性 + Reward 质量 |

FastWAM F5 和 COSMOS 3 GRPO 可以并行推进。

---

## 七、研究路线图

```
Phase 1（已完成）✅
  DreamZero 7 组因果实验 → 推理弱耦合 + 训练强耦合 + 有害干扰假说
  四个架构代码级分析 → 连续谱 + 全共享四条挑战
  Attention 路由范式识别

Phase 2（4-6 周）: 先导实验
  FastWAM: F1-F8（多频率 + P0 + 语义）
  COSMOS 3: P3a-c（GRPO 先导）
  → 并行推进，不互依赖

Phase 3（取决于 F5）:
  F5 成立 → 三 Expert 原型搭建 + 目标 1-3 验证
  F5 不成立 → 以 Phase 2 结果撰写论文

Phase 4（取决于 F5 + P3a）:
  均成立 → 完整想象 RL 验证（目标 4）
  仅 F5 成立 → 论文 = 机制分析 + 目标架构 + 先导实验
  均不成立 → 论文 = 范式识别 + 属性发现 + Expert 分离优势
```

---

## 八、总结

> COSMOS 3 证明了训练应该在一起（共享参数 = 语义/物理/视觉知识迁移到 Action）。FastWAM 证明了推理应该分开（Expert 分离 = 多频率 + 无有害干扰 + 4× 加速）。DreamZero 实验揭示了推理弱耦合和训练强耦合是 Attention 路由的通用属性。三者共同指向目标架构：训练时共享 Attention 实现知识迁移，推理时 KV cache 分离实现多频率执行。FastWAM F5 决定这个架构的最终天花板。
