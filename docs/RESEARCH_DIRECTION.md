# 机器人世界模型架构探索：潜空间对齐与 Attention 路由

> 基于对 JEPA、DreamZero、FastWAM、COSMOS 3、π₀ 的分析
> 2026-08-11

---

## 摘要

未来机器人世界模型的架构存在三种方向：

1. **JEPA 路线**：通过 Encoder 将不同模态压缩到统一隐空间，在隐空间中进行预测。这是**潜空间对齐**。
2. **WAM 路线**（DreamZero / FastWAM / COSMOS 3）：Video Diffusion + Action，在 Attention 层进行跨模态信息路由。
3. **VLA 路线**（π₀）：LLM/VLM + Action Expert，同样在 Attention 层进行跨模态信息路由。

方向 2 和方向 3 共享同一个底层机制——**Attention 层跨模态路由**——即不同模态保留独立的 Expert（各自的 Q/K/V/FFN），仅在 Attention 层通过 Q·K^T 内积交换信息。这与 JEPA 的"潜空间对齐"是两种正交的跨模态交互范式。

本文聚焦 Attention 路由方向，探讨两个核心维度：（1）架构维度——全共享到全分离的连续谱；（2）主干维度——VLM、DiT、还是 VLM+DiT 双模。基于 DreamZero 上 7 组因果实验揭示的 Attention 路由关键属性（推理弱耦合、当共享参数时训练强耦合），我们探讨未来架构的可能形态。

---

## 一、机器人世界模型的三种架构方向

### 1.1 JEPA：潜空间对齐

```
JEPA (Joint Embedding Predictive Architecture):

  observation ──→ Encoder ──→ latent_z ──→ Predictor ──→ latent_{z+1}  ←── Encoder ←── future_obs
  (当前观测)                (统一隐空间)   (隐空间预测)    (预测的未来隐表示)

  核心思想：在 Encoder 输出的隐空间中进行预测，而非在像素空间重建
  代表工作：I-JEPA、V-JEPA
```

- **对齐方式**：Encoder 将不同模态/时刻的输入压缩到同一个向量空间
- **交互机制**：Predictor 在统一空间中做状态转移（通常是小网络）
- **优势**：隐空间预测避免像素重建，计算高效
- **局限**：Encoder 承受全部对齐压力，统一空间中的表示彼此交织，推理无法分离

### 1.2 WAM 路线：Video + Action 联合去噪

```
  Video Token ──→ Video Expert (DiT) ──→ Q_v, K_v, V_v ──┐
                                                            ├── Joint Attention ──→ 各自 FFN
  Action Token ─→ Action Expert (DiT) ─→ Q_a, K_a, V_a ──┘

  代表：DreamZero（全共享 DiT）→ FastWAM（双 Expert MoT）→ COSMOS 3（AR+DiT 双模）
  训练：Video + Action 联合 Flow Matching
  推理：生成未来视频 + 预测动作（可分离程度逐步增强）
```

### 1.3 VLA 路线：LLM + Action 联合推理

```
  Image + Lang Token ──→ LLM Expert (Gemma 2B) ──→ Q_l, K_l, V_l ──┐
                                                                       ├── Joint Attention ──→ 各自 FFN
  State + Action Token ─→ Action Expert (300M) ──→ Q_a, K_a, V_a ──┘

  代表：π₀ / π₀-FAST / π₀.₅
  训练：Action Flow Matching（给定图像+指令+状态，预测动作 chunk）
  推理：LLM prefix 一次 forward → 缓存 K/V → Action 10 步独立去噪
```

### 1.4 WAM 与 VLA 的共同本质

**两者都在 Attention 层进行跨模态信息路由，与 JEPA 的潜空间对齐是两种正交的交互范式。**

| | WAM 路线 | VLA 路线 | JEPA 路线 |
|---|---|---|---|
| **核心模态** | 视频 + 动作 | 语言/图像 + 动作 | 观测 → 隐状态 |
| **主干类型** | DiT（扩散模型） | LLM（自回归） | Encoder（压缩） |
| **交互机制** | Q·K^T 路由 | Q·K^T 路由 | Encoder 压缩 |
| **推理分离** | 可（逐步增强） | 可（KV cache） | 不可 |

### 1.5 两种范式的本质对照

| | 潜空间对齐（JEPA） | Attention 路由（WAM / VLA） |
|---|---|---|
| **交互方式** | Encoder 压缩到统一空间 | Expert 独立编码 + Q·K^T 无参数路由 |
| **表示空间** | 统一（压缩后共享） | 独立（各模态保留专属空间） |
| **信息损失** | 有（压缩到统一维度） | 无 |
| **推理分离** | 不可（表示交织） | **可以**（Attention 结束即分离） |
| **预训练利用** | 难（Encoder 随机初始化） | 易（LLM/DiT 各自预训练） |

**本文的核心问题**：如果走 Attention 路由路线，未来架构应该是什么形态？

---

## 二、Attention 路由的两个架构维度

### 2.1 维度一：Expert 分离程度（全共享 ↔ 全分离）

四个架构并非同质的"Attention 路由"，而是分布在一条连续谱上：

```
全共享 ←──────────────────────────────────────────────→ 全分离

COSMOS 3          DreamZero        FastWAM            π₀
│                 │                 │                  │
同一组 blocks      全共享 DiT        独立 Q/K/V/FFN     独立 Q/K/V/FFN
AR 和 DiT         仅 token 类型区分  2 Expert MoT       2 Expert LLM
共享 Q/K/V/FFN     video+action     30 层              18 层
仅 attention mask  在同一 DiT 内     仅 head_dim 对齐    仅 head_dim 对齐
不同 (causal vs full)
│                 │                 │                  │
最左端             偏左              偏右                最右端
```

**COSMOS 3 是最左端——全共享。** AR Reasoner 和 DiT Generator 使用**同一组 Transformer blocks、同一组 Q/K/V/FFN 权重**。区别仅在 attention mask（causal vs full）。这与 DreamZero 的本质相同——都是共享参数——只是共享的粒度不同：COSMOS 3 是不同 mode 共享同一组 blocks，DreamZero 是 video 和 action token 共享同一组 blocks。

### 2.3 全共享架构面临的挑战

全共享架构（COSMOS 3、DreamZero）虽然参数效率高、训练耦合强，但存在四个系统性问题：

**1. 推理性能受限。** 所有模态的 token 必须一起通过所有 Transformer 层。即使只需要 Action 输出，Video token 也必须完成全部 forward 计算。DreamZero 的 AO 模式（跳过视频去噪）仍需要 Video token 在 16 步中去噪 16 次 Attention——这些计算没有产生任何有用的视频输出，但一个也不能省。

**2. 模态间有害干扰。** 共享的 Q/K/V/FFN 权重需要同时服务不同模态的表示需求。我们的实验 B 证明：在 DreamZero 中，Video 和 Action 使用不同的噪声分布（`Beta(3,1)` vs `Uniform`），共享权重被两种冲突的去噪目标拉扯，导致 Video token 在 Attention 中对 Action token 产生有害表示偏移。替换 video latent 为纯噪声后 Action 反而改善 13%——这正是有害干扰被消除的证据。

**3. 频率无法分离。** Video 去噪（16-30 步）和 Action 去噪（通常相同的步数）被绑在一起执行。无法实现 LLM ~1Hz / Video ~10Hz / Action ~50Hz 的多频率调度。每次需要新的 Action 输出，都必须跑完整的 Video + Action 联合去噪。

**4. 模态扩展代价高。** 加一个新模态意味着共享权重需要重新适应新的 token 分布和噪声调度。新的模态可能与已有模态的表示需求冲突，进一步加剧有害干扰。

这四条挑战解释了为什么 FastWAM 和 π₀ 选择了 Expert 分离路线——它们本质上是对全共享架构的问题的回应。但代价是训练耦合的强度未知（P0）。

### 2.4 谱的架构含义

### 2.2 维度二：主干类型（VLM / DiT / 双模）

```
               VLM 主干                     DiT 主干
          ┌─────────────────┐        ┌─────────────────┐
          │ π₀              │        │ DreamZero       │
          │ LLM(AR) + Action │        │ Video(Diff)     │
          │ 语义理解 + Action │        │ 视觉预测 + Action │
          └────────┬─────────┘        └────────┬─────────┘
                   │                           │
                   └───────────┬───────────────┘
                               │
                     ┌─────────▼──────────┐
                     │ COSMOS 3           │
                     │ AR + DiT 双模      │
                     │ 语义 + 视觉 + Action│
                     └────────────────────┘
```

| | VLM (π₀) | DiT (DreamZero / FastWAM) | 双模 (COSMOS 3) |
|---|---|---|---|
| **语义理解** | ✓（LLM 原生） | ✗（仅 T5 文本编码） | ✓（Reasoner） |
| **视觉预测** | ✗（不生成视频） | ✓（Video DiT） | ✓（Generator） |
| **推理效率** | 高（LLM 1次 + Action 10步） | 低（Video 16步 + Action 16步） | 灵活（按模式按需执行） |
| **架构复杂度** | 低 | 低 | 高 |

**架构演进的主线**：从单一主干（VLM 或 DiT）走向双模统一（COSMOS 3），同时 Expert 分离程度从全共享（DreamZero）走向更灵活的分离（FastWAM/π₀→COSMOS 3）。**COSMOS 3 是目前唯一同时跨越两个维度的架构。**

---

## 三、未来的架构方向：FastWAM、π₀、还是新的综合？

我们已经认识到 Expert 分离优于全共享（四个挑战），且全共享架构（COSMOS 3、DreamZero）和 Expert 分离架构（FastWAM、π₀）都位于谱上。那么下一步架构应该走向哪里？

### 3.1 FastWAM 方案：DiT + Action Expert（世界模型优先）

```
Video Expert (DiT 30层) ──→ Q_v, K_v, V_v ──┐
                                               ├── Joint Attention ──→ Video 输出 + Action 输出
Action Expert (DiT 30层) ─→ Q_a, K_a, V_a ──┘

优势：可以生成未来视频（世界模型），支持 forward dynamics + policy + inverse dynamics
      已验证推理加速（video KV cache → action 独立去噪，4× speedup）
局限：没有语言推理能力，文本仅通过 T5 cross-attention 注入（弱语义理解）
```

### 3.2 π₀ 方案：LLM + Action Expert（语义理解优先）

```
LLM Expert (Gemma 2B 18层) ──→ Q_l, K_l, V_l ──┐
                                                  ├── Joint Attention ──→ Action 输出
Action Expert (300M 18层) ───→ Q_a, K_a, V_a ──┘

优势：原生语言推理（任务规划、常识判断），已验证 prefix KV cache + action 独立去噪
局限：不生成视频（无法做世界模型式的 rollout 想象），模型较小（2B+300M vs DiT 的 5B-14B）
```

### 3.3 两方案的本质差异：能力边界 vs 语义深度

FastWAM 和 π₀ 选择了不同的"缺失能力"作为代价：

| | FastWAM | π₀ |
|---|---|---|
| **有什么** | 视频生成（世界模型） | 语言推理（任务理解） |
| **缺什么** | 语义推理 | 视觉未来预测 |
| **优势场景** | 需要 rollout 想象的环境 | 需要语言理解的任务 |
| **适合 RL** | ✓（有世界模型，可以 rollout） | ✗（无视频生成，只能 BC） |
| **适合推理加速** | ✓ | ✓ |

### 3.4 第三种方案：三 Expert 分离 + 双模统一

既然全共享架构可以同时容纳 LLM 和 DiT（COSMOS 3 已证明），Expert 分离架构也应该可以。将 π₀ 的 LLM Expert 和 FastWAM 的 Video Expert 组合：

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
```

### 3.5 优势一：多频率分离执行

三 Expert 的推理弱耦合（四个架构已验证）直接支持按各自需求独立执行：

```
语义层（LLM Expert, AR/causal）：    ★               ★               ★
  "任务是什么？如何完成？"              ↑ ~1 Hz（任务切换时重新推理）
  K_l/V_l 缓存，有效期内不变            │
                                       │
视觉层（Video Expert, DiT/diff）：    ★─★─★─★─★─★─★─★─★─★─★─★
  "动作执行后画面会变成什么样？"         ↑ ~10 Hz（N 个 action chunk 更新一次）
  K_v/V_v 缓存，视频上下文变化时刷新     │
                                       │
动作层（Action Expert, DiT/diff）：   ★★★★★★★★★★★★★★★★★★★★★★★★★★★★
  "具体关节位置是什么？"                ↑ ~50 Hz（每步仅 forward Action Expert）
  读 (K_l, V_l) + (K_v, V_v)，O(1)
```

每步 Action 执行只需要 Action Expert 的 forward + O(1) 读取 LLM/Video K/V 缓存。LLM 和 Video Expert 的 FFN 不需要每次重新计算。这直接解决了全共享架构的"性能受限"挑战。

### 3.6 优势二：完整的功能覆盖

三 Expert 方案继承 COSMOS 3 的全部模式，且每种模式都可以更高效地执行：

| COSMOS 3 模式 | 三 Expert 方案 | 改进 |
|---|---|---|
| **Policy**（首帧+指令→video+action） | LLM 理解指令 + Video/Action 联合去噪 | Expert 分离，无有害干扰 |
| **Forward Dynamics**（首帧+action→video） | Action Expert + Video Expert 联合去噪 | Video 独立生成，不受 Action 噪声干扰 |
| **Inverse Dynamics**（video→action） | Video Expert 编码 + Action Expert 去噪 | 两者的 noise schedule 独立 |
| **Reasoner**（text+image→text） | LLM Expert 独立推理 | 不需要 Generator 参与 |

COSMOS 3 能做的，三 Expert 方案都能做——而且每个模式只用到需要的 Expert，不受无关模态的噪声干扰。

### 3.7 优势三：RL 闭环——在想象中学习

这是三 Expert 方案最独特的价值。三个 Expert 各自扮演 Dreamer V3 框架中的角色：

```
Dreamer V3 组件          三 Expert 对应

World Model        →    Video Expert（forward dynamics: frame_t + action_t → frame_{t+1}）
Actor / Policy     →    Action Expert（给定 context + noise → action_chunk）
Critic / Reward    →    LLM Expert（观看生成的视频 → "任务完成了吗？" → reward）
```

闭环流程：

```
1. LLM: "wipe the countertop, starting from top-left" → 任务分解 → K_l 缓存（1次）
2. Action: N 条不同噪声种子的去噪轨迹 → N 组 (action_chunk, video_pred)（每步仅 Action Expert）
3. Video: autoregressive rollout → frame_{t+H}（Video Expert 充当世界模型）
4. LLM 裁判: 观看 rollout 视频 → "Did the robot successfully wipe?" → reward
5. GRPO: advantage → ∂L/∂Expert_weights → 改善 Action（通过 Joint Attention 梯度传递）
```

**不需要真机，不需要仿真器，不需要 GT action 标注。** 模型在自身的"想象"中学习。我们已经在 DreamZero 上验证了核心机制——纯视频优化 100 步后 action 改善 43.7%（实验 E）。三 Expert 方案将此机制推广到多步 rollout 和语义级 reward。

### 3.8 对比四种方案

| | COSMOS 3（全共享双模） | FastWAM（DiT 双 Expert） | π₀（LLM 双 Expert） | **三 Expert 分离双模** |
|---|---|---|---|---|
| **语义理解** | ✓（Reasoner） | ✗ | ✓（LLM） | ✓（LLM Expert） |
| **视频生成** | ✓（Generator） | ✓（Video Expert） | ✗ | ✓（Video Expert） |
| **多频率执行** | ✗（全共享，绑一起） | ✓（video KV cache） | ✓（prefix KV cache） | ✓（三类频率独立） |
| **有害干扰** | 可能有（全共享） | 预期无（分离） | 预期无（分离） | **预期无（分离）** |
| **RL 闭环** | 基础设施完备（FD rollout + Reasoner），但无 GRPO | 有 FD rollout，无语义 reward | 无语义 reward 来源 | **完整：FD rollout + Reasoner + GRPO** |
| **训练耦合** | 有（共享参数） | 未知（P0 待验证） | 未知（P0 待验证） | 未知（P0 待验证） |

**核心风险**：三 Expert 的训练资源需求显著高于双 Expert 方案。务实起点：从 π₀（LLM + Action, 已开源, 有 PyTorch port）出发，加 Video Expert（LoRA fine-tune，冻结 LLM 和 Video 主干）。

---

## 四、实验发现：Attention 路由的耦合属性

### 3.1 推理弱耦合（已验证）

| 实验 | 结论 |
|------|------|
| flow_pred 注入 σ=0~1.0 噪声 | Action 精度**完全不受影响** |
| video latent 替换为纯噪声/全零 | Action **反而改善 13%**（有害干扰被消除） |
| 第一帧换 episode/旋转/遮挡 | 对 Action **0% 影响** |

**结论**：视频输出质量与动作精度无因果关联。这已在 DreamZero 上严格验证，且 Expert 分离架构（FastWAM/π₀）中的主动 KV cache 分离提供了更强的佐证。

### 3.2 训练强耦合（在共享参数架构中已验证，泛化性待验证）

| 实验 | 结论 |
|------|------|
| 单步梯度：仅 backward video_loss | Action **改变 58.8%** |
| 100 步纯视频优化 | Action **改善 43.7%** |

**结论**：Video loss 梯度通过**共享权重**传播到 action。但在 Expert 分离架构中，梯度路径完全不同——仅通过 Q·K^T 中的跨模态项——强度未知。**这是最高优先级的待验证假设。**

### 3.3 实验 B 的反直觉结果：有害干扰假说

全共享架构中 random latent → action 改善 13%。代码级解释：共享 Q/K/V 权重需要同时服务 Video 的 `Beta(3,1)` 噪声分布和 Action 的 `Uniform` 分布 → 两种去噪动态相互冲突 → video 对 action 产生有害表示偏移。

**如果此假说成立，Expert 分离不是可选的优化，而是防止模态间有害干扰的必要设计。** 这需要在 Expert 分离架构中验证（P2）。

---

## 五、未来架构推测

基于已有证据，Attention 路由路线下的最优架构应具备：

**1. Expert 分离而非全共享**：全共享架构的四条挑战——推理性能、有害干扰、频率分离、模态扩展——在 Expert 分离架构中全部可以得到缓解。FastWAM 和 π₀ 已经验证了推理性能（4× 加速）和频率分离（KV cache）。如果 P2 进一步验证了无害干扰，则 Expert 分离在四个维度上都优于全共享。**全共享架构的唯一潜在优势——训练强耦合——在 Expert 分离中的强度待 P0 验证。**

**2. 双模统一 + Expert 分离的组合**：COSMOS 3 证明了双模统一可以在全共享架构中实现。但实现它的同时承受了全共享的四条代价。未来的问题是：能否在 Expert 分离架构中实现双模统一？即 AR Reasoner 和 DiT Generator 各自独立的 Expert，通过 Joint Attention 交互，同时避免全共享的代价。

**3. 多频率执行**：利用推理弱耦合，LLM ~1Hz、Video ~10Hz、Action ~50Hz。FastWAM 的 video KV cache 和 π₀ 的 prefix KV cache 已验证可行。

**4. COSMOS 3 的核心价值**：它是目前唯一跨越"Expert 分离"和"主干类型"两个维度的架构，也是最接近"最优未来架构"的现有实现。理解它在连续谱上的确切位置至关重要。

---

## 六、研究路线图

| 优先级 | 实验 | 验证问题 | 决定什么 |
|--------|------|---------|---------|
| **P0** | π₀ 梯度传播 | 强耦合是否泛化到 Expert 分离架构？ | **论文 ceiling** |
| **P1** | π₀ prefix 扰动 | 弱耦合是否泛化到 VLA 架构？ | 范式通用性 |
| **P2** | FastWAM/π₀ random latent 替换 | Expert 分离中是否存在有害干扰？ | Expert 分离的必要性 |
| **P3** | COSMOS 3 耦合分析 | 全共享双模架构中是否存在 DreamZero 同款有害干扰？ | COSMOS 3 全共享的代价 |

```
Phase 1（已完成）: DreamZero 先导实验 + 理论框架
Phase 2: P0/P1/P2 关键假设验证（决定论文定位）
Phase 3: 架构原型或论文撰写（取决于 P0）
```

**论文策略**：主贡献 = 识别 Attention 路由为独立范式（已可撰写）；延伸 = 训练强耦合的泛化边界 + 最优架构形态（P0 决定）。P0 不影响可发表性，只影响天花板。

---

## 七、总结

> 机器人世界模型的跨模态交互有两种范式：JEPA 的潜空间对齐，和 WAM/VLA 的 Attention 路由。后者已被四个架构实现——COSMOS 3 和 DreamZero 位于谱的左端（全共享），FastWAM 和 π₀ 位于谱的右端（Expert 分离）。全共享架构存在有害干扰风险（实验 B），Expert 分离架构的训练耦合泛化性待验证（P0）。未来架构的核心问题是：在 Attention 路由的框架下，双模统一（VLM+DiT）能否在 Expert 分离架构中实现？最优的 Expert 分离程度在哪里？这是我们正在回答的问题。
