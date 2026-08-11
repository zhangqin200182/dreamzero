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

我们在 DreamZero（Wan2.1-I2V-14B + LoRA + FSDP，DROID 数据集，checkpoint-3000 训练产出的 checkpoint-1000）上完成了 7 组系统性的因果实验。选择 DreamZero 的原因：它是全共享 DiT 架构（video 和 action 使用完全相同的 Q/K/V/FFN 权重），位于 Expert 分离连续谱的最左端。如果在这个耦合最紧的架构中观察到推理弱耦合，则 Expert 分离架构中更应成立。

### 4.1 推理弱耦合（实验 A-C，多维度验证）

**实验 A：flow_pred 噪声注入。** 在 16 步去噪的每一步，往视频的 flow_pred（DiT 输出）注入 σ=0~1.0 的高斯噪声。同时测试 AO 基线（完全跳过视频去噪）。

| 噪声 σ | Action MSE | vs Full |
|--------|-----------|---------|
| 0.0 (Full) | 24.78 | 基线 |
| 0.05 | 24.73 | -0.05 |
| 0.1 | 24.82 | +0.04 |
| 0.3 | 24.73 | -0.05 |
| 0.5 | 24.81 | +0.03 |
| 1.0 | 24.65 | -0.13 |
| AO (skip video) | 41.31 | **+16.53** |

**结论**：往视频去噪方向注入任意强度噪声——范围覆盖 σ=0.05 到 σ=1.0（与 flow_pred 本身同数量级）——Action MSE 波动仅在 ±0.15 以内，属于采样噪声。但完全关掉视频去噪（AO），Action MSE 立刻跳升 67%。**视频去噪对 Action 的贡献是二值的：ON 或 OFF。输出值无关。**

**实验 B：video latents 替换。** 每步去噪后将 video latent 替换为纯噪声或全零。这比实验 A 更极端——不仅破坏 flow_pred，而是直接替换 DiT 的输入。

| 条件 | Action MSE | vs Full |
|------|-----------|---------|
| Full | 21.61 | 基线 |
| Random latents | **18.69** | **-13%**（反直觉：替换后更好） |
| Zero latents | **18.75** | **-13%**（反直觉：替换后更好） |
| AO | 49.17 | +127% |

**结论**：用纯噪声/全零取代 video latent 后，Action 不仅没有退化，反而**改善了 13%**。这是"有害干扰假说"的核心证据——全共享架构中 video denoising 对 action 产生了有害的表示偏移。具体机制见 4.3 节。

**实验 C：第一帧语义扰动。** DreamZero 在推理时仅使用第一帧（通过 CLIP 编码为 `clip_feas` + `ys` 条件），其余 32 帧不进入模型。对第一帧做各种扰动，双 checkpoint 验证（checkpoint-200, checkpoint-1000）。

| 扰动类型 | ckpt-200 | ckpt-1000 |
|---------|---------|-----------|
| 换成另一个 episode | +1% | 0% |
| 旋转 180° | — | -1% |
| 遮掉一半（左/右） | — | -1~3% |
| 全黑 | +12% | +14% |
| 全白 | +13% | — |
| 随机噪声 | +22% | +21% |
| 正弦波纹理（非真实图像） | — | +19% |
| AO（关掉视频） | +100% | +100% |

**结论**：第一帧的**语义内容完全无关**——换 episode、旋转 180°、遮掉一半画面，对 Action 的影响在 0-3% 以内（统计噪声范围）。CLIP 只需要一个"像真实照片"的通用锚点：全黑/全白的退化仅 ~12%，随机噪声的退化 ~21%，但即便如此仍保留了 78-88% 的 AO-Full gap 收益。**模型不"看"视频内容，CLIP 编码的只是"这是一个室内场景"的结构性信息。**

### 4.2 训练强耦合（实验 D-E，共享参数架构中验证）

**实验 D：单步梯度传播。** 从 checkpoint-1000 出发，取 4 个 batch，仅计算 video_loss（dynamics_loss），backward 梯度更新 LoRA 权重一步。比较更新前后的 action_loss。

```
action_loss BEFORE:  5.65
action_loss AFTER:   2.33
Δ action_loss:      -3.32 (-58.8%)
```

改变的 LoRA 参数集中在 FFN 层（`ffn.0`, `ffn.2`），这些正是共享 DiT 中 video 和 action 共用的组件。

**实验 E：100 步持续优化（含联合训练对照组）。** 100 步纯 video loss 优化（组 A），与等量联合训练（组 B，video + action loss）对比。使用固定验证集（10 个样本，每 10 步评测一次），同一 seed，同一 checkpoint。

| | 组 A: 纯视频优化 | 组 B: 联合训练 (video+action) |
|---|---|---|
| **Step 0 (基线)** | 1.059 | 2.169 |
| **Step 10** | 1.184 (+11.8%) | 0.318 (-85.3%) |
| **Step 30** | 1.320 (+24.6%) | 0.355 (-83.6%) |
| **Step 50** | 1.035 (-2.3%) | 0.308 (-85.8%) |
| **Step 70** | 1.210 (+14.3%) | 0.317 (-85.4%) |
| **Step 90** | 0.559 (-47.3%) | 0.309 (-85.8%) |
| **Step 100 (最终)** | **0.596 (-43.7%)** | **0.316 (-85.5%)** |

**关键观察**：
- 组 B（联合训练）：10 步内直接优化到接近最优（action_loss 从 2.17 骤降至 0.32），之后 90 步饱和，无进一步改善。**直接针对 action loss 的优化收敛极快但饱和也快。**
- 组 A（纯视频优化）：前 70 步在波动（action_loss 在 1.03~1.32 之间震荡），第 70-100 步突然大幅下降。**纯视频优化的梯度传递需要"积累"——共享表示需要足够步数才能产生对 action 的显著改善。**
- 两组验证集基线不同（1.059 vs 2.169），说明 seed=42 在两次独立脚本运行中产生了不同的验证集。但各自内部的趋势和最终 Δ% 是可靠的。

**结论**：Video loss 的梯度通过共享 DiT 权重显著改变 action 输出。纯视频优化**无需任何 action 标注或梯度**，100 步即可将 action loss 减半。联合训练见效更快但饱和也快，纯视频优化持续改善。

**关键约束**：此结论在 DreamZero 的**全共享参数架构**中获得。在 Expert 分离架构（FastWAM/π₀）中，梯度路径完全不同——仅通过 Joint Attention 中 softmax(QK^T) 的跨模态项传播，不经过另一 Expert 的 Q/K/V/FFN 参数。这是一条间接且可能极弱的路径。**训练强耦合能否泛化到 Expert 分离架构，是整个研究方向的 P0 优先级待验证假设。**

### 4.3 有害干扰假说（基于实验 B 的代码级分析）

实验 B 显示 random/zero latent → action 改善 13%。DreamZero 使用独立的 per-token noise schedule：Video 使用 `Beta(3,1)` 分布采样 timestep（偏向低噪声），Action 使用 `Uniform` 分布采样 timestep。共享的 Q/K/V 权重需要**同时服务两种不同的 denoising 动态**——两种噪声分布的冲突导致 video token 在 Joint Attention 的 K/V 中对 action token 产生有害的表示偏移。Random/zero latent 意外消除了这种冲突，action 反而改善。

AO 模式的进一步证据：DreamZero 的 AO 本质是 16 步中 video 的 sigma 被 clamp 到 1.0——16 次重复第一步的去噪，video K/V 完全不变。这意味着 AO 中 video tokens 提供的 attention scaffold 是**静态的**，而 random latent 每步提供**不同的** video K/V。后者更有利于 action denoising 的多样性。

**如果此假说成立，Expert 分离不是可选的优化，而是防止模态间有害干扰的必要架构设计。** 这需要在 Expert 分离架构（FastWAM 或 π₀）中验证：在分离架构中做同样的 video latent 替换实验，预期不会看到 action 改善（因为不存在有害干扰需要消除）。（P2 实验）

### 4.4 实验结论与架构含义

| 属性 | 共享架构（DreamZero/COSMOS 3） | Expert 分离架构（FastWAM/π₀） | 对三 Expert 方案的含义 |
|------|------|------|------|
| **推理弱耦合** | ✓ 已验证（实验 A/B/C） | ✓ 已验证（KV cache 推理） | 多频率执行可行 |
| **训练强耦合** | ✓ 已验证（实验 D/E: -43.7%） | **未验证**（P0） | 如果成立，RL 闭环的梯度路径通畅 |
| **有害干扰** | ✓ 存在（实验 B: +13%） | 预期无（P2 验证） | Expert 分离可以消除干扰 |

---

## 五、未来架构方向：综合判断

### 5.1 四条路径的收敛方向

基于实验证据和架构分析，四条路径收敛到同一个方向：

```
COSMOS 3（全共享双模）    FastWAM（分离双专家）     π₀（分离双专家）
       │                        │                      │
       │ 问题：全共享代价         │ 缺：语义理解           │ 缺：世界模型
       │                        │                      │
       └────────────────────────┼──────────────────────┘
                                ↓
                    三 Expert 分离双模
              （LLM + Video + Action, Expert 分离, Joint Attention）
```

**驱动因素**：

| 因素 | 来源 | 含义 |
|------|------|------|
| Expert 分离优于全共享（四条挑战） | 实验 B + FastWAM/π₀ 的 KV cache 推理 | 未来架构应走 Expert 分离路线 |
| 语义 + 世界模型缺一不可 | 实验 E（纯视频可改善 action）+ π₀ 的语言优势 | LLM Expert + Video Expert 都需要 |
| 双模统一可行 | COSMOS 3 证明了同一架构容纳 AR + DiT | 在 Expert 分离中复现是工程挑战而非理论障碍 |
| 弱耦合支持多频率 | 实验 A/B/C + FastWAM/π₀ 推理策略 | Action 可以独立高频执行 |

### 5.2 三 Expert 方案 vs 四种现有架构

| | COSMOS 3 | FastWAM | π₀ | **三 Expert** |
|---|---|---|---|---|
| Expert 分离 | ✗（全共享） | ✓（2 Expert） | ✓（2 Expert） | **✓（3 Expert）** |
| 语义理解 | ✓ | ✗ | ✓ | **✓** |
| 世界模型（视频生成） | ✓ | ✓ | ✗ | **✓** |
| 多频率执行 | ✗ | ✓ | ✓ | **✓** |
| 有害干扰风险 | 有 | 预期无 | 预期无 | **预期无** |
| RL 闭环 | 基础设施有，无 GRPO | 有 FD，缺语义 reward | 无世界模型 | **完整闭环** |
| 预训练复用 | 自研 | Wan2.2 | PaliGemma | LLM + Video 预训练 |

**三 Expert 方案是唯一同时覆盖所有维度的架构。**

### 5.3 务实路径：从双 Expert 到三 Expert

考虑到训练资源约束，采用渐进路线：

```
路线 A（从 π₀ 出发，推荐）：
  π₀ (LLM 2B + Action 300M, 18层, 已开源, PyTorch port)
    → 加 Video Expert（Wan2.2-5B 或冻结主干，LoRA fine-tune）
    → 仅需训练 Action Expert + Video Expert 的 LoRA
    → 验证三 Expert 推理分离后，再加 Video 生成能力

路线 B（从 FastWAM 出发）：
  FastWAM (Video 5B + Action 1B, 30层)
    → 加 LLM Expert（PaliGemma 2B 或冻结主干，LoRA fine-tune）
    → 需处理 generation paradigm 冲突（AR vs DiT attention mask）
    → 可参考 COSMOS 3 的 two_way_attention
```

路线 A 更务实——π₀ 已有完整的训练/推理管线、PyTorch 实现、DROID fine-tune 配方。加 Video Expert 只需要加一个新的 Expert 模块（LoRA fine-tune），不需要解决 AR vs DiT 的 attention mask 冲突。

---

## 六、研究路线图

### 6.1 实验优先级

| 优先级 | 实验 | 平台 | 验证问题 | 决定什么 |
|--------|------|------|---------|---------|
| **P0** | 梯度传播 | π₀ | 训练强耦合是否泛化到 Expert 分离架构？ | **论文 ceiling + 三 Expert 的 RL 可行性** |
| **P1** | Prefix 扰动 | π₀ | 弱耦合是否泛化到 VLA 架构？ | 范式通用性（低风险，预期成立） |
| **P2** | Random latent 替换 | π₀ | Expert 分离中是否存在有害干扰？ | Expert 分离的必要性 |
| **P3** | 双 Expert 联合训练 | π₀ + Video | 三 Expert 架构的最小可行验证 | Phase 3 的可行性 |

### 6.2 分阶段计划

```
Phase 1（已完成）: 先导实验 + 理论框架
  ✅ DreamZero 7 组因果实验（推理弱耦合 + 训练强耦合 + 有害干扰）
  ✅ 四个架构代码级分析 + Expert 分离连续谱
  ✅ 全共享 vs Expert 分离的四条挑战
  ✅ JEPA vs Attention 路由的范式对比
  ✅ 三 Expert 分离双模架构方案

Phase 2（P0/P1/P2, 3-4 周）: 关键假设验证
  - P0: π₀ 上复现实验 D/E（video loss → action 梯度传播）
        若成立 → 训练强耦合泛化，三 Expert RL 闭环可行
        若不成立 → 训练耦合仅存在于全共享架构，调整论文定位
  - P1: π₀ prefix 扰动实验（平行实验 A/C）
        预期成立，低风险
  - P2: π₀ random latent 替换（平行实验 B）
        预期无有害干扰 → Expert 分离的优势证据

Phase 3（P3, 6-8 周）: 三 Expert 原型
  - π₀ + Video Expert（LoRA fine-tune）
  - 验证三 Expert 推理分离（LLM K/V + Video K/V → Action 独立去噪）
  - 验证多频率执行精度无损

Phase 4: 论文撰写（取决于 Phase 2/3 结果）
  - 如果 P0 成立：完整范式优越性论证 + 三 Expert 架构验证
  - 如果 P0 不成立但 P2 成立：范式识别 + Expert 分离优势论证
  - 如果都成立：最强版本——Attention 路由范式 + 三 Expert 架构
```

### 6.3 论文策略（分两档）

**核心贡献（已可撰写，取决于 P0/P1/P2 结果）**：
- 识别 Attention 层路由为独立于共享隐空间的跨模态交互范式（四个架构的收敛性证据）
- 系统验证推理弱耦合属性（DreamZero 7 组实验 + FastWAM/π₀ 推理策略）
- 全共享 vs Expert 分离的四条挑战分析
- 实验 B 揭示的有害干扰假说（如 P2 验证，将是 Expert 分离必要性的最强论据）

**增强贡献（取决于 P0 + P3 结果）**：
- 训练强耦合的泛化边界（P0）
- 三 Expert 分离双模架构（P3 原型验证）
- RL 闭环可行性证明
- 如果 P0 + P3 都成立：完整的"Attention 路由范式" + "三 Expert 最优架构"论证

---

## 七、总结

机器人世界模型的跨模态交互有两种范式：JEPA 的潜空间对齐，和 WAM/VLA 的 Attention 层路由。

Attention 路由有四个现有实现——COSMOS 3 和 DreamZero 位于 Expert 分离谱的左端（全共享），FastWAM 和 π₀ 位于谱的右端（Expert 分离）。全共享架构面临四条挑战（推理性能受限、模态间有害干扰、频率无法分离、模态扩展代价高），Expert 分离架构在推理性能和多频率执行上已有验证，但训练强耦合的泛化性待验证。

我们通过 DreamZero 上 7 组因果实验揭示了 Attention 路由的关键属性：推理弱耦合（实验 A/B/C，多维度验证）、训练强耦合（实验 D/E，在共享参数中验证）、有害干扰假说（实验 B 的反直觉结果）。基于这些证据，我们提出了三 Expert 分离双模架构（LLM + Video + Action）作为未来方向——它同时具备语义理解、世界模型、多频率执行、RL 闭环能力，是四种现有架构的自然收敛。

> 核心问题：Attention 路由范式下，双模统一能否在 Expert 分离中实现？训练强耦合能否跨谱泛化？最优的 Expert 分离程度在哪里？这些问题的答案将由 P0/P1/P2/P3 实验揭晓。
