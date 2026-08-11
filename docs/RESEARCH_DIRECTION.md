# Expert-Interleaved Self-Attention: A Unified Framework for Scalable Robot Control

> 基于对 DreamZero、FastWAM、COSMOS 3、π₀ 和 JEPA 的深入分析
> 2026-08-11

---

## 摘要

我们分析了四个代表性机器人学习架构（DreamZero、FastWAM、COSMOS 3、π₀），发现它们共享同一个核心机制——**Expert-Interleaved Self-Attention**——即不同模态的 Transformer 专家通过拼接 Q/K/V 在同一 Attention 层进行跨模态对齐。我们通过 7 组先导实验在 DreamZero 上验证了这一机制的两个关键属性：**训练时强耦合**（跨模态梯度互通，仅优化视频损失即可显著改善动作预测）和**推理时弱耦合**（动作预测精度不依赖于视频生成质量）。基于此，我们提出一个统一框架：LLM（语义理解）+ Video DiT（视觉预测）+ Action DiT（动作执行）三专家通过 Joint Self-Attention 对齐，支持多频率独立推理。在此框架之上，我们定位了两个互补的研究方向：(1) 利用强耦合在潜空间中实现无需真机的"想象 RL 训练"，(2) 利用弱耦合实现多频率分离的推理加速。

---

## 一、背景与挑战

### 1.1 机器人学习的三个瓶颈

当前机器人学习面临三个相互关联的瓶颈：

**数据瓶颈。** 机器人动作数据极其昂贵。DROID 数据集历时 18 个月、跨 7 个机构收集了 76K episodes，已经是目前最大的公开机器人操控数据集之一，但其视觉多样性远小于互联网视频（数十亿级）。在新的机器人、新的场景、新的任务面前，几乎必然面临"只有视频、没有动作标注"的情况——模型如何继续提升？

**推理瓶颈。** World Action Models (WAM) 通过视频扩散模型同时预测未来画面和动作轨迹。以 DreamZero 为例，一次推理需要 16 步去噪 × 40 层 DiT = 约 10 秒。这在真机部署中是不可接受的。但视频生成真的是动作预测所必需的吗？如果不是，推理成本可以降低一个数量级。

**仿真瓶颈。** 仿真器是 RL 训练的常用媒介，但构建高保真机器人仿真环境需要大量工程工作，且 sim-to-real gap 始终存在。如果模型能够不依赖真机、不依赖仿真器，仅凭**内部想象**（在潜空间中通过视频生成来推演动作后果）就能自我改进，将从根本上改变机器人学习的范式。

### 1.2 WAM 的未解之谜

World Action Models（DreamZero, FastWAM, COSMOS 3）和 Vision-Language-Action models（π₀）都在架构中同时处理视频/图像和动作。论文声称的视频-动作联合建模带来性能提升已被广泛报告，但**为什么**有效？机制是什么？目前没有一个统一的解释。

具体而言，以下问题尚未被系统回答：

1. **视频和动作的耦合到底发生在哪里？** 是数据层面（视频和动作 token 一起进入同一个 transformer），还是参数层面（梯度通过共享权重传播），还是两者兼有？

2. **推理时视频输出质量是否影响动作精度？** 如果去掉视频生成，动作预测会退化多少？如果扰乱视频生成过程，动作是否跟着变差？

3. **训练时视频信号如何影响动作学习？** 视频损失函数的梯度能否通过共享网络权重传递到动作预测？这种传递是正向的还是负向的？

4. **跨模态 Attention 是否具有通用属性？** DreamZero（DiT）、FastWAM（双专家 MoT）、π₀（LLM + Action Expert）、COSMOS 3（统一 MoT）看似架构各异，但它们是否共享同一个底层机制？

这些问题之所以重要，是因为它们的答案直接决定了三个瓶颈的解决路径：

| 如果... | 那么可以... |
|---------|-----------|
| 推理时视频与动作弱耦合 | 跳过视频生成，推理加速 4×+ |
| 训练时视频与动作强耦合 | 通过优化视频来间接改善动作（无需 GT action） |
| 跨模态 Attention 具有通用属性 | 统一理论框架，指导新架构设计 |

### 1.3 已有线索

相关工作的消融实验提供了关键线索但未给出统一解释：

- **FastWAM** 发现：推理时去掉视频生成，action 精度几乎不变（LIBERO 97.6 vs 98.0）；训练时去掉视频联合建模，精度暴跌（91.8 → 83.8）。但没有解释**为什么**推理分离可行。
- **π₀** 实现了 prefix KV cache + action 独立去噪的推理模式，但没有研究 prefix 质量对 action 的影响。
- **COSMOS 3** 支持三种 action 模式（Policy / Inverse Dynamics / Forward Dynamics），但没有分析它们之间的耦合关系。
- **DreamZero** 是唯一的全共享 DiT 架构（没有 expert 分离），提供了最纯粹的"耦合测试"平台。

---

## 二、先导实验：DreamZero 上的 7 组因果验证

我们在 DreamZero（Wan2.1-I2V-14B + LoRA + FSDP，DROID 数据集，checkpoint-1000）上完成了系统性的因果验证。选择 DreamZero 作为实验平台的原因：其全共享 DiT 架构（video 和 action 使用同一组 Q/K/V/FFN 权重，没有 expert 分离）提供了最严格的耦合测试条件——如果在此架构中观察到弱耦合，那么在 Expert 分离的架构中更应成立。

我们在 DreamZero（Wan2.1-I2V-14B + LoRA + FSDP，DROID 数据集，checkpoint-1000）上完成了系统性的因果验证。

### 2.1 推理时：弱耦合——视频质量不影响动作精度

**实验 2a：flow_pred 噪声注入。** 在 16 步去噪的每一步，往视频 flow_pred 注入 σ=0~1.0 的高斯噪声。

| 噪声 σ | Action MSE | vs Full |
|--------|-----------|---------|
| 0.0 (Full) | 24.78 | baseline |
| 0.05 ~ 1.0 | 24.65 ~ 24.82 | ±0.1 |
| AO (skip video) | 41.31 | +16.53 |

**结论**：往视频去噪方向注入任意强度的噪声，动作精度完全不受影响。去噪过程对噪声鲁棒。

**实验 2b：video latents 替换。** 每步去噪后将 video latent 替换为纯噪声或全零。

| 条件 | Action MSE | vs Full |
|------|-----------|---------|
| Full | 21.61 | baseline |
| Random latents | **18.69** | **-13%** |
| Zero latents | **18.75** | **-13%** |
| AO | 49.17 | +127% |

**结论**：用纯噪声取代视频反而**改善了**动作。视频去噪任务在共享 DiT 中与动作竞争 attention 预算。核心机制：AO 模式下 video K/V 在 16 步中完全不变 → attention 冗余；Fresh random 每步不同的 K/V → attention 多样性保持。

**实验 2c：第一帧语义扰动。** 模型推理时只使用第一帧（CLIP 编码），对第一帧做各种扰动，双 checkpoint 验证。

| 扰动 | ckpt-200 vs Full | ckpt-1000 vs Full |
|------|-----------------|-------------------|
| 换成另一个 episode | +1% | 0% |
| 旋转 180° | — | -1% |
| 遮掉一半画面 | — | -1~3% |
| 全黑/全白/全灰 | +12~13% | +14% |
| 随机噪声 | +22% | +21% |
| 非真实图像（正弦波） | — | +19% |
| AO (关掉视频) | +100% | +100% |

**结论**：第一帧语义内容完全无关——换 episode、旋转、遮挡都无影响。CLIP 只需要"像真实照片"的通用锚点，不关心场景语义。

### 2.2 训练时：强耦合——视频梯度显著改变动作

**实验 3a：单步梯度传播。** 4 个 batch，仅 backward video_loss，更新 LoRA 一步。

```
action_loss BEFORE:  5.65
action_loss AFTER:   2.33
Δ:                  -3.32 (-58.8%)
```

**实验 3b：100 步纯视频优化。** 100 步只优化 dynamics_loss，固定验证集评测，与等量联合训练对比。

| | 纯视频优化 (组 A) | 联合训练 (组 B) |
|---|---|---|
| **基线 action_loss** | 1.059 | 2.169 |
| **100 步后** | 0.596 (**-43.7%**) | 0.316 (**-85.5%**) |
| **收敛模式** | 慢启动 (70 步后加速) | 快饱和 (10 步到最优) |

**结论**：Video loss 的梯度通过共享 DiT 权重显著改变 action 输出。纯视频优化无需任何 action 标注或梯度即可将 action loss 减半。联合训练见效更快但饱和也快，纯视频优化持续改善。

### 2.3 统一解释

```
                    训练时（梯度流）              推理时（数据流）
                ┌───────────────────┐         ┌───────────────────┐
video loss ───→ 共享 DiT 权重 ←─── action     video_pred ──✗──→ action_pred
                │ 紧耦合              │         │ 弱耦合              │
                │ 一方改善，另一方受益  │         │ 一方质量，不影响另一方  │
                └───────────────────┘         └───────────────────┘
```

**两者同时成立，不矛盾。** 推理时的弱耦合是因为 action 不直接依赖 video token 的具体数值（只依赖 attention pattern 和位置编码），训练时的强耦合是因为梯度通过共享权重传播。

---

## 三、理论框架：Expert-Interleaved Self-Attention

### 3.1 四种架构，同一种机制

我们分析了 DreamZero、FastWAM、COSMOS 3 和 π₀，发现它们尽管主干类型不同，但跨模态对齐的核心机制完全一致：

```
Layer i (共 N 层):

  Expert A (dim D₁):                 Expert B (dim D₂):
    Q_a = W_qa(x_a)                     Q_b = W_qb(x_b)
    K_a = W_ka(x_a)                     K_b = W_kb(x_b)  
    V_a = W_va(x_a)                     V_b = W_vb(x_b)
         │                                    │
         └────────── Q = cat(Q_a, Q_b) ───────┘
                     K = cat(K_a, K_b)
                     V = cat(V_a, V_b)
                          ↓
                  Joint Flash Attention
                          ↓
          输出切回各 Expert: o_proj → residual → FFN → next layer
```

| | DreamZero | FastWAM | COSMOS 3 | π₀ |
|---|---|---|---|---|
| **Expert A** | 无分离 | Video DiT | AR Reasoner | LLM (Gemma 2B) |
| **Expert B** | 无分离 | Action DiT | DiT Generator | Action (300M) |
| **Expert 数** | 0 (全共享) | 2 | N (灵活) | 2 |
| **对齐机制** | Token 混排 | Joint attn | 模式切换 | **Joint attn** |
| **主干类型** | DiT | DiT | MoT | **LLM** |
| **推理分离** | ✗ | ✓ (KV cache) | 部分 | **✓ (prefix cache)** |

### 3.2 与 JEPA 的对比

JEPA (Joint Embedding Predictive Architecture) 追求的是同模态统一潜空间中的预测：

```
JEPA:
  context ──→ Encoder ──→ shared_latent ──→ Predictor ──→ shared_latent ←── Encoder ←── target
  (同模态)                                 (MLP/Transf)                (共享/EMA)
```

我们的框架是跨模态 Attention 对齐：

```
Ours:
  modality A ──→ Expert A ──→ Q_a, K_a, V_a ──┐
                                                ├── Joint Attention ──→ 各自 FFN
  modality B ──→ Expert B ──→ Q_b, K_b, V_b ──┘
  (不同模态，各自 Encoder/Patch Embedding 独立)
```

| | 统一潜空间 (JEPA) | 跨模态 Attention (我们的框架) |
|---|---|---|
| **对齐层次** | Encoder 层（压缩到统一维度） | Attention 层（Q·K^T 内积，无参数对齐） |
| **信息损失** | 大（video 56320→5120 dim） | 小（各模态保留独立表示空间） |
| **预训练优势** | 无（Encoder 从随机开始） | **有**（DiT/LLM 预训练提供 Attention 先验） |
| **推理分离性** | 紧耦合（统一空间交织） | **弱耦合**（Attention 结束各回各空间） |
| **模态扩展** | 差（新模态需重训 Encoder） | 好（新 Expert + token） |
| **训练难度** | Encoder 承受全部对齐压力 | 40 层分布式对齐，每层学一点 |

**核心洞察**：JEPA 需要一个显式的 Predictor 网络在潜空间做预测；在 Expert-Interleaved Self-Attention 中，**Attention 层就是 Predictor**——Q·K^T 内积在每一层做无参数的跨模态路由。

### 3.3 我们的理论贡献

> 在 Expert-Interleaved Self-Attention 框架下，跨模态耦合在**训练时**和**推理时**表现出不对称的特性：推理时 Token 层面的耦合是弱的（输出质量互相独立），训练时 Gradient 层面的耦合是强的（梯度通过共享 Attention 互通）。**两者同时成立，是同一架构的不同侧面。**

---

## 四、统一架构：LLM + Video DiT + Action DiT 三专家多频率控制

### 4.1 架构设计

将 LLM、Video DiT 和 Action DiT 三个 Expert 通过 Joint Self-Attention 连接，每个 Expert 保留独立的权重（Q/K/V 投影、FFN、RMSNorm），仅在同一层的 Attention 计算中合并。

```
Layer i (共 N 层):

  LLM Expert (AR, causal):     Video Expert (DiT, diffusion):    Action Expert (DiT, diffusion):
    Q_l, K_l, V_l (2048)         Q_v, K_v, V_v (3072)              Q_a, K_a, V_a (1024)
         │                              │                                 │
         └──────────────────────────────┼─────────────────────────────────┘
                                        ↓
              Q = cat(Q_l, Q_v, Q_a)   K = cat(K_l, K_v, K_a)   V = cat(V_l, V_v, V_a)
                                        ↓
                               Joint Flash Attention
                                        ↓
              LLM: o_proj → residual → FFN_l  |  Video: o_proj → FFN_v  |  Action: o_proj → FFN_a
```

### 4.2 多频率执行

三类 Expert 天然对应机器人控制的三个时域层次：

```
语义层（LLM, AR/causal）：       ★               ★               ★
  "任务是什么？"                  ↑ 低频（~1 Hz）
  K_l/V_l 缓存，任务切换时刷新     │
                                  │
视觉层（Video DiT, diffusion）： ★─★─★─★─★─★─★─★─★─★─★─★
  "接下来画面变成什么样？"         ↑ 中频（~4-10 Hz）
  K_v/V_v 缓存，N 个 action 更新一次│
                                   │
动作层（Action DiT, diffusion）： ★★★★★★★★★★★★★★★★★★★★★★★★
  "具体关节位置？"                 ↑ 高频（~20-50 Hz）
  每步 Forward 仅 Action Expert     │
  读 (K_l, V_l) + (K_v, V_v)       │
```

**每步 Action 去噪只需要 Action Expert 的 forward + O(1) 读取缓存的 LLM/Video K/V**，不需要碰 LLM 和 Video 的 FFN。

### 4.3 与现有架构的关系

```
COSMOS 3 MoT（模式切换）：          π₀（双专家）：              本文（三专家多频率）：
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────────┐
│ 同一组 blocks          │    │ LLM + Action         │    │ LLM + Video + Action     │
│ AR mode OR DiT mode   │    │ 2 experts            │    │ 3 experts                │
│ 不能同时               │    │ 无频率分离            │    │ 多频率独立执行              │
└──────────────────────┘    └──────────────────────┘    └──────────────────────────┘
      ↓ 扩展                      ↓ 扩展                      ↓
      加 Action Expert            加 Video Expert             从推理加速到想象训练
      支持多频率                  支持多频率                  完整的两个应用方向
```

---

## 五、两个应用方向

### 方向 1：利用弱耦合——多频率分离推理加速

**核心思路**：LLM 和 Video 的 K/V 被缓存后，Action 可以独立高频去噪。无需等待 LLM 重新推理或视频重新生成。

**已有证据**：
- π₀：prefix (image+language) KV cache → action 10 步 Euler 去噪
- FastWAM：video KV cache → action 16 步去噪，精度不变，快 4.3×
- 我们的实验：video 输出质量与 action 精度无因果关联（7 组实验一致验证）

**验证实验**：在 π₀ 和 FastWAM 上分别验证 prefix/video 扰动不影响 action 精度（平行于 DreamZero 实验 2a/2c），证明弱耦合是 Expert-Interleaved Attention 的通用属性，而非特定实现的偶然现象。

### 方向 2：利用强耦合——潜空间想象 RL 训练

**核心思路**：在三专家架构中，Video Expert 充当世界模型（autoregressive rollout 预测未来帧），LLM Expert 充当语义裁判（判断任务是否完成），Action Expert 作为被训练的策略。GRPO 的 reward 来自 LLM 对生成视频的语义评估，梯度通过 Joint Attention 传递到 Action。

```
想象 RL 闭环：

1. LLM 推理： "wipe the countertop" → 任务分解 → K_l 缓存
2. Action 采样： N 条不同噪声种子的去噪轨迹 → N 组 action chunk + video_pred
3. Video rollout： fd(frame_t, action_t) → frame_{t+1} → ... → frame_{t+H}
4. LLM 裁判： "Did the robot successfully wipe the countertop?" → yes/no → reward
5. GRPO 更新： advantage → ∂L/∂W → 改善 Action（通过共享 Attention 梯度传递）
```

**为什么 COSMOS 3 是 ideal 平台**：同时具备 Policy（生成 video+action）、Forward Dynamics（autoregressive rollout）和 Reasoner（VLM 语义评估）。

**已有证据**：
- 我们的梯度传播实验：video loss → action 改善 58.8%（单步）/ 43.7%（100 步）
- COSMOS 3 的 Forward Dynamics：支持 autoregressive rollout（天然世界模型）
- COSMOS 3 的 Reasoner：内置 VLM，可做任务完成度判断

**待解决**：Reasoner 对机器人任务的判断准确率（需评测和可能的微调）；Autoregressive rollout 的累积误差。

### 方向 3（综合）：架构验证论文

两个方向共享同一个架构和同一个核心 insight。一篇统一论文的叙事结构：

```
Title: Expert-Interleaved Self-Attention for Hierarchical Robot Control
       with Decoupled Inference and Coupled Training

贡献 1（架构）: LLM + Video + Action 三专家多频率架构
贡献 2（实验）: 系统验证 Expert-Interleaved Attention 的强耦合/弱耦合双属性
贡献 3（应用 A）: 多频率分离推理加速（弱耦合利用）
贡献 4（应用 B）: 潜空间想象 RL 训练（强耦合利用）
```

---

## 六、实现路线图

```
Phase 1（已完成）: DreamZero 先导实验
  ✅ 训练强耦合验证（单步 -58.8%，多步 -43.7%）
  ✅ 推理弱耦合验证（7 组实验，多维度一致性）
  ✅ FastWAM / COSMOS 3 / π₀ / JEPA 深度分析
  ✅ 统一理论框架建立
  ✅ 两个应用方向定义

Phase 2（4-6 周）: 弱耦合验证 + 推理加速原型
  - π₀ prefix 扰动实验（平行 DreamZero 实验 2a/2c）
  - FastWAM/π₀ 多频率分离推理实现
  - DROID/LIBERO 精度+延迟评测
  - 产出：弱耦合通用性验证 + 加速方法

Phase 3（6-8 周）: 三专家架构搭建
  - 从 π₀ (LLM + Action) 出发，加 Video DiT Expert
  - LoRA fine-tune，冻结 LLM + Video 主干
  - 多频率执行验证（LLM 1Hz / Video 10Hz / Action 50Hz）

Phase 4（8-12 周）: 潜空间想象 RL
  - 在 COSMOS 3 上实现 GRPO 训练管线
  - Reasoner-based reward 函数设计
  - 对比：纯 SFT vs SFT+RL 的 action 精度

Phase 5（综合）: 论文撰写
  - 整合 Phase 2-4 的实验结果
  - 理论框架 + 架构设计 + 两个应用方向的实验验证
```

---

## 七、总结

> **四种架构（DreamZero, FastWAM, COSMOS 3, π₀）共享同一个核心机制：Expert-Interleaved Self-Attention。此机制的 Attention 层作为无参数跨模态 Predictor，推理时弱耦合（支持独立执行），训练时强耦合（支持跨模态梯度传递）。基于此，我们提出 LLM + Video + Action 三专家多频率架构，定位了两个互补的应用方向：弱耦合推理加速和强耦合想象 RL 训练。**
