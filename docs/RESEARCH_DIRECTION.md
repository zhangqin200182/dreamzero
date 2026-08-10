# DreamZero 视频-动作耦合机制研究：从推理加速到潜空间想象训练

> 日期：2026-08-10
> 基于：DreamZero (Wan2.1-I2V-14B + LoRA + FSDP)、FastWAM (Wan2.2-5B + MoT)、COSMOS 3 (Omnimodal MoT)
> 数据：DROID v2/v3、LIBERO

---

## 一、研究动机

World Action Models (WAM) 通过视频 Diffusion Transformer 同时预测未来视频画面和机器人动作。一个核心但未被充分研究的问题是：**视频预测和动作预测之间的耦合机制是怎样的？这种耦合在推理和训练中分别意味着什么？**

回答这个问题将解锁两个直接的应用方向：
1. **推理加速**：利用弱耦合跳过冗余的视频生成计算
2. **想象训练**：利用强耦合通过优化视频生成来间接优化动作，无需真机

---

## 二、先导实验与核心发现

我们在 DreamZero checkpoint-1000 上完成了 7 组实验（详见 `VIDEO_ACTION_COUPLING_REPORT.md`），核心发现如下：

### 2.1 推理时：弱耦合——视频质量不影响动作精度

| 实验 | 方法 | 关键结果 |
|------|------|---------|
| **flow_pred 噪声注入** | 往视频 flow_pred 注入 σ=0~1.0 高斯噪声 | Action MSE 波动 < 0.2%，**完全不受影响** |
| **video latents 替换** | 每步去噪后替换视频 latent 为纯噪声/全零 | Action **反而更好** (-13%)，视频去噪与动作竞争 attention |
| **第一帧语义扰动** | 替换/旋转/遮挡第一帧（双 checkpoint 验证） | 换 episode = 0% 退化，旋转 180° = 0% 退化，正弦波图案 = 19% 退化 |
| **AO 模式机制分析** | DreamZero AO 本质：sigma 卡在 1.0，16 步重复 | 与随机视频 latent 对比，发现关键在视频 K/V 值是否变化而非去噪进度 |

**结论**：视频去噪对动作的贡献是二值的——ON（video tokens 在场）或 OFF（AO 模式）。视频输出质量与动作精度无因果关联。

### 2.2 训练时：强耦合——视频梯度显著改变动作

| 实验 | 方法 | 关键结果 |
|------|------|---------|
| **单步梯度传播** | 4 batch，仅 backward video_loss，更新 LoRA 一步 | Action 改变 **-58.8%** |
| **100 步纯视频优化** | 100 步只优化 dynamics_loss，固定验证集评测 | Action **-43.7%**（组 A），联合训练 **-85.5%**（组 B） |

**结论**：Video loss 的梯度通过共享 DiT 权重传播到 action。纯视频优化 100 步即可将 action loss 减半——**不需要任何 action 标注或梯度**。

### 2.3 统一解释

```
                    训练时（梯度流）              推理时（数据流）
                ┌───────────────────┐         ┌───────────────────┐
video loss ───→ 共享 DiT 权重 ←─── action     video_pred ──✗──→ action_pred
                │ 紧耦合              │         │ 弱耦合              │
                │ 一方改善，另一方受益  │         │ 一方质量，不影响另一方  │
                └───────────────────┘         └───────────────────┘
```

两者同时成立，不矛盾。RL 通过训练时的梯度耦合起作用，不依赖推理时的因果链。

---

## 三、相关工作与独立验证

### 3.1 FastWAM（arXiv 2603.16666）

- Wan2.2-5B + MoT（video expert 3072 dim + action expert 1024 dim，共享 attention）
- **独立验证了我们的核心发现**：推理时去掉视频生成，精度几乎不变（LIBERO 97.6 vs 98.0，RoboTwin 91.8 vs 91.3）
- 训练时去掉视频联合建模，精度暴跌（LIBERO -4.1，RoboTwin -8.0）
- 推理加速 4.3×（190ms vs 810ms），通过 `prefill_video_cache` + `infer_action` 实现

### 3.2 COSMOS 3（NVIDIA, 2026.5）

- 统一 MoT：AR Reasoner + DiT Generator，3D mRoPE 跨模态位置编码，5 种模态
- **三种 action 模式**：Policy（首帧→video+action）、Inverse Dynamics（video→action）、Forward Dynamics（首帧+action→video）
- **Forward Dynamics 支持 autoregressive rollout**：天然的世界模型，支持潜空间多步想象
- 基础设施完备但**尚无 RL 训练配方**——这正好是我们切入的机会

### 3.3 三项目对比

| | DreamZero | FastWAM | COSMOS 3 |
|---|---|---|---|
| **DiT 基座** | Wan2.1-14B | Wan2.2-5B | 自研 |
| **架构** | 全共享单 DiT | 双专家 MoT | 统一 MoT（AR+DiT+3D mRoPE） |
| **耦合方式** | 所有层共享 | 共享 attention，独立 FFN | 同一 blocks，模式切换 |
| **推理可分离** | ✗（拆不开） | ✓（KV cache） | 部分（policy 必须生成 video） |
| **多模态** | Video+Action+Text | Video+Action+Text | +Audio+Image+多 embodiment |
| **RL 配方** | 无 | 无 | 无（基础完备） |

---

## 四、理论框架：Cross-Modal Latent Alignment via Attention-as-Predictor

### 4.1 Attention 层 = 跨模态 Predictor

我们的实验发现可以统一到一个框架下：**Attention 层本身就是跨模态的潜空间对齐机制**。

```
                              ┌──────────────────────────┐
                              │   Shared Attention Space  │
                              │   (40 层 DiT + mRoPE)     │
                              │                          │
  video ──→ VAE ──→ v_token ──┤                          ├──→ Head ──→ video_pred
                              │   Q/K/V 跨模态对齐        │
  action ─→ Encoder → a_token─┤                          ├──→ Decoder → action_pred
                              │   "Attention = Predictor" │
  text ───→ T5 ────→ t_token ─┤                          ├──→ (AR 输出)
                              │                          │
  audio ──→ Encoder → s_token─┤                          ├──→ Decoder → audio_pred
                              └──────────────────────────┘
```

**核心洞察**：JEPA 需要一个显式的 Predictor 网络（通常是 MLP 或小型 Transformer）在潜空间中做预测。但在 DiT 架构中，**40 层 cross-attention 本身就是 Predictor**——它在每一层通过 Q/K/V 计算跨模态的 attention score，这是一个无参数的跨模态对齐机制。

### 4.2 JEPA 对比：统一潜空间 vs 跨模态 Attention

JEPA 追求的是**同模态统一潜空间**：
```
JEPA:
  context blocks ──→ Encoder ──→ shared_latent ──→ Predictor ──→ shared_latent  ←── Encoder ←── target blocks
  (可见 patch)                              (同空间)   (MLP/Transf)   (同空间)       (共享/EMA)     (被遮挡 patch)
  
  关键：x 和 y 是同一模态的不同部分
       Encoder 共享（或 EMA），产生同一潜空间的表示
       Predictor 在同一空间内做状态转移
```

我们的框架是**跨模态 Attention 对齐**：
```
Ours:
  video token (VAE latent) ──→ Qv,Kv,Vv ──┐
                                            ├── 联合 Attention ──→ 输出切回各自空间
  action token (MLP latent) ─→ Qa,Ka,Va ──┘
  
  关键：video 和 action 是不同的模态
       Encoder 各自独立（VAE vs MLP），潜空间完全不同
       Attention 让它们在同一层"对话"，但不统一它们的潜空间
```

### 4.3 两种方案对比

| | 统一潜空间（JEPA 追求） | 跨模态 Attention（我们的框架） |
|---|---|---|
| **对齐层次** | Encoder 层（压缩到统一维度） | Attention 层（Q/K/V 内积，无参数） |
| **信息损失** | 大：video 56320-dim → 5120-dim 损失巨大 | 小：各模态保留自己的表示空间 |
| **Encoder 压力** | 全部对齐压力在 Encoder | 分散到 40 层 attention，每层学一点 |
| **预训练优势** | 无（Encoder 从随机开始） | **有**：DiT 预训练已经学会了视频 attention 模式 |
| **推理分离性** | 紧耦合（统一空间内彼此交织） | **弱耦合**（attention 结束各自独立） |
| **模态可扩展性** | 差：新模态 → 重新训练 Encoder | 好：新模态 → 加 token + expert |
| **注意力对齐** | `Q·K^T` 无参数路由 | 同（无参数） |
| **训练难度** | Encoder 承受全部对齐压力 | 40 层分布式对齐 + 预训练提供起点 |

### 4.4 三个层次的对齐对应

```
JEPA 类型                     | 我们的对应
─────────────────────────────────────────────────────
Image JEPA:                  | DreamZero 第一帧
  context → latent → target  | CLIP(第一帧) → attention → y_conditioning

Video JEPA:                  | COSMOS 3 Forward Dynamics
  past_frames → latent →      | frame_t + action_t → attention → frame_{t+1}
  future_frames              | autoregressive rollout = video JEPA

Multi-modal JEPA:            | FastWAM / COSMOS 3
  (video, text) → latent →   | (video, language) → attention → (action, next_video)
  (action, video)            | 跨模态对齐在同一个 attention 空间
```

### 4.5 我们的理论贡献：区分两种耦合

```
JEPA 框架下：
  推理时（Token 层）：
    video_token 的数值 ──✗──→ action_token 的数值
    弱耦合。符合 JEPA——潜空间预测不需要像素重建。
    跨模态 Attention 的"潜空间"是 attention pattern，
    不是 token 值，所以 token 值改变不影响对齐。

  训练时（梯度层）：
    ∂L_video/∂W ──→ 共享权重 ←── ∂L_action/∂W
    紧耦合。超出 JEPA 范围——JEPA 通常分开训练各组件。
    共享权重的梯度互通是跨模态 Attention 的特有能力。
```

**这就是为什么"弱耦合推理 + 强耦合训练"可以同时成立**。JEPA 文献关注前者（避免像素重构），我们的 RL 方案利用后者（跨模态梯度传递）。两者是同一架构的不同侧面，互补而非矛盾。

---

## 五、研究方向

### 方向 1：弱耦合推理加速

**目标**：利用推理时视频-动作弱耦合，实现 action-only 快速推理，精度不变，速度 3-4×。

**核心思路**（参考 FastWAM）：
```
当前（COSMOS 3 policy 或 DreamZero Full）：
  video tokens 30 步去噪 + action tokens 30 步去噪

加速方案：
  1. prefill：video tokens 过一次 DiT → 缓存各层 K/V
  2. action denoising：30 步，每步用缓存的 video K/V
  3. 跳过 video 的 denoising update 和 VAE encode/decode
```

**可行性**：
- FastWAM 已证明该方案在 LIBERO 和 RoboTwin 上精度无损
- 我们的实验从机制层面验证了 video 质量与 action 精度无因果关联
- DreamZero 当前全共享 DiT 无法物理分离，但可以借鉴思路做 KV cache
- COSMOS 3 的 MoT 架构天然适合此方案

**风险**：
- COSMOS 3 源码（cosmos-framework）未完全开放
- 需要在 Diffusers pipeline 层面实现，或等待框架开源

**预期成果**：
- 在 DROID 场景下验证 action-only 推理精度不降
- 推理延迟降低至当前的 25-30%
- 发表"视频-动作弱耦合推理"的方法论文

---

### 方向 2：强耦合潜空间想象训练

**目标**：利用训练时视频-动作强耦合，通过 RL 优化视频生成质量来间接优化动作——在"想象"中训练，无需真机或仿真器。

**核心思路**：
```
COSMOS 3 闭环 RL 流程：

1. Policy 生成 action + video：
   首帧 + 指令 → action_chunk + video_pred

2. Forward Dynamics 模拟结果（可选多步 rollout）：
   frame_t + action_t → frame_{t+1}

3. Reasoner 评估任务完成度（reward）：
   Reasoner(video_pred, "Did the robot successfully wipe the countertop?") → {yes/no}

4. GRPO 更新 Policy：
   advantage → ∂L/∂weights → 改善 action
```

**为什么 COSMOS 3 适合做这个**：

| 能力 | COSMOS 3 | DreamZero |
|------|---------|-----------|
| 视频生成（Policy） | ✓ | ✓ |
| 世界模型（FD rollout） | ✓（autoregressive） | ✗ |
| 语义评估（Reasoner） | ✓（内置 VLM） | ✗ |
| 多模态统一 | ✓（3D mRoPE） | ✗ |

**COMSOS 3 是这一研究方向的理想实验平台。**

**挑战**：
- Reasoner 对机器人任务理解的准确率未知（可能需要先微调）
- 无参考实现——首次尝试
- Autoregressive rollout 有累积误差

**预期成果**：
- 证明"视频 RL → action 改善"的闭环可行性
- 建立无需真机的视觉想象训练范式
- 发表"潜空间想象 RL 训练"的原创方法

---

## 六、实现路径

```
Phase 1（已完成）: DreamZero 先导实验
  ✅ 梯度耦合验证（单步 -58.8%，多步 -43.7%）
  ✅ 推理弱耦合验证（flow_pred 噪声、video latent 替换、第一帧扰动）
  ✅ FastWAM + COSMOS 3 深度分析
  ✅ 两个研究方向定义

Phase 2（方向 1, 3-4 周）: 推理加速原型
  - 在 DreamZero 或 COSMOS 3 上实现 video KV cache + action-only 推理
  - DROID 场景评测（精度 + 延迟 vs Full 模式）
  - 产出：推理加速方法论文初稿

Phase 3（方向 2, 6-8 周）: 潜空间想象 RL 原型
  - 在 COSMOS 3 上实现 GRPO 训练管线
  - Reasoner-based reward 函数设计与消融
  - DROID 场景评测（action MSE + rollout 视频质量）
  - 产出：想象 RL 方法论文初稿

Phase 4（综合）: 双方向整合
  - 加速推理 + RL 训练的组合效果
  - 产出：完整研究论文
```

---

## 七、分工建议

| 方向 | 核心贡献 | 适合角色 |
|------|---------|---------|
| **推理加速** | 工程实现为主，理论清晰 | 工程能力强的研究者 |
| **想象训练** | 方法设计新颖，风险较高 | 研究能力强的核心作者 |
| **FastWAM 复现/对比** | 作为 baseline 验证 | 可独立完成 |
| **COSMOS 3 平台搭建** | 训练+推理环境 | 工程基础 |

---

## 八、核心一句话

> **跨模态 Attention 本身就是潜空间 Predictor——它在 token 层面做无参数对齐，在梯度层面做有参数耦合。推理时弱耦合（token 值无关），训练时强耦合（梯度互通）。前者用于推理加速，后者用于想象训练。这是 JEPA 在多模态场景下的自然推广。**
