# 基于 Attention 路由的下一代机器人世界模型架构

> 基于对 JEPA、DreamZero、FastWAM、COSMOS 3、π₀ 的分析
> 2026-08-11

---

## 摘要

**COSMOS 3 证明了训练应该在一起（共享参数 = 语义/物理/视觉知识迁移到 Action）。FastWAM 证明了推理应该分开（Expert 分离 = 多频率 + 无有害干扰 + 4× 加速）。DreamZero 实验揭示了推理弱耦合和训练强耦合是 Attention 路由的通用属性。三者共同指向同一个目标架构：训练时共享 Attention 实现知识迁移，推理时 KV cache 分离实现多频率执行。**

本文梳理了从现有架构中识别 Attention 路由范式、揭示其耦合属性、推导目标架构、设计先导实验的完整逻辑链。

---

## 一、起点：两个关键观察

### 1.1 COSMOS 3 的全共享：训练在一起的价值

COSMOS 3 的 AR Reasoner 和 DiT Generator 使用**同一组 Transformer blocks、同一组 Q/K/V/FFN 权重**，仅 attention mask 不同（causal vs full）。这个设计不是偶然的。共享参数在训练中产生四个层面的正迁移：

1. **语言理解 → 动作指令落地**：AR 训练的指令理解能力通过共享 Q 投影矩阵直接用于 Action 生成
2. **物理推理 → 动作合理性约束**：Reasoner 的物理常识（VideoPhy-2 SFT）通过共享 FFN 约束 Policy 输出
3. **视觉理解 → 更好的状态表征**：Reasoner 的视觉能力通过共享视觉塔和 Transformer 编码直接迁移
4. **Action CoT → 推理与生成的桥梁**：语言化动作推理通过共享权重加速连续动作生成

配方证据：Policy-DROID 训练从完整的 Cosmos3-Nano omni-checkpoint 启动，而非 Generator-only 子集。

**结论：全共享架构的训练知识迁移是真实且显著的。这是我们目标架构必须继承的能力。**

### 1.2 FastWAM 的 Expert 分离：推理分开的价值

FastWAM 将 Video Expert 和 Action Expert 分离为独立的 Q/K/V/FFN，仅通过 Joint Self-Attention 在每层进行跨模态路由。推理时：

- `prefill_video_cache`：Video tokens 一次 forward → 缓存 30 层 K/V
- `forward_action_with_video_cache`：Action 每步去噪仅 forward Action Expert + O(1) 读取 video K/V

效果：精度不变，延迟 4× 降低（190ms vs 810ms）。**这是多频率分离推理的完整实现。**

**结论：Expert 分离架构的推理性能优势是真实且显著的。这是我们目标架构必须采用的推理策略。**

---

## 二、洞察：Attention 路由是统一的底层范式

### 2.1 四个架构，同一个机制

 | DreamZero | FastWAM | COSMOS 3 | π₀ |
|---|---|---|---|
| **Expert 分离** | 无（全共享 DiT） | Video + Action | 无（全共享 blocks） | LLM + Action |
| **跨模态交互** | Token 混排 Attention | Joint Self-Attention | Mode Switch Attention | Joint Self-Attention |
| **主干类型** | DiT | DiT | MoT（AR+DiT） | LLM（AR） |
| **团队/时间** | GEAR Lab / 2025 | 清华&上海AI Lab / 2026.3 | NVIDIA / 2026.5 | Physical Intelligence / 2025 |

四个独立团队、四种不同主干、三个不同时间点，不约而同走向了同一个底层机制：**Attention 层跨模态信息路由**——不同模态的 Expert 在 Attention 层通过 Q·K^T 内积进行无参数路由，Attention 结束后各回各自表示空间。这是独立于 JEPA（潜空间对齐）的第二种跨模态交互范式。

### 2.2 Expert 分离程度：连续谱而非二元

四个架构并非同质的"分离"，而是落在一条连续谱上：

```
全共享 ←──────────────────────────────────────────→ 全分离

COSMOS 3          DreamZero        FastWAM            π₀
同一组 blocks      全共享 DiT        独立 Q/K/V/FFN     独立 Q/K/V/FFN
AR+DiT 共享       仅 token 区分      2 Expert, 30 层     2 Expert, 18 层
```

**谱上每个位置的架构都做出了权重共享 vs 推理分离的 trade-off。** 左端：训练迁移强但推理性能差。右端：推理性能好但不知道训练迁移能否保留。

---

## 三、关键属性：推理弱耦合与训练强耦合

### 3.1 DreamZero 实验揭示的耦合属性

DreamZero 位于谱的最左端（全共享 DiT），是耦合最紧的架构。如果在此架构中观察到弱耦合，则 Expert 分离架构中更应成立。我们在 DreamZero 上完成了 7 组因果实验：

**推理弱耦合（实验 A-C）**：

| 实验 | 方法 | 结论 |
|------|------|------|
| A: flow_pred 噪声 | σ=0~1.0 注入 | Action 精度 ±0.1，**完全不受影响** |
| B: video latent 替换 | 纯噪声/全零 | Action **反而改善 13%**（有害干扰） |
| C: 第一帧扰动 | 换 episode/旋转/遮挡 | **0% 退化**；全黑 ~12%；随机 ~21% |

**统一结论**：视频输出质量与动作精度无因果关联。视频的贡献是二值的：token 在场（ON）或不在场（AO，退化 +100%）。

**训练强耦合（实验 D-E，在共享参数架构中）**：

| 实验 | 方法 | 结论 |
|------|------|------|
| D: 单步梯度 | 仅 backward video_loss 一步 | Action 改变 **-58.8%** |
| E: 100 步优化 | 纯视频优化 vs 联合训练 | 纯视频 **-43.7%**，联合 **-85.5%** |

**统一结论**：Video loss 的梯度通过共享权重传播到 Action。纯视频优化无需任何 Action 标注即可将 Action loss 减半。

### 3.2 有害干扰假说（实验 B 的代码级解释）

实验 B：全共享架构中 random latent 替换 video → action 改善 13%。DreamZero 的 Video 使用 `Beta(3,1)` 噪声分布，Action 使用 `Uniform` 分布。共享 Q/K/V 权重需要同时服务两种不同去噪动态 → 相互冲突 → video 对 action 产生有害表示偏移。Expert 分离架构预期不存在此问题（各 Expert 独立噪声调度）。

**如果假说成立，Expert 分离不是可选的优化——是防止模态间有害干扰的必要设计。**

---

## 四、目标架构：训练共享 Attention + 推理 KV cache 分离

### 4.1 架构设计

将 COSMOS 3 的训练迁移优势（2.1 节）和 FastWAM 的推理分离优势（1.2 节）结合在同一个架构中：

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
推理: LLM 1 次 → K_l 缓存 → Video 1 次 → K_v 缓存 → Action N 步（仅 Action Expert）
      多频率执行: LLM ~1Hz / Video ~10Hz / Action ~50Hz
```

### 4.2 与现有架构的对比

| | COSMOS 3 | FastWAM | π₀ | **三 Expert** |
|---|---|---|---|---|
| **训练知识迁移** | ✓（共享参数） | 未知（P0） | ✓（共享 Attention，LLM→Action） | **✓（共享 Attention）** |
| **推理分离** | ✗（全共享） | ✓（Video KV cache） | ✓（LLM KV cache） | **✓（LLM + Video KV cache）** |
| **有害干扰** | 有（全共享） | 预期无 | 预期无 | **预期无** |
| **多频率执行** | ✗ | ✓（2 Expert） | ✓（2 Expert） | **✓（3 Expert，语义→视觉→动作）** |
| **RL 闭环** | 基础设施有，无 GRPO | 有 FD，缺语义 reward | 无世界模型 | **完整闭环** |

### 4.3 Expert 分离程度的选择

三 Expert 位于谱的右端（Expert 分离）。这个选择的依据：

- COSMOS 3 的全共享（左端）：训练迁移好，但推理性能差 + 有害干扰
- 左端的好处（训练迁移）通过共享 Attention 保留——不需要共享参数
- 左端的代价（性能/有害干扰）通过 Expert 分离消除
- **最优位置：共享 Attention（保留训练迁移）+ Expert 分离（消除代价）**

训练耦合在 Expert 分离架构中的强度由 FastWAM F5 实验验证。如果 F5 成立 → 最优位置确认。如果 F5 不成立 → 需要在谱上右移（更多共享）以保留训练迁移。

---

## 五、先导实验：在现有平台上验证关键假设

### 5.1 FastWAM：多频率 + 梯度耦合 + 语义验证

FastWAM 已有 Expert 分离 + Video KV cache，不需新架构即可验证核心假设。

| 实验 | 验证问题 | 对应目标 |
|------|---------|---------|
| **F1 ★★★** | 分离推理 vs 联合推理的 Action 精度 | 目标 3（多频率精度） |
| **F2 ★★★** | K/V 缓存持久性（连续 50 步不刷新） | 目标 3（频率鲁棒性） |
| **F3 ★★** | 视频 K/V 刷新策略消融 | 目标 3（刷新策略） |
| **F5 ★★★** | **仅 backward video loss → Action 变化？** | **决定目标 4（RL 闭环）的可行性** |
| **F6 ★★** | Expert 分离中 video latent 替换 → Action？ | 目标 1（有害干扰验证） |
| **F7 ★★★** | FastWAM + 冻结 AR LLM → 三 Expert block-causal 推理 | 目标 3（三层多频率） |
| **F8 ★★★** | 共享 Attention 训练 vs 独立训练 vs T5 基线 → Action 精度 | AR LLM 训练迁移是否必要？ |

F5 是整个项目的 P0——决定训练强耦合能否泛化到 Expert 分离架构。F8 验证 AR LLM 的共享 Attention 训练是否优于独立训练 + 文本注入。

### 5.2 COSMOS 3：想象 RL 先导

COSMOS 3 是目前唯一具备完整闭环基础设施的平台（Policy + Forward Dynamics + Reasoner）。

| 实验 | 验证问题 |
|------|---------|
| **P3a ★★★** | GRPO 在 WAM 上的工程可行性（N=4 采样，Reasoner reward，200 步） |
| **P3b ★★★** | Reasoner 做机器人任务成功检测的准确率 |
| **P3c ★★** | GRPO vs FM SFT 的 Action 精度对比（平行确认 DreamZero 实验 E） |

### 5.3 平台分工

| 平台 | 角色 | 不依赖 |
|------|------|--------|
| **DreamZero** | 全共享属性基线（已完成） | — |
| **FastWAM** | 多频率 + P0 + 语义训练迁移验证 | COSMOS 3 |
| **COSMOS 3** | 想象 RL 先导 | FastWAM F5 |

FastWAM F5 和 COSMOS 3 GRPO 可以并行推进。Phase 4 汇合。

---

## 六、研究路线图

```
Phase 1（已完成）: DreamZero 先导实验 → 全共享属性画像 + 耦合机制发现

Phase 2（4-6 周）: FastWAM 先导实验（F1-F8）+ COSMOS 3 GRPO 先导（P3a-c）
  → 并行推进，不互相依赖

Phase 3（取决于 F5）:
  → F5 成立: 三 Expert 原型搭建 + 全模态能力验证 + 多频率验证
  → F5 不成立: 以已完成的实验撰写论文（范式识别 + 弱耦合 + Expert 分离优势）

Phase 4（取决于 F5 + P3a）:
  → 两者均成立: COSMOS 3 GRPO + 三 Expert 架构 → 完整想象 RL 验证
  → 仅 F5 成立: 论文 = 机制分析 + 目标架构 + 先导实验
  → 均不成立: 论文 = 范式识别 + 属性发现 + 分离优势论证
```

---

## 七、总结

> COSMOS 3 证明了训练在一起的价值（知识迁移），FastWAM 证明了推理分开的价值（性能 + 无害干扰），DreamZero 实验揭示了 Attention 路由的耦合属性。三者共同建立了目标架构的理论基础：训练时共享 Attention 实现知识迁移，推理时 KV cache 分离实现多频率执行。FastWAM F5 决定这个架构的最终天花板。
