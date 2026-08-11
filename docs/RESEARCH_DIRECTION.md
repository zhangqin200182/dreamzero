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

**这个谱对架构选择有直接影响**：

- **偏左（共享）**：参数效率高，训练强耦合（梯度通过共享参数传播）。但存在有害干扰风险——我们在 DreamZero 上观察到全共享架构中 video denoising 对 action 产生有害表示偏移。
- **偏右（分离）**：推理可分离（弱耦合），各模态独立优化。但训练耦合的强度未知——梯度仅通过 Attention score 矩阵中的跨模态项传播，路径间接。
- **最优位置在哪里？** 目前左侧（COSMOS 3、DreamZero）已验证训练强耦合但存在有害干扰风险，右侧（FastWAM、π₀）已实现推理分离但训练耦合的泛化性未知。**最优解可能不在极端——但 COSMOS 3 在左端的事实意味着它也无法避免全共享的有害干扰风险。**

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

## 三、实验发现：Attention 路由的耦合属性

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

## 四、未来架构推测

基于已有证据，Attention 路由路线下的最优架构应具备：

**1. 适当的 Expert 分离**：全共享架构（COSMOS 3 和 DreamZero 都在谱的左端）存在有害干扰风险（实验 B）。如果 P2 在 Expert 分离架构中验证了无害干扰，则最优位置必然在谱的右侧移动。**这意味着 COSMOS 3 的全共享设计可能不是最优解——它无法避免我们在 DreamZero 上观察到的模态间有害干扰。**

**2. 双模统一与 Expert 分离的组合**：COSMOS 3 证明了双模统一可以在全共享架构中实现。未来的问题是：能否在 Expert 分离架构中实现双模统一？即 AR Reasoner 和 DiT Generator 各自独立的 Expert，通过 Joint Attention 交互，同时支持多频率分离执行。

**3. 多频率执行**：利用推理弱耦合，LLM ~1Hz、Video ~10Hz、Action ~50Hz。FastWAM 的 video KV cache 和 π₀ 的 prefix KV cache 已验证可行。

**4. COSMOS 3 的核心价值**：它是目前唯一跨越"Expert 分离"和"主干类型"两个维度的架构，也是最接近"最优未来架构"的现有实现。理解它在连续谱上的确切位置至关重要。

---

## 五、研究路线图

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

## 六、总结

> 机器人世界模型的跨模态交互有两种范式：JEPA 的潜空间对齐，和 WAM/VLA 的 Attention 路由。后者已被四个架构实现——COSMOS 3 和 DreamZero 位于谱的左端（全共享），FastWAM 和 π₀ 位于谱的右端（Expert 分离）。全共享架构存在有害干扰风险（实验 B），Expert 分离架构的训练耦合泛化性待验证（P0）。未来架构的核心问题是：在 Attention 路由的框架下，双模统一（VLM+DiT）能否在 Expert 分离架构中实现？最优的 Expert 分离程度在哪里？这是我们正在回答的问题。
