# DreamZero NPU 训练 + A/B 评测计划

## 原始目标

**核心实验问题：跳过视频去噪，动作预测精度降低多少？**

每个 checkpoint（每 200 步）触发 A/B 评测：

| 模式 | 方法 | 目的 |
|------|------|------|
| Full | video + action joint denoising | 精度基线 |
| Action-Only | 跳过视频去噪，只预测动作 | 推理加速可行性 |

**产出**：每个 checkpoint 的关节轨迹图 + Full vs AO 对比 + MSE 趋势图。

---

## 架构

```
本地  ──SSH──→  113.46.41.54 (跳板机，纯转发)  ──SSH──→  192.168.106.239 (NPU 训练服务器)
                                                              ├── 16 × Ascend 910, 64GB/卡
                                                              ├── 内存 2TB (8 NUMA 节点)
                                                              ├── Docker: dreamzero_train
                                                              ├── 代码: /workspace/dreamzero/ (bind mount)
                                                              ├── 数据: /data/droid/ (bind mount, 1.2TB)
                                                              └── 模型: /checkpoints/ (bind mount, 294GB)
```

---

## 代码修复清单 (6 项)

| # | 修复 | 位置 | 解决的问题 |
|---|------|------|-----------|
| 1 | `torch_npu.npu.set_device(local_rank)` | base.py | FSDP rank 未绑定物理 NPU → optimizer step 崩溃 |
| 2 | `_save` 按 `lora_` key 过滤 | base.py save_model() | `get_peft_model_state_dict` 对 FSDP wrapper 不可用 |
| 3 | `_save_optimizer_and_scheduler` override | base.py | `save_lora_only=true` 时阻止写入 62GB pytorch_model_fsdp.bin |
| 4 | `RESUME_CKPT` 环境变量 | base.py run() | 绕过 Hydra config，直接指定 resume checkpoint 路径 |
| 5 | `resume_from_checkpoint: null` | conf.yaml | Hydra CLI override 可用 |
| 6 | matplotlib 导入保护 | image_utils.py | 无头环境不崩溃 |

**评测脚本**：`scripts/eval/visual_eval.py`, `batch_eval.py`, `trend_analysis.py`

---

## 九项原则

### 1. 所有操作指向 239，非跳板机
所有 SSH 命令通过跳板机到达 239：
```
sshpass ... ssh -o ProxyCommand="sshpass ... ssh -W %h:%p root@113.46.41.54" root@192.168.106.239 "docker exec ..."
```
**绝不**：`ssh root@113.46.41.54 'docker exec ...'`（这会连到跳板机的 Docker）

### 2. 容器重启后验证 fix 存在
```bash
grep set_device /workspace/dreamzero/groot/vla/experiment/base.py
grep save_optimizer /workspace/dreamzero/groot/vla/experiment/base.py
```
缺失则重新 scp 部署。

### 3. 训练时设置 NPU 内存分配器
启动命令必须包含：
```bash
docker exec -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512 ...
```

### 4. SSH 连续失败 3 次即停止
不反复重试不可达的主机。通知用户："跳板机不可达，需要新 IP"。

### 5. Checkpoint < 1GB
每个 checkpoint 只应包含：
- adapter_model.safetensors (~75MB)
- trainer_state.json (~KB)
- optimizer.bin (~830MB)
- scheduler.pt (~KB)

**不应包含**：pytorch_model_fsdp.bin (62GB)

### 6. 不修改配置文件的缩进/sed
配置修改通过本地文件 → scp → docker exec tee 部署。避免 sed 截断。

### 7. 周期性监控，失败立即调查根因
- 每 10 分钟检查：进程数、SIGKILL/NPU OOM/Error、当前 step、磁盘空间
- 失败后**不盲目重启**：先看日志中完整错误栈 → 定位根因 → 修复 → 部署 → resume

### 8. 必须从最近 checkpoint resume
训练失败后**绝不从头训练**。设置 `RESUME_CKPT=/path/to/checkpoint-NNN` 从最近 checkpoint 恢复。
```
step 0 ──→ 200 ──→ 400 ──→ 600 ──→ ...
  ↓ 崩溃     ↓ resume  ↓ resume  ↓ resume
  从头来     从200     从400     从600
```

### 9. Checkpoint 保存/加载代码质量
**历史 6 个 Bug 不再重现**：

| Bug | 根因 | 修复 |
|-----|------|------|
| 保存时 `peft_config` AttributeError | FSDP wrapper 被传给 PEFT | 按 key 名过滤 `lora_` |
| checkpoint 63GB | `_save_optimizer_and_scheduler` 绕过 save_model | override 该方法 |
| resume 找不到 checkpoint | Hydra 不识别的 key | 改 RESUME_CKPT 环境变量 |
| resume 时 FileNotFoundError | Docker overlay 16 进程并发读 | checkpoint 放独立目录 |
| resume 被静默跳过 | `train()` override 不触发 auto-detect | 显式设 RESUME_CKPT |
| 磁盘满保存失败 | 63GB × 6 = 378GB | 修复后每个 < 1GB |

**Checkpoint 检查清单**：
- [ ] 保存后验证文件数和大小（< 1GB）
- [ ] Resume 前验证文件存在且可读
- [ ] Resume 后验证 global_step 正确（非 0）
- [ ] 不使用 `get_peft_model_state_dict`
- [ ] `save_lora_only` 同时拦截 `_save` 和 `_save_optimizer_and_scheduler`

---

## 执行流程

### Phase 1: 部署验证
1. 确认 ProxyCommand 到 239 通路
2. 检查容器状态，如停则 `docker start`
3. 验证 base.py fix（set_device + save_optimizer）
4. 缺失则 scp 部署
5. 清理旧 output 目录

### Phase 2: 训练
6. 启动训练（从头开始，PYTORCH_NPU_ALLOC_CONF 已设）
7. 验证 17→33 进程
8. 每 10 分钟监控：进程数、SIGKILL/NPU OOM、进度
9. checkpoint-200 后验证：大小 < 1GB，适配器可加载

### Phase 3: 评测
10. 每个 checkpoint 产出后，停训练
11. 运行 `batch_eval.py` 做 A/B 对比（10 样本）
12. 运行 `trend_analysis.py` 产出 MSE vs step 趋势图
13. Resume 训练，继续到下一个 checkpoint
14. 累积 4+ checkpoint 后做完整趋势分析

### Phase 4: A/B 对比分析
15. 汇总所有 checkpoint 的 Full vs AO MSE
16. 判断：AO 精度损失是否可接受
17. 产出最终报告

---

## 当前状态

| 项目 | 状态 |
|------|------|
| 代码修复 | ✅ 6 项全部完成，本地 git 已推送 |
| 评测脚本 | ✅ visual_eval.py / batch_eval.py / trend_analysis.py |
| 239 容器 | ✅ 运行正常，稳定 |
| 239 fix 部署 | ✅ set_device=1, save_optimizer=2 |
| 239 训练 | ⚪ 未启动 |
