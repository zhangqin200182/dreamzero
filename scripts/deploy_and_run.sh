#!/bin/bash
# ===========================================================================
# DreamZero NPU: 一键部署 + 评测 + 训练 + 监控
#
# 用法 (在 239 容器内):
#   bash deploy_and_run.sh
#
# 或分步执行:
#   bash deploy_and_run.sh deploy    # 仅部署代码
#   bash deploy_and_run.sh smoke     # 冒烟测试
#   bash deploy_and_run.sh train     # 启动训练 (nohup 后台)
#   bash deploy_and_run.sh eval      # 批量评测所有 checkpoint
#   bash deploy_and_run.sh trend     # 生成趋势图
#   bash deploy_and_run.sh all       # 全部执行 (默认)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/workspace/dreamzero"
EVAL_OUTPUT_DIR="${PROJECT_DIR}/eval"
CHECKPOINT_ROOT="/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v2"
STEP="${1:-all}"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { log "ERROR: $*"; exit 1; }

# ---------------------------------------------------------------------------
# 0. 环境检查
# ---------------------------------------------------------------------------
check_env() {
    log "=== 环境检查 ==="
    cd "$PROJECT_DIR" || die "项目目录不存在: $PROJECT_DIR"

    log "Git branch: $(git branch --show-current)"
    log "Git HEAD:   $(git log --oneline -1)"

    # NPU 可用性
    python3 -c "
import torch, torch_npu
c = torch_npu.npu.device_count()
print(f'NPU count: {c}')
for i in range(c):
    p = torch_npu.npu.get_device_properties(i)
    print(f'  NPU {i}: {p.name}, total_memory={p.total_mem/1024**3:.0f} GiB')
" || die "NPU 不可用"

    # 检查 checkpoint 目录
    if [ -d "$CHECKPOINT_ROOT" ]; then
        ckpt_count=$(find "$CHECKPOINT_ROOT" -maxdepth 1 -name "checkpoint-*" -type d | wc -l)
        log "已有 $ckpt_count 个 checkpoint"
        if [ $ckpt_count -gt 0 ]; then
            log "最新: $(find "$CHECKPOINT_ROOT" -maxdepth 1 -name "checkpoint-*" -type d | sort -V | tail -1)"
        fi
    else
        log "Checkpoint 目录不存在，将新建"
    fi

    log "环境检查通过"
}

# ---------------------------------------------------------------------------
# 1. 部署代码
# ---------------------------------------------------------------------------
do_deploy() {
    log "=== 部署代码 ==="
    cd "$PROJECT_DIR"

    # 拉取最新代码
    if git remote get-url fork &>/dev/null; then
        log "从 fork 拉取..."
        git fetch fork feature/npu-fsdp-fix || {
            log "fork fetch 失败，尝试 origin"
            git fetch origin feature/npu-fsdp-fix || die "git fetch 失败"
        }
    else
        git fetch origin feature/npu-fsdp-fix || die "git fetch 失败"
    fi

    # 切换分支
    git checkout feature/npu-fsdp-fix 2>/dev/null || git checkout -b feature/npu-fsdp-fix origin/feature/npu-fsdp-fix
    git pull --ff-only origin feature/npu-fsdp-fix 2>/dev/null || true

    log "部署完成: $(git log --oneline -1)"
}

# ---------------------------------------------------------------------------
# 2. 冒烟测试 (单 checkpoint, 少量样本)
# ---------------------------------------------------------------------------
do_smoke() {
    log "=== 冒烟测试 ==="

    # 找最新的 checkpoint
    local latest_ckpt
    latest_ckpt=$(find "$CHECKPOINT_ROOT" -maxdepth 1 -name "checkpoint-*" -type d | sort -V | tail -1)
    if [ -z "$latest_ckpt" ]; then
        log "没有 checkpoint，跳过冒烟测试 (训练会生成)"
        return 0
    fi

    log "测试 checkpoint: $latest_ckpt"
    cd "$PROJECT_DIR"

    python3 scripts/eval/batch_eval.py \
        --checkpoint_root "$CHECKPOINT_ROOT" \
        --output_root "$EVAL_OUTPUT_DIR" \
        --num_samples 3 \
        --steps "$(basename "$latest_ckpt" | grep -o '[0-9]*')"

    if [ $? -eq 0 ]; then
        log "冒烟测试通过!"
        # 显示结果
        local summary_file="$EVAL_OUTPUT_DIR/step_$(printf '%04d' $(basename "$latest_ckpt" | grep -o '[0-9]*'))/summary.json"
        if [ -f "$summary_file" ]; then
            python3 -c "
import json
s = json.load(open('$summary_file'))
print(f'  Full MSE:        {s[\"full\"][\"action_mse_mean\"]}')
print(f'  Action-Only MSE: {s[\"action_only\"][\"action_mse_mean\"]}')
print(f'  Full Time:       {s[\"full\"][\"time_mean_sec\"]}s')
print(f'  AO Time:         {s[\"action_only\"][\"time_mean_sec\"]}s')
"
        fi
    else
        log "冒烟测试失败 (训练仍然可以启动)"
    fi
}

# ---------------------------------------------------------------------------
# 3. 启动训练 (后台 nohup)
# ---------------------------------------------------------------------------
do_train() {
    log "=== 启动训练 ==="
    cd "$PROJECT_DIR"

    local LOGFILE="$PROJECT_DIR/train_$(date +%Y%m%d_%H%M%S).log"

    # 如果已有训练在跑，检查是否应该 resume
    local latest_ckpt
    latest_ckpt=$(find "$CHECKPOINT_ROOT" -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1 || true)

    local RESUME_FLAG=""
    if [ -n "$latest_ckpt" ]; then
        local step=$(basename "$latest_ckpt" | grep -o '[0-9]*')
        log "检测到已有 checkpoint: step=$step，将 resume"
        RESUME_FLAG="resume_from_checkpoint=true"
    fi

    log "训练日志: $LOGFILE"

    nohup bash scripts/train/droid_16gpu.sh $RESUME_FLAG \
        > "$LOGFILE" 2>&1 &

    local TRAIN_PID=$!
    echo "$TRAIN_PID" > /tmp/dreamzero_train.pid
    log "训练已启动, PID=$TRAIN_PID"
    log "监控: tail -f $LOGFILE"
    log "停止: kill $TRAIN_PID"

    # 等几秒确认正常
    sleep 5
    if kill -0 $TRAIN_PID 2>/dev/null; then
        log "训练进程正常运行"
        log "初始日志:"
        head -20 "$LOGFILE"
    else
        die "训练进程已退出，查看日志: $LOGFILE"
    fi
}

# ---------------------------------------------------------------------------
# 4. 监控训练进度
# ---------------------------------------------------------------------------
do_monitor() {
    log "=== 训练监控 ==="

    local LOGFILE
    LOGFILE=$(ls -t "$PROJECT_DIR"/train_*.log 2>/dev/null | head -1)
    if [ -z "$LOGFILE" ]; then
        die "找不到训练日志"
    fi

    # 检查训练是否还在跑
    local TRAIN_PID
    if [ -f /tmp/dreamzero_train.pid ]; then
        TRAIN_PID=$(cat /tmp/dreamzero_train.pid)
        if kill -0 "$TRAIN_PID" 2>/dev/null; then
            log "训练进程存活 (PID=$TRAIN_PID)"
        else
            log "训练进程已退出 (PID=$TRAIN_PID)"
        fi
    else
        # 尝试通过进程名查找
        TRAIN_PID=$(pgrep -f "groot/vla/experiment/experiment.py" | head -1 || echo "")
        if [ -n "$TRAIN_PID" ]; then
            log "找到训练进程 (PID=$TRAIN_PID)"
        else
            log "未找到运行中的训练进程"
        fi
    fi

    # 最新日志
    echo ""
    echo "=== 最新 30 行日志 ==="
    tail -30 "$LOGFILE"

    # Checkpoint 统计
    echo ""
    echo "=== Checkpoint 统计 ==="
    if [ -d "$CHECKPOINT_ROOT" ]; then
        echo "已有 checkpoints:"
        find "$CHECKPOINT_ROOT" -maxdepth 1 -name "checkpoint-*" -type d | sort -V | while read d; do
            local step=$(basename "$d" | grep -o '[0-9]*')
            local adapter="$d/adapter_model.safetensors"
            if [ -f "$adapter" ]; then
                local size=$(du -h "$adapter" | cut -f1)
                echo "  step=$step  adapter=$size"
            else
                echo "  step=$step  (no adapter yet)"
            fi
        done
    fi

    # NPU 内存
    echo ""
    echo "=== NPU 内存 ==="
    python3 -c "
import torch, torch_npu
for i in range(torch_npu.npu.device_count()):
    mem = torch_npu.npu.memory_stats(i)
    alloc = mem.get('allocated_bytes.all.current', 0) / 1024**3
    reserve = mem.get('reserved_bytes.all.current', 0) / 1024**3
    print(f'  NPU {i}: allocated={alloc:.1f} GiB, reserved={reserve:.1f} GiB')
" 2>/dev/null || echo "  无法获取 NPU 内存信息"
}

# ---------------------------------------------------------------------------
# 5. 批量评测
# ---------------------------------------------------------------------------
do_eval() {
    log "=== 批量评测 ==="
    cd "$PROJECT_DIR"

    # 检查是否有训练在跑 (评测需要 NPU)
    if pgrep -f "groot/vla/experiment/experiment.py" > /dev/null; then
        log "WARNING: 训练正在运行，评测会争抢 NPU 资源"
        log "建议等训练完成或 kill 训练后再评测"
        read -p "继续评测? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "跳过评测"
            return 0
        fi
    fi

    python3 scripts/eval/batch_eval.py \
        --checkpoint_root "$CHECKPOINT_ROOT" \
        --output_root "$EVAL_OUTPUT_DIR" \
        --num_samples 10

    log "评测完成, 结果: $EVAL_OUTPUT_DIR"
}

# ---------------------------------------------------------------------------
# 6. 趋势分析
# ---------------------------------------------------------------------------
do_trend() {
    log "=== 趋势分析 ==="
    cd "$PROJECT_DIR"

    python3 scripts/eval/trend_analysis.py --eval_root "$EVAL_OUTPUT_DIR"

    log "趋势图已生成:"
    ls -la "$EVAL_OUTPUT_DIR"/trend_*.png 2>/dev/null || log "无趋势图产出"
}

# ---------------------------------------------------------------------------
# 7. 全部流程
# ---------------------------------------------------------------------------
do_all() {
    check_env
    do_deploy
    do_smoke
    do_train
    log "训练已启动，监控命令: bash $0 monitor"
    log "训练完成后执行: bash $0 eval && bash $0 trend"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "$STEP" in
    check)    check_env ;;
    deploy)   check_env && do_deploy ;;
    smoke)    do_smoke ;;
    train)    check_env && do_deploy && do_train ;;
    monitor)  do_monitor ;;
    eval)     do_eval ;;
    trend)    do_trend ;;
    all)      do_all ;;
    *)
        echo "Usage: $0 {check|deploy|smoke|train|monitor|eval|trend|all}"
        exit 1
        ;;
esac
