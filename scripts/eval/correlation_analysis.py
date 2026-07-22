#!/usr/bin/env python3
"""
Video-Action Correlation Analysis

核心问题: 更好的视频预测能力是否与更好的动作预测能力相关？
→ 如果正相关: RL 优化视频生成质量 → 共享 DiT 表示改进 → Action 精度提升
→ 如果无相关: 改进视频生成对 Action 无帮助，RL 方向不可行

方法:
  对每个 checkpoint，对 N 个样本各跑一次训练前向传播 (teacher forcing)，
  收集 per-sample dynamics_loss (视频噪声预测 MSE) 和 action_loss (动作噪声预测 MSE)，
  计算 Pearson 和 Spearman 相关系数。

  训练前向路径: model.forward(inp) → {loss, dynamics_loss, action_loss}
  batch_size=1 时这些标量就是 per-sample 值。
"""
import os, sys, json, time, argparse
import numpy as np

os.environ["DREAMZERO_DEVICE"] = "npu"
sys.path.insert(0, "/workspace/dreamzero")

import torch
import torch.distributed as dist

# torch.compile requires triton drivers only available under torchrun multi-process.
# In single-process eval, disable it globally.
torch._dynamo.config.disable = True

# Use a unique port from env or default, to allow sequential runs in same container
_MASTER_PORT = os.environ.get("CORR_MASTER_PORT", "29505")
for k, v in [("MASTER_ADDR", "localhost"), ("MASTER_PORT", _MASTER_PORT),
             ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]:
    os.environ[k] = v  # override, not setdefault — we want this specific port

if not dist.is_initialized():
    dist.init_process_group(backend="hccl")

from scipy import stats
from omegaconf import OmegaConf
from hydra.utils import instantiate
from safetensors.torch import load_file
from groot.vla.common.utils.device import DEVICE


def find_config(checkpoint_dir):
    """Find config file near a checkpoint directory."""
    search_paths = [
        os.path.join(checkpoint_dir, "experiment_cfg"),                        # inside checkpoint
        os.path.join(os.path.dirname(checkpoint_dir), "experiment_cfg"),       # parent dir
        "/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v3/experiment_cfg", # v3 fallback
        "/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v2/experiment_cfg", # v2 fallback
        "/workspace/dreamzero/groot/vla/configs",                              # project default
    ]
    for cfg_dir in search_paths:
        if os.path.isdir(cfg_dir):
            for root, dirs, files in os.walk(cfg_dir):
                for f in files:
                    if f in ("conf.yaml", "config.yaml"):
                        return os.path.join(root, f)
    raise FileNotFoundError(f"No config found near {checkpoint_dir}. Searched: {search_paths}")


def compute_correlation(x, y, label_x="dynamics_loss", label_y="action_loss"):
    """Compute Pearson and Spearman correlation between two arrays."""
    x, y = np.asarray(x), np.asarray(y)
    # Remove NaN/inf
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return {"n": n, "error": "too few samples"}

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    return {
        "n": n,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        f"{label_x}_mean": float(np.mean(x)),
        f"{label_x}_std": float(np.std(x)),
        f"{label_y}_mean": float(np.mean(y)),
        f"{label_y}_std": float(np.std(y)),
        f"{label_x}_min": float(np.min(x)),
        f"{label_x}_max": float(np.max(x)),
        f"{label_y}_min": float(np.min(y)),
        f"{label_y}_max": float(np.max(y)),
    }


def main():
    parser = argparse.ArgumentParser(description="Video-Action Correlation Analysis")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir with adapter_model.safetensors")
    parser.add_argument("--output_dir", default="eval/correlation", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset sampling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load config ---
    cfg_file = find_config(args.checkpoint)
    print(f"Config: {cfg_file}")
    cfg = OmegaConf.load(cfg_file)

    # --- Instantiate model ---
    print("Loading model...")
    model = instantiate(cfg.model)
    model.eval()
    model.to(DEVICE)

    ah = model.action_head
    ah.post_initialize()
    print("post_initialize done")

    # --- Load LoRA weights ---
    adapter_file = os.path.join(args.checkpoint, "adapter_model.safetensors")
    sd = load_file(adapter_file)
    # Strip .base_layer. prefix (added by PEFT save)
    stripped_sd = {}
    for k, v in sd.items():
        new_k = k.replace(".base_layer.", ".") if ".base_layer." in k else k
        stripped_sd[new_k] = v
    missing, unexpected = model.load_state_dict(stripped_sd, strict=False)
    lora_keys = [k for k in stripped_sd if "lora_" in k]
    print(f"Loaded LoRA: {len(stripped_sd)} keys ({len(lora_keys)} lora), "
          f"{len(missing)} missing, {len(unexpected)} unexpected")

    # --- Load samples ---
    print(f"Loading {args.num_samples} samples...")
    ds = instantiate(cfg.train_dataset)
    it = iter(ds)
    samples = [next(it) for _ in range(args.num_samples)]
    print(f"Loaded {len(samples)} samples")

    # --- Per-sample evaluation ---
    dynamics_losses = []
    action_losses = []
    losses = []
    errors = 0

    from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer

    print(f"\nEvaluating {len(samples)} samples...")
    t_start = time.time()

    for i, s in enumerate(samples):
        try:
            # Build input dict from transformed sample (same as eval_ab.py)
            tk = HuggingfaceTokenizer(name="/checkpoints/umt5-xxl", seq_len=512, clean='whitespace')
            txt = s.get("text", "") or ""
            ids, amask = tk([txt], return_mask=True)
            neg_ids, neg_mask = tk([""], return_mask=True)

            inp = {
                "images": s["images"],
                "state": s["state"],
                "action": s["action"],
                "text": ids,
                "text_attention_mask": amask,
                "text_negative": neg_ids,
                "text_attention_mask_negative": neg_mask,
                "embodiment_id": s["embodiment_id"],
                "has_real_action": s.get("has_real_action", True),
                "action_mask": s.get("action_mask", torch.ones(len(s["action"]), 32)),
            }

            # Move to device, add batch dim
            _skip_unsqueeze = {"text", "text_attention_mask", "text_negative", "text_attention_mask_negative"}
            for k in list(inp.keys()):
                v = inp[k]
                if isinstance(v, torch.Tensor):
                    inp[k] = v.to(DEVICE) if k in _skip_unsqueeze else v.unsqueeze(0).to(DEVICE)
                elif isinstance(v, np.ndarray):
                    if v.ndim == 0:
                        inp[k] = torch.tensor([v.item()], device=DEVICE)
                    else:
                        inp[k] = torch.from_numpy(v).unsqueeze(0).to(DEVICE)
                elif isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_)):
                    inp[k] = torch.tensor([int(v) if isinstance(v, (bool, np.bool_)) else v], device=DEVICE)

            # Training forward expects state tokens to match video frames:
            #   (num_latent_frames - 1) / num_state_tokens == num_frame_per_block / num_state_per_block
            # The dataset may provide a variable number of state tokens.
            # We need to keep the state shape as-is (no truncation) for training forward.
            # The training forward handles the shape relationship internally.

            # Drop the last action_timestep entry if it was added by the transform
            # (some transforms add a final dummy timestep that doesn't match)

            # Run training forward pass (teacher forcing, batch_size=1)
            with torch.no_grad():
                out = model.forward(inp)

            d_loss = float(out["dynamics_loss"].cpu())
            a_loss = float(out["action_loss"].cpu())
            t_loss = float(out["loss"].cpu())

            dynamics_losses.append(d_loss)
            action_losses.append(a_loss)
            losses.append(t_loss)

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(samples)}] d_loss={d_loss:.6f} a_loss={a_loss:.6f} "
                      f"({rate:.1f} samples/s)")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Sample {i} ERROR: {e}")

    elapsed = time.time() - t_start
    print(f"\nDone: {len(dynamics_losses)} samples in {elapsed:.1f}s "
          f"({len(dynamics_losses)/elapsed:.1f} samples/s), {errors} errors")

    if len(dynamics_losses) < 10:
        print("ERROR: Too few successful samples for correlation analysis")
        dist.destroy_process_group()
        return

    # --- Correlation Analysis ---
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS: Video Loss vs Action Loss")
    print("="*60)

    result = compute_correlation(dynamics_losses, action_losses,
                                 label_x="dynamics_loss", label_y="action_loss")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Interpretation
    print("\n--- Interpretation ---")
    if "pearson_r" in result:
        r = result["pearson_r"]
        p = result["pearson_p"]
        if p < 0.05:
            if r > 0.3:
                print(f"✓ SIGNIFICANT POSITIVE correlation (r={r:.3f}, p={p:.4f})")
                print("  → Better video prediction correlates with better action prediction.")
                print("  → RL on video generation is LIKELY to improve action prediction via shared DiT.")
            elif r > 0.1:
                print(f"~ WEAK positive correlation (r={r:.3f}, p={p:.4f})")
                print("  → Marginal relationship. RL on video MAY help action, but effect is small.")
            elif r > -0.1:
                print(f"≈ NO correlation (r={r:.3f}, p={p:.4f})")
                print("  → Video and action prediction quality are independent.")
                print("  → RL on video generation is UNLIKELY to improve action prediction.")
            else:
                print(f"✗ NEGATIVE correlation (r={r:.3f}, p={p:.4f})")
                print("  → Better video prediction correlates with WORSE action prediction!")
                print("  → RL on video generation may DEGRADE action prediction.")
        else:
            print(f"≈ NOT statistically significant (r={r:.3f}, p={p:.4f})")
            print("  → Need more samples to determine relationship.")

    # Per-checkpoint summary
    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": len(dynamics_losses),
        "num_errors": errors,
        "correlation": result,
        "per_sample": [
            {"dynamics_loss": float(d), "action_loss": float(a), "loss": float(l)}
            for d, a, l in zip(dynamics_losses, action_losses, losses)
        ],
    }

    # Save
    ckpt_name = os.path.basename(args.checkpoint.rstrip("/"))
    out_file = os.path.join(args.output_dir, f"{ckpt_name}_correlation.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_file}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
