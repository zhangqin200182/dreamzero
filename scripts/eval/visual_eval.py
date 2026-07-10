#!/usr/bin/env python3
"""
Visual evaluation for DreamZero checkpoints with A/B comparison.
Compares two inference modes per checkpoint:
  - full:         video + action joint denoising (normal)
  - action_only:  only denoise action, keep video noisy (fast)

Output per checkpoint:
  eval/step_{N}/
    full/ep_{id}_action.png    ← action trajectory plot
    action_only/ep_{id}_action.png
    comparison.png             ← side-by-side: full vs action_only action trajectories
    summary.json               ← {mode: {action_mse, inference_time_sec}}
"""
import os, sys, json, time, argparse
import numpy as np

os.environ["DREAMZERO_DEVICE"] = "npu"

# torch.compile requires triton drivers only available under torchrun multi-process.
# In single-process eval, disable it globally so that post_initialize()'s
# @torch.compile calls become no-ops.
import torch
torch._dynamo.config.disable = True

# Init distributed (required by model creation — HCCL backend)
import torch.distributed as dist
for k, v in [("MASTER_ADDR", "localhost"), ("MASTER_PORT", "29501"),
             ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]:
    os.environ.setdefault(k, v)
if not dist.is_initialized():
    dist.init_process_group(backend="hccl")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _reset_model_state(ah):
    """Reset inference state so inference starts fresh for a new sample.

    lazy_joint_video_action is stateful — current_start_frame and language
    track position across incremental calls.  Reset them before each sample.
    """
    ah.current_start_frame = 0
    ah.language = None
    ah.clip_feas = None
    ah.ys = None
    ah.kv_cache1 = None
    ah.kv_cache_neg = None
    ah.crossattn_cache = None
    ah.crossattn_cache_neg = None


def load_checkpoint(checkpoint_path):
    """Load LoRA checkpoint and return model + config."""
    from omegaconf import OmegaConf
    from groot.vla.common.utils.device import DEVICE

    # --- Load config ---
    # checkpoint-200 is one level inside the experiment dir
    exp_dir = os.path.dirname(checkpoint_path)
    cfg_dir = os.path.join(exp_dir, "experiment_cfg")
    cfg_file = None
    for root, dirs, files in os.walk(cfg_dir):
        for f in files:
            if f in ("config.yaml", "conf.yaml"):
                cfg_file = os.path.join(root, f)
                break

    if cfg_file:
        cfg = OmegaConf.load(cfg_file)
    else:
        raise FileNotFoundError(f"No config found in {cfg_dir}")

    # --- Instantiate model ---
    from hydra.utils import instantiate
    model = instantiate(cfg.model)
    model.eval()
    model.to(DEVICE)

    # --- Post-initialize ---
    # Moves text_encoder, image_encoder, vae to NPU, sets dtype to bfloat16,
    # and optionally torch.compile's them (no-op since dynamo is disabled).
    model.post_initialize()

    # --- Load LoRA weights ---
    from safetensors.torch import load_file
    adapter_file = os.path.join(checkpoint_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"No adapter_model.safetensors found in {checkpoint_path}")

    state_dict = load_file(adapter_file)
    # Strip .base_layer. prefix (PEFT internal) if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace(".base_layer.", ".")] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded checkpoint: {len(state_dict)} keys, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        print(f"  Missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys (first 5): {unexpected[:5]}")

    return model, cfg


def load_samples_from_dataset(cfg, max_samples=5, seed=42):
    """Load properly formatted samples using the training dataset pipeline."""
    from hydra.utils import instantiate

    ds = instantiate(cfg.train_dataset)
    np.random.seed(seed)
    it = iter(ds)
    samples = []
    for _ in range(max_samples):
        try:
            samples.append(next(it))
        except StopIteration:
            break
    print(f"Loaded {len(samples)} samples from dataset")
    return samples


def run_inference_full(model, sample):
    """Full inference: joint video + action denoising (causal, stateful)."""
    from groot.vla.common.utils.device import DEVICE
    from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer

    ah = model.action_head
    _reset_model_state(ah)

    # Build input dict from transformed sample
    tk = HuggingfaceTokenizer(name="/checkpoints/umt5-xxl", seq_len=512, clean='whitespace')
    txt = sample.get("text", "") or ""
    ids, amask = tk([txt], return_mask=True)
    neg_ids, neg_mask = tk([""], return_mask=True)

    inp = {
        "images": sample["images"],
        "state": sample["state"],
        "action": sample["action"],
        "text": ids,
        "text_attention_mask": amask,
        "text_negative": neg_ids,
        "text_attention_mask_negative": neg_mask,
        "embodiment_id": sample["embodiment_id"],
        "has_real_action": sample.get("has_real_action", True),
        "action_mask": sample.get("action_mask", torch.ones(len(sample["action"]), 32)),
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

    # Truncate state tokens to match model expectation
    if inp["state"].shape[1] > ah.model.num_state_per_block:
        inp["state"] = inp["state"][:, -ah.model.num_state_per_block:]

    with torch.no_grad():
        with torch.autocast(device_type="npu", dtype=torch.bfloat16):
            t0 = time.time()
            output = model.lazy_joint_video_action_causal(inp)
            elapsed = time.time() - t0

    return output, elapsed


def run_inference_action_only(model, sample):
    """Action-only inference: skip video denoising, predict action directly."""
    from groot.vla.common.utils.device import DEVICE
    from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer

    ah = model.action_head
    _reset_model_state(ah)

    # Build input dict (same as full mode)
    tk = HuggingfaceTokenizer(name="/checkpoints/umt5-xxl", seq_len=512, clean='whitespace')
    txt = sample.get("text", "") or ""
    ids, amask = tk([txt], return_mask=True)
    neg_ids, neg_mask = tk([""], return_mask=True)

    inp = {
        "images": sample["images"],
        "state": sample["state"],
        "action": sample["action"],
        "text": ids,
        "text_attention_mask": amask,
        "text_negative": neg_ids,
        "text_attention_mask_negative": neg_mask,
        "embodiment_id": sample["embodiment_id"],
        "has_real_action": sample.get("has_real_action", True),
        "action_mask": sample.get("action_mask", torch.ones(len(sample["action"]), 32)),
    }

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

    if inp["state"].shape[1] > ah.model.num_state_per_block:
        inp["state"] = inp["state"][:, -ah.model.num_state_per_block:]

    # Enable action-only mode: decouple inference noise + skip video denoising
    saved_decouple = ah.config.decouple_inference_noise
    saved_final_noise = ah.config.video_inference_final_noise
    ah.config.decouple_inference_noise = True
    ah.config.video_inference_final_noise = 1.0

    try:
        with torch.no_grad():
            with torch.autocast(device_type="npu", dtype=torch.bfloat16):
                t0 = time.time()
                output = model.lazy_joint_video_action_causal(inp)
                elapsed = time.time() - t0
    finally:
        ah.config.decouple_inference_noise = saved_decouple
        ah.config.video_inference_final_noise = saved_final_noise

    return output, elapsed


def save_action_plot(gt, pred, output_path, title="Action Prediction"):
    """Plot GT vs predicted action trajectories across joint dims."""
    if gt is None or pred is None:
        return

    T_gt, D = min(len(gt), len(pred)), min(gt.shape[1], pred.shape[1], 7)
    fig, axes = plt.subplots(D, 1, figsize=(10, 2 * D), sharex=True)
    if D == 1:
        axes = [axes]

    for d in range(D):
        axes[d].plot(range(T_gt), gt[:T_gt, d], 'b-', label='GT', linewidth=2, alpha=0.7)
        axes[d].plot(range(T_gt), pred[:T_gt, d], 'r--', label='Pred', linewidth=2, alpha=0.7)
        axes[d].set_ylabel(f"Joint {d}")
        axes[d].legend(loc='upper right', fontsize=7)
        axes[d].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def save_comparison_plot(results, output_path):
    """Side-by-side comparison: full vs action_only across all samples."""
    n = len(results)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]

    for i, r in enumerate(results):
        gt = r.get("gt_action")
        full_pred = r.get("full_pred")
        ao_pred = r.get("action_only_pred")

        if gt is None:
            continue

        dim = 0  # Show first joint dim as summary
        T = min(len(gt), len(full_pred or gt), len(ao_pred or gt))
        axes[i].plot(range(T), gt[:T, dim], 'k-', label='GT', linewidth=2)
        if full_pred is not None:
            axes[i].plot(range(T), full_pred[:T, dim], 'b--', label='Full (video+action)', linewidth=1.5, alpha=0.8)
        if ao_pred is not None:
            axes[i].plot(range(T), ao_pred[:T, dim], 'r--', label='Action-only', linewidth=1.5, alpha=0.8)
        axes[i].set_ylabel(f"Ep {r['episode']} Joint 0")
        axes[i].legend(loc='upper right', fontsize=7)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    plt.suptitle("Full vs Action-Only Inference Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def compute_action_mse(gt, pred):
    """Mean squared error between GT and predicted actions."""
    if gt is None or pred is None:
        return float('inf')
    T = min(len(gt), len(pred))
    return float(np.mean((gt[:T] - pred[:T]) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="/data/droid")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare_modes", action="store_true", default=True,
                        help="Run both full and action-only modes for comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {args.checkpoint}...")
    model, cfg = load_checkpoint(args.checkpoint)

    # Load samples from dataset (properly formatted by training pipeline)
    samples = load_samples_from_dataset(cfg, max_samples=args.num_samples, seed=args.seed)
    if len(samples) == 0:
        print("ERROR: No samples loaded from dataset")
        return

    results = []
    full_dir = os.path.join(args.output_dir, "full")
    ao_dir = os.path.join(args.output_dir, "action_only")
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(ao_dir, exist_ok=True)

    for i, sample in enumerate(samples):
        print(f"\n{'='*50}")
        print(f"Sample {i+1}/{len(samples)}")

        gt = sample["action"].float().numpy() if hasattr(sample["action"], 'numpy') else np.asarray(sample["action"])
        result = {"episode": i, "gt_action": gt}

        # ----- Mode 1: Full inference -----
        print("  [Full] Joint video+action denoising...")
        try:
            output_full, time_full = run_inference_full(model, sample)
            full_pred = output_full["action_pred"][0].cpu().float().numpy()
            result["full_pred"] = full_pred
            result["full_time"] = time_full
            result["full_mse"] = compute_action_mse(gt, full_pred)
            save_action_plot(gt, full_pred,
                             os.path.join(full_dir, f"action_ep{i:02d}.png"),
                             f"Full Denoising - Episode {i}")
            print(f"    Action MSE: {result['full_mse']:.6f}, Time: {time_full:.1f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    Full inference failed: {e}")
            result["full_pred"] = None

        # ----- Mode 2: Action-only inference -----
        print("  [Action-Only] Skipping video denoising...")
        try:
            output_ao, time_ao = run_inference_action_only(model, sample)
            ao_pred = output_ao["action_pred"][0].cpu().float().numpy()
            result["action_only_pred"] = ao_pred
            result["action_only_time"] = time_ao
            result["action_only_mse"] = compute_action_mse(gt, ao_pred)
            save_action_plot(gt, ao_pred,
                             os.path.join(ao_dir, f"action_ep{i:02d}.png"),
                             f"Action-Only - Episode {i}")
            print(f"    Action MSE: {result['action_only_mse']:.6f}, Time: {time_ao:.1f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    Action-only inference failed: {e}")
            result["action_only_pred"] = None

        results.append(result)

    # ----- Comparison summary -----
    if len(results) > 0:
        save_comparison_plot(results, os.path.join(args.output_dir, "comparison.png"))

        full_mses = [r["full_mse"] for r in results if r.get("full_mse") is not None and r["full_mse"] != float('inf')]
        ao_mses = [r["action_only_mse"] for r in results if r.get("action_only_mse") is not None and r["action_only_mse"] != float('inf')]
        full_times = [r["full_time"] for r in results if r.get("full_time") is not None]
        ao_times = [r["action_only_time"] for r in results if r.get("action_only_time") is not None]

        summary = {
            "checkpoint": args.checkpoint,
            "num_samples": len(results),
            "full": {
                "action_mse_mean": float(np.mean(full_mses)) if full_mses else None,
                "time_mean_sec": float(np.mean(full_times)) if full_times else None,
            },
            "action_only": {
                "action_mse_mean": float(np.mean(ao_mses)) if ao_mses else None,
                "time_mean_sec": float(np.mean(ao_times)) if ao_times else None,
            },
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*50}")
        print("SUMMARY")
        fmse = summary['full']['action_mse_mean']
        ftime = summary['full']['time_mean_sec']
        amse = summary['action_only']['action_mse_mean']
        atime = summary['action_only']['time_mean_sec']
        print(f"  Full:        MSE={'N/A' if fmse is None else f'{fmse:.6f}'}, Time={'N/A' if ftime is None else f'{ftime:.1f}s'}")
        print(f"  Action-Only: MSE={'N/A' if amse is None else f'{amse:.6f}'}, Time={'N/A' if atime is None else f'{atime:.1f}s'}")
        if full_mses and ao_mses and fmse and amse and fmse > 0:
            delta = (amse - fmse) / fmse * 100
            print(f"  Delta: {delta:+.1f}% (action_only vs full)")

    print(f"\nDone! Output: {args.output_dir}")


if __name__ == "__main__":
    main()
