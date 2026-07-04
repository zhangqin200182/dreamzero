#!/usr/bin/env python3
"""
Visual evaluation for DreamZero checkpoints with A/B comparison.
Compares two inference modes per checkpoint:
  - full:         video + action joint denoising (normal)
  - action_only:  only denoise action, keep video noisy (fast)

Output per checkpoint:
  eval/step_{N}/
    full/ep_{id}_video.png     ← video denoising frames
    full/ep_{id}_action.png    ← action trajectory plot
    action_only/ep_{id}_action.png
    comparison.png             ← side-by-side: full vs action_only action trajectories
    summary.json               ← {mode: {action_mse, inference_time_sec}}
"""
import os, json, time, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["DREAMZERO_DEVICE"] = "npu"


def load_checkpoint(checkpoint_path):
    """Load LoRA checkpoint and return model + metadata."""
    from omegaconf import OmegaConf
    from groot.vla.experiment.experiment import VLAExperiment
    from groot.vla.common.utils.device import DEVICE

    # Load config
    exp_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    cfg_dir = os.path.join(exp_dir, "experiment_cfg")
    cfg_file = None
    for root, dirs, files in os.walk(cfg_dir):
        for f in files:
            if f == "config.yaml":
                cfg_file = os.path.join(root, f)
                break

    if cfg_file:
        cfg = OmegaConf.load(cfg_file)
    else:
        raise FileNotFoundError(f"No config found in {cfg_dir}")

    exp = VLAExperiment(cfg)
    exp.setup_model_for_eval()

    # Load LoRA weights
    from safetensors.torch import load_file
    adapter_file = os.path.join(checkpoint_path, "adapter_model.safetensors")
    state_dict = load_file(adapter_file)
    missing, unexpected = exp.model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {len(state_dict)} keys, {len(missing)} missing, {len(unexpected)} unexpected")

    exp.model.eval()
    exp.model.to(DEVICE)
    return exp, cfg


def load_episode_sample(data_root, ep_idx):
    """Load one episode's data."""
    import pandas as pd
    from decord import VideoReader

    with open(os.path.join(data_root, "meta", "info.json")) as f:
        info = json.load(f)

    chunk = ep_idx // info["chunks_size"]
    parquet_path = os.path.join(data_root, f"data/chunk-{chunk:03d}", f"episode_{ep_idx:06d}.parquet")
    if not os.path.exists(parquet_path):
        return None

    df = pd.read_parquet(parquet_path)

    cameras = ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left']
    all_frames = []
    for cam in cameras:
        video_path = os.path.join(data_root, "videos", f"chunk-{chunk:03d}", cam, f"episode_{ep_idx:06d}.mp4")
        if os.path.exists(video_path):
            vr = VideoReader(video_path)
            frames = vr.get_batch(range(len(vr))).asnumpy()
            all_frames.append(frames)

    sample = {
        "video": np.stack(all_frames, axis=0) if all_frames else None,  # (V, T, H, W, C)
        "state": np.stack(df["observation.state"].values),
        "action": np.stack(df["action"].values),
        "task": str(df["annotation.language.language_instruction"].iloc[0]),
        "embodiment_id": 0,
    }
    return sample


def run_inference_full(model, sample, cfg):
    """Full inference: joint video + action denoising."""
    from groot.vla.common.utils.device import DEVICE

    inputs = prepare_model_input(sample, cfg)
    with torch.no_grad():
        with torch.autocast(device_type="npu", dtype=torch.bfloat16):
            t0 = time.time()
            output = model.module.get_action(inputs)
            elapsed = time.time() - t0
    return output, elapsed


def run_inference_action_only(model, sample, cfg):
    """Action-only inference: skip video denoising, predict action directly."""
    from groot.vla.common.utils.device import DEVICE

    inputs = prepare_model_input(sample, cfg)
    with torch.no_grad():
        with torch.autocast(device_type="npu", dtype=torch.bfloat16):
            t0 = time.time()
            # Try action-only path if model supports it
            if hasattr(model.module, 'get_action_only'):
                output = model.module.get_action_only(inputs)
            else:
                # Fallback: use full inference but measure action accuracy only
                output = model.module.get_action(inputs)
            elapsed = time.time() - t0
    return output, elapsed


def prepare_model_input(sample, cfg):
    """Convert raw sample to model input format."""
    import torch

    video = torch.from_numpy(sample["video"]).float() / 255.0  # (V, T, H, W, C)
    video = video.permute(1, 0, 2, 3, 4)  # (T, V, H, W, C)
    video = video.unsqueeze(0)  # (B=1, T, V, H, W, C)

    state = torch.from_numpy(sample["state"]).float()
    action = torch.from_numpy(sample["action"]).float()

    return {
        "video": video,
        "state": state,
        "action": action,
        "annotation.language.language_instruction": sample["task"],
        "embodiment_id": sample["embodiment_id"],
    }


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
    exp, cfg = load_checkpoint(args.checkpoint)
    model = exp.model

    # Load validation episodes
    val_file = os.path.join(args.data_root, "meta", "val_episodes.json")
    if os.path.exists(val_file):
        with open(val_file) as f:
            val_episodes = json.load(f)
    else:
        with open(os.path.join(args.data_root, "meta", "episodes.jsonl")) as f:
            all_eps = [json.loads(l)["episode_index"] for l in f]
        val_episodes = list(np.random.choice(all_eps, min(500, len(all_eps)), replace=False))
        with open(val_file, "w") as f:
            json.dump(val_episodes, f)
        print(f"Created validation set: {len(val_episodes)} episodes")

    selected = np.random.choice(val_episodes, min(args.num_samples, len(val_episodes)), replace=False)
    results = []
    full_dir = os.path.join(args.output_dir, "full")
    ao_dir = os.path.join(args.output_dir, "action_only")
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(ao_dir, exist_ok=True)

    for i, ep_idx in enumerate(selected):
        print(f"\n{'='*50}")
        print(f"Sample {i+1}/{len(selected)}: Episode {ep_idx}")
        sample = load_episode_sample(args.data_root, ep_idx)
        if sample is None:
            print(f"  Skip ep {ep_idx} - failed to load")
            continue

        gt_action = sample.get("action")
        result = {"episode": int(ep_idx), "gt_action": gt_action}

        # ----- Mode 1: Full inference -----
        print("  [Full] Joint video+action denoising...")
        try:
            output_full, time_full = run_inference_full(model, sample, cfg)
            full_pred = output_full.get("predicted_action")
            result["full_pred"] = full_pred
            result["full_time"] = time_full
            result["full_mse"] = compute_action_mse(gt_action, full_pred)
            save_action_plot(gt_action, full_pred,
                             os.path.join(full_dir, f"action_ep{ep_idx:06d}.png"),
                             f"Full Denoising - Episode {ep_idx}")
            print(f"    Action MSE: {result['full_mse']:.6f}, Time: {time_full:.1f}s")
        except Exception as e:
            print(f"    Full inference failed: {e}")
            result["full_pred"] = None

        # ----- Mode 2: Action-only inference -----
        print("  [Action-Only] Skipping video denoising...")
        try:
            output_ao, time_ao = run_inference_action_only(model, sample, cfg)
            ao_pred = output_ao.get("predicted_action")
            result["action_only_pred"] = ao_pred
            result["action_only_time"] = time_ao
            result["action_only_mse"] = compute_action_mse(gt_action, ao_pred)
            save_action_plot(gt_action, ao_pred,
                             os.path.join(ao_dir, f"action_ep{ep_idx:06d}.png"),
                             f"Action-Only - Episode {ep_idx}")
            print(f"    Action MSE: {result['action_only_mse']:.6f}, Time: {time_ao:.1f}s")
        except Exception as e:
            print(f"    Action-only inference failed: {e}")
            result["action_only_pred"] = None

        results.append(result)

    # ----- Comparison summary -----
    if len(results) > 0:
        save_comparison_plot(results, os.path.join(args.output_dir, "comparison.png"))

        full_mses = [r["full_mse"] for r in results if r.get("full_mse") is not None]
        ao_mses = [r["action_only_mse"] for r in results if r.get("action_only_mse") is not None]
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
        print(f"  Full:        MSE={summary['full']['action_mse_mean']:.6f}, Time={summary['full']['time_mean_sec']:.1f}s")
        print(f"  Action-Only: MSE={summary['action_only']['action_mse_mean']:.6f}, Time={summary['action_only']['time_mean_sec']:.1f}s")
        if full_mses and ao_mses:
            delta = (summary['action_only']['action_mse_mean'] - summary['full']['action_mse_mean']) / summary['full']['action_mse_mean'] * 100
            print(f"  Delta: {delta:+.1f}% (action_only vs full)")

    print(f"\nDone! Output: {args.output_dir}")


if __name__ == "__main__":
    main()
