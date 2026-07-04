#!/usr/bin/env python3
"""
Visual evaluation script for DreamZero checkpoints.
For each validation sample, saves:
  - Video denoising comparison (original, noisy, denoised frames)
  - Action prediction vs ground truth trajectory plot

Usage:
  python scripts/eval/visual_eval.py \
    --checkpoint /checkpoints/dreamzero_droid_npu_16gpu_bs32/checkpoint-1000 \
    --data_root /data/droid \
    --output_dir ./eval/step_1000 \
    --num_samples 5
"""
import os, json, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Must set before any torch import in the training code
os.environ["DREAMZERO_DEVICE"] = "npu"

from omegaconf import OmegaConf
from groot.vla.experiment.experiment import VLAExperiment
from groot.vla.common.utils.device import DEVICE, empty_cache


def load_checkpoint(checkpoint_path):
    """Load a saved LoRA checkpoint."""
    cfg_path = os.path.join(os.path.dirname(checkpoint_path), "..", "experiment_cfg")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "experiment_cfg")

    # Find the saved config
    cfg_file = None
    for root, dirs, files in os.walk(cfg_path):
        for f in files:
            if f.endswith('.yaml') or f.endswith('.json'):
                cfg_file = os.path.join(root, f)
                break

    if cfg_file and cfg_file.endswith('.json'):
        cfg = OmegaConf.load(cfg_file)
    else:
        # Use default training config
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(version_base=None, config_dir=os.path.join(cfg_path, ".hydra")):
            cfg = compose(config_name="config")

    # Create experiment and load model
    exp = VLAExperiment(cfg)
    exp.setup_model_for_eval()

    # Load checkpoint weights
    from safetensors.torch import load_file
    state_dict = load_file(os.path.join(checkpoint_path, "adapter_model.safetensors"))
    exp.model.load_state_dict(state_dict, strict=False)

    exp.model.eval()
    exp.model.to(DEVICE)
    return exp


def run_inference(model, sample_data):
    """Run full denoising inference and return results."""
    with torch.no_grad():
        with torch.autocast(device_type="npu", dtype=torch.bfloat16):
            result = model.module.get_action(sample_data)
    return result


def save_video_comparison(frames_orig, frames_noisy, frames_denoised, output_path):
    """Save side-by-side video frame comparison."""
    T = min(len(frames_orig), len(frames_denoised))
    fig, axes = plt.subplots(3, min(T, 6), figsize=(12, 6))

    for i in range(min(T, 6)):
        if frames_orig is not None and i < len(frames_orig):
            axes[0, i].imshow(frames_orig[i])
        axes[0, i].set_title(f"Original t={i}")
        axes[0, i].axis('off')

        if frames_noisy is not None and i < len(frames_noisy):
            axes[1, i].imshow(frames_noisy[i])
        axes[1, i].set_title(f"Noisy t={i}")
        axes[1, i].axis('off')

        if frames_denoised is not None and i < len(frames_denoised):
            axes[2, i].imshow(frames_denoised[i])
        axes[2, i].set_title(f"Denoised t={i}")
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def save_action_comparison(gt_action, pred_action, output_path, action_dim_names=None):
    """Plot ground truth vs predicted action trajectories."""
    if gt_action is None or pred_action is None:
        return

    T_gt, D = gt_action.shape
    T_pred, D_pred = pred_action.shape

    n_dims = min(D, 7)  # Show first 7 dims (joint positions)
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 2 * n_dims), sharex=True)

    for d in range(n_dims):
        ax = axes[d] if n_dims > 1 else axes
        ax.plot(range(T_gt), gt_action[:, d], 'b-', label='GT', linewidth=2, alpha=0.7)
        ax.plot(range(T_pred), pred_action[:, d], 'r--', label='Pred', linewidth=2, alpha=0.7)
        label = action_dim_names[d] if action_dim_names else f"Dim {d}"
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    plt.suptitle("Action Prediction vs Ground Truth", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default="/data/droid")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    exp = load_checkpoint(args.checkpoint)
    model = exp.model

    # Load validation samples (use fixed seed for reproducibility)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load episodes from validation subset
    val_episodes_file = os.path.join(args.data_root, "meta", "val_episodes.json")
    if os.path.exists(val_episodes_file):
        with open(val_episodes_file) as f:
            val_episodes = json.load(f)
    else:
        # Fallback: use random episodes
        with open(os.path.join(args.data_root, "meta", "episodes.jsonl")) as f:
            all_eps = [json.loads(l)["episode_index"] for l in f]
        val_episodes = list(np.random.choice(all_eps, min(500, len(all_eps)), replace=False))
        # Save for reproducibility
        with open(val_episodes_file, "w") as f:
            json.dump(val_episodes, f)
        print(f"Created validation set: {len(val_episodes)} episodes")

    selected = np.random.choice(val_episodes, min(args.num_samples, len(val_episodes)), replace=False)

    results = []
    for i, ep_idx in enumerate(selected):
        print(f"\nSample {i+1}/{len(selected)}: episode {ep_idx}")

        # Load data for this episode (simplified - uses the dataloader)
        sample = load_episode_sample(args.data_root, ep_idx)
        if sample is None:
            print(f"  Skipping episode {ep_idx} - failed to load")
            continue

        # Run inference
        try:
            output = run_inference(model, sample)
        except Exception as e:
            print(f"  Inference failed: {e}")
            continue

        # Save video comparison
        video_path = os.path.join(args.output_dir, f"video_ep{ep_idx:06d}.png")
        frames_orig = sample.get("video_orig")  # (T, H, W, C) numpy
        frames_denoised = output.get("decoded_video")  # decoded VAE output
        save_video_comparison(frames_orig, None, frames_denoised, video_path)

        # Save action comparison
        action_path = os.path.join(args.output_dir, f"action_ep{ep_idx:06d}.png")
        save_action_comparison(
            sample.get("action"),  # Ground truth
            output.get("predicted_action"),  # Predicted
            action_path,
        )

        results.append({
            "episode": int(ep_idx),
            "video_plot": video_path,
            "action_plot": action_path,
        })

    # Save summary
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} samples saved to {args.output_dir}")


def load_episode_sample(data_root, ep_idx):
    """Load a single episode's data for evaluation."""
    import pandas as pd
    from decord import VideoReader

    # Read info.json for paths
    with open(os.path.join(data_root, "meta", "info.json")) as f:
        info = json.load(f)

    chunk = ep_idx // info["chunks_size"]
    parquet_path = os.path.join(data_root, f"data/chunk-{chunk:03d}", f"episode_{ep_idx:06d}.parquet")
    if not os.path.exists(parquet_path):
        return None

    df = pd.read_parquet(parquet_path)

    # Load video frames for each camera
    cameras = ['exterior_image_1_left', 'exterior_image_2_left', 'wrist_image_left']
    all_frames = []
    for cam in cameras:
        video_path = os.path.join(data_root, "videos", f"chunk-{chunk:03d}", cam, f"episode_{ep_idx:06d}.mp4")
        if not os.path.exists(video_path):
            continue
        vr = VideoReader(video_path)
        frames = vr.get_batch(range(len(vr))).asnumpy()  # (T, H, W, C)
        all_frames.append(frames)

    # Build sample dict similar to what the model expects
    sample = {
        "video_orig": np.stack(all_frames, axis=0) if all_frames else None,  # (V, T, H, W, C)
        "state": np.stack(df["observation.state"].values),
        "action": np.stack(df["action"].values),
        "annotation.language.language_instruction": str(df["annotation.language.language_instruction"].iloc[0]),
        "embodiment_id": 0,  # OXE_DROID
    }
    return sample


if __name__ == "__main__":
    main()
