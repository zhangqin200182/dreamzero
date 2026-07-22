#!/usr/bin/env python3
"""
Semantic Video Perturbation Experiment

Tests whether video SEMANTIC CONTENT matters for action prediction.
Unlike adding Gaussian noise (which DiT handles trivially), these perturbations
replace the video with semantically wrong but visually meaningful content.

Conditions:
  Full (correct):    original video → action baseline
  Wrong video:       video from another episode → mismatch language vs visuals
  Shuffled frames:   randomize frame order within same video → break temporal causality
  Reversed:          reverse frame order → invert temporal flow
  AO:                skip video denoising → lower bound
"""
import os, sys, json, time, argparse
import numpy as np

os.environ["DREAMZERO_DEVICE"] = "npu"
sys.path.insert(0, "/workspace/dreamzero")

import torch, torch.distributed as dist
torch._dynamo.config.disable = True

for k, v in [("MASTER_ADDR", "localhost"), ("MASTER_PORT", "29770"),
             ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]:
    os.environ.setdefault(k, v)
if not dist.is_initialized():
    dist.init_process_group(backend="hccl")

from omegaconf import OmegaConf
from hydra.utils import instantiate
from safetensors.torch import load_file
from groot.vla.common.utils.device import DEVICE


def _reset_model_state(ah):
    ah.current_start_frame = 0
    ah.language = None
    ah.clip_feas = None
    ah.ys = None
    ah.kv_cache1 = None
    ah.kv_cache_neg = None
    ah.crossattn_cache = None
    ah.crossattn_cache_neg = None


def shuffle_frames(video):
    """Randomly permute frame order: [T, H, W, C] -> [T', H, W, C]"""
    if isinstance(video, torch.Tensor):
        idx = torch.randperm(video.shape[0])
        return video[idx]
    else:  # numpy
        idx = np.random.permutation(video.shape[0])
        return video[idx]


def reverse_frames(video):
    """Reverse frame order."""
    if isinstance(video, torch.Tensor):
        return video.flip(0)
    else:
        return video[::-1].copy()


def find_config(checkpoint_dir):
    search_paths = [
        os.path.join(checkpoint_dir, "experiment_cfg"),
        os.path.join(os.path.dirname(checkpoint_dir), "experiment_cfg"),
        "/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v3/experiment_cfg",
        "/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v2/experiment_cfg",
        "/workspace/dreamzero/groot/vla/configs",
    ]
    for cfg_dir in search_paths:
        if os.path.isdir(cfg_dir):
            for root, dirs, files in os.walk(cfg_dir):
                for f in files:
                    if f in ("conf.yaml", "config.yaml"):
                        return os.path.join(root, f)
    raise FileNotFoundError(f"No config found near {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="eval/semantic_perturbation")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    cfg_file = find_config(args.checkpoint)
    cfg = OmegaConf.load(cfg_file)
    print(f"Config: {cfg_file}")

    print("Loading model...")
    model = instantiate(cfg.model)
    model.eval()
    model.to(DEVICE)
    ah = model.action_head
    ah.post_initialize()

    adapter_file = os.path.join(args.checkpoint, "adapter_model.safetensors")
    sd = load_file(adapter_file)
    stripped_sd = {k.replace(".base_layer.", ".") if ".base_layer." in k else k: v for k, v in sd.items()}
    model.load_state_dict(stripped_sd, strict=False)
    print(f"Loaded LoRA: {sum(1 for k in stripped_sd if 'lora_' in k)} lora keys")

    # --- Load samples ---
    print(f"Loading {args.num_samples} samples...")
    torch.manual_seed(args.seed)
    ds = instantiate(cfg.train_dataset)
    it = iter(ds)
    samples = [next(it) for _ in range(args.num_samples)]

    from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer

    # Preprocess all samples
    all_inps = []
    all_gts = []
    all_videos = []  # keep raw videos for perturbation
    for s in samples:
        raw_video = s["images"]  # keep reference before device move
        all_videos.append(raw_video)

        tk = HuggingfaceTokenizer(name="/checkpoints/umt5-xxl", seq_len=512, clean='whitespace')
        txt = s.get("text", "") or ""
        ids, amask = tk([txt], return_mask=True)
        neg_ids, neg_mask = tk([""], return_mask=True)

        inp = {
            "images": raw_video,  # will be perturbed per-condition
            "state": s["state"], "action": s["action"],
            "text": ids, "text_attention_mask": amask,
            "text_negative": neg_ids, "text_attention_mask_negative": neg_mask,
            "embodiment_id": s["embodiment_id"],
            "has_real_action": s.get("has_real_action", True),
            "action_mask": s.get("action_mask", torch.ones(len(s["action"]), 32)),
        }
        all_inps.append(inp)
        gt = s["action"].float().numpy() if hasattr(s["action"], 'numpy') else np.asarray(s["action"])
        all_gts.append(gt)

    # --- Helper: prepare input dict for model (move to device, add batch dim) ---
    def prepare_input(inp_template, video):
        """Create a fresh input dict with the given video, moved to device."""
        inp = dict(inp_template)  # shallow copy
        inp["images"] = video  # override video
        _skip = {"text", "text_attention_mask", "text_negative", "text_attention_mask_negative"}
        for k in list(inp.keys()):
            v = inp[k]
            if isinstance(v, torch.Tensor):
                inp[k] = v.to(DEVICE) if k in _skip else v.unsqueeze(0).to(DEVICE)
            elif isinstance(v, np.ndarray):
                if v.ndim == 0:
                    inp[k] = torch.tensor([v.item()], device=DEVICE)
                else:
                    inp[k] = torch.from_numpy(v).unsqueeze(0).to(DEVICE)
            elif isinstance(v, (int, float, bool)):
                inp[k] = torch.tensor([int(v) if isinstance(v, bool) else v], device=DEVICE)
        if inp["state"].shape[1] > ah.model.num_state_per_block:
            inp["state"] = inp["state"][:, -ah.model.num_state_per_block:]
        return inp

    # --- Run conditions ---
    # The model only uses the FIRST FRAME for CLIP/VAE conditioning during inference.
    # Perturb only the first frame to test whether visual CONTENT matters.
    def black_first_frame(i, video):
        v = video.copy() if isinstance(video, np.ndarray) else video.clone()
        v[0] = 0.0  # black first frame
        return v

    def random_first_frame(i, video):
        v = video.copy() if isinstance(video, np.ndarray) else video.clone()
        v[0] = np.random.randn(*v[0].shape).astype(v.dtype) if isinstance(v, np.ndarray) else torch.randn_like(v[0])
        return v

    def wrong_first_frame(i, video):
        """Replace first frame with first frame from another sample."""
        other_video = all_videos[(i + 1) % len(all_videos)]
        v = video.copy() if isinstance(video, np.ndarray) else video.clone()
        v[0] = other_video[0]  # only first frame swapped
        return v

    conditions = [
        ("Full (correct)",           lambda i, v: v),                            # identity
        ("1st frame = black",        black_first_frame),                         # no visual info
        ("1st frame = random",       random_first_frame),                        # garbage visual info
        ("1st frame = wrong ep",     wrong_first_frame),                         # wrong semantic info
        ("All frames = wrong ep",    lambda i, v: all_videos[(i+1)%len(all_videos)]),  # sanity check
    ]

    all_results = {}

    for label, perturb_fn in conditions:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        mses = []
        for i in range(len(all_inps)):
            _reset_model_state(ah)
            try:
                video = perturb_fn(i, all_videos[i])
                inp = prepare_input(all_inps[i], video)
                with torch.no_grad():
                    t0 = time.time()
                    out = model.lazy_joint_video_action_causal(inp)
                    dt = time.time() - t0
                pred = out["action_pred"][0].cpu().float().numpy()
                gt = all_gts[i]
                mse = float(np.mean((gt[:len(pred)] - pred) ** 2))
                mses.append(mse)
                print(f"  [{i}]: MSE={mse:.4f}, t={dt:.1f}s")
            except Exception as e:
                print(f"  [{i}] ERROR: {e}")
                import traceback
                traceback.print_exc()

        all_results[label] = {"mse_per_sample": mses, "mse_mean": float(np.mean(mses)) if mses else None}
        print(f"  Mean MSE: {all_results[label]['mse_mean']:.4f}")

    # --- AO baseline ---
    print(f"\n{'='*60}")
    print(f"  AO (skip video)")
    print(f"{'='*60}")
    saved_d, saved_f = ah.config.decouple_inference_noise, ah.config.video_inference_final_noise
    ah.config.decouple_inference_noise = True
    ah.config.video_inference_final_noise = 1.0
    ao_mses = []
    for i in range(len(all_inps)):
        _reset_model_state(ah)
        try:
            inp = prepare_input(all_inps[i], all_videos[i])
            with torch.no_grad():
                out = model.lazy_joint_video_action_causal(inp)
            pred = out["action_pred"][0].cpu().float().numpy()
            mse = float(np.mean((all_gts[i][:len(pred)] - pred) ** 2))
            ao_mses.append(mse)
            print(f"  AO[{i}]: MSE={mse:.4f}")
        except Exception as e:
            print(f"  AO[{i}] ERROR: {e}")
    ah.config.decouple_inference_noise = saved_d
    ah.config.video_inference_final_noise = saved_f
    all_results["AO"] = {"mse_per_sample": ao_mses, "mse_mean": float(np.mean(ao_mses)) if ao_mses else None}

    # --- Summary ---
    baseline_full = all_results.get("Full (correct)", {}).get("mse_mean")
    baseline_ao = all_results.get("AO", {}).get("mse_mean")
    gap = baseline_ao - baseline_full if baseline_ao and baseline_full else 0

    print(f"\n{'='*75}")
    print(f"  SEMANTIC PERTURBATION RESULTS")
    print(f"{'='*75}")
    print(f"{'Condition':<25} {'Action MSE':>12} {'vs Full':>10} {'% of gap':>12}")
    print("-" * 65)
    for label in ["Full (correct)", "1st frame = black", "1st frame = random", "1st frame = wrong ep", "All frames = wrong ep"]:
        r = all_results.get(label, {})
        mse = r.get("mse_mean")
        if mse is not None:
            vs = mse - baseline_full
            pct = f"{vs/gap*100:.0f}%" if gap else "--"
            print(f"{label:<25} {mse:>12.4f} {vs:>+10.4f} {pct:>12}")
    r = all_results.get("AO", {})
    if r.get("mse_mean") is not None:
        print(f"{'AO (skip video)':<25} {r['mse_mean']:>12.4f} {r['mse_mean'] - baseline_full:>+10.4f} {'100%':>12}")

    # Per-sample detail
    print(f"\n  Per-sample comparison:")
    for i in range(len(all_gts)):
        vals = []
        for label in ["Full (correct)", "1st frame = black", "1st frame = random", "1st frame = wrong ep", "All frames = wrong ep", "AO"]:
            r = all_results.get(label, {})
            samples = r.get("mse_per_sample", [])
            if i < len(samples):
                vals.append(f"{samples[i]:.1f}")
        if vals:
            print(f"  Sample {i}: " + " | ".join(vals))

    print(f"\n{'='*75}")
    print(f"  INTERPRETATION")
    print(f"{'='*75}")
    print(f"  AO-Full gap: {gap:.2f}")
    for label in ["1st frame = black", "1st frame = random", "1st frame = wrong ep", "All frames = wrong ep"]:
        r = all_results.get(label, {})
        mse = r.get("mse_mean")
        if mse is not None and baseline_full is not None and baseline_ao is not None:
            degradation = mse - baseline_full
            if degradation < gap * 0.1:
                print(f"  {label}: {degradation:+.2f} vs Full — NEGLIGIBLE (<10% of gap)")
                print(f"    → Video semantic content barely matters for action")
            elif degradation < gap * 0.5:
                print(f"  {label}: {degradation:+.2f} vs Full — PARTIAL ({degradation/gap*100:.0f}% of gap)")
                print(f"    → Video content has MODERATE effect on action")
            else:
                print(f"  {label}: {degradation:+.2f} vs Full — SIGNIFICANT ({degradation/gap*100:.0f}% of gap)")
                print(f"    → Video content is IMPORTANT for action")

    summary = {"checkpoint": args.checkpoint, "num_samples": args.num_samples, "results": all_results}
    out_file = os.path.join(args.output_dir, "semantic_perturbation_results.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_file}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
