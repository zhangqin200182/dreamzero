#!/usr/bin/env python3
"""
Video Denoising Interference Experiment

核心问题: 视频去噪的质量对 Action 预测有因果贡献吗？
  - 在推理时往视频去噪过程中注入噪声扰动
  - 测量 Action MSE 如何随噪声强度变化
  - 剂量-响应曲线揭示视频去噪"过程"对 Action 的因果贡献

假说:
  H1 (过程健壮): 只要在去噪, 结果差也无所谓 → 噪声不显著影响 Action
  H2 (质量敏感): 去噪过程越干净, Action越好 → 噪声损害 Action
  H3 (开关效应): 注噪声 = 等价关掉视频 → 即使轻噪声也退化到 AO 水平

实验组:
  noise_level = 0.0 (baseline Full), 0.05, 0.1, 0.3, 0.5, 1.0, + AO baseline
"""

import os, sys, json, time, argparse
import numpy as np

os.environ["DREAMZERO_DEVICE"] = "npu"
sys.path.insert(0, "/workspace/dreamzero")

import torch
import torch.distributed as dist

torch._dynamo.config.disable = True

for k, v in [("MASTER_ADDR", "localhost"), ("MASTER_PORT", "29750"),
             ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]:
    os.environ.setdefault(k, v)

if not dist.is_initialized():
    dist.init_process_group(backend="hccl")

from omegaconf import OmegaConf
from hydra.utils import instantiate
from safetensors.torch import load_file
from groot.vla.common.utils.device import DEVICE
from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler


# Global noise level for the current experiment run
CURRENT_NOISE_LEVEL = 0.0

# Save original step method
_ORIGINAL_SCHEDULER_STEP = FlowUniPCMultistepScheduler.step


def _noisy_step(self, model_output, timestep, sample, step_index, return_dict=False):
    """Monkey-patched scheduler step: inject noise into video flow_pred ONLY."""
    if CURRENT_NOISE_LEVEL > 0 and model_output.ndim == 5:
        # Video flow_pred: [B, C, T, H, W] — 5D
        # Action flow_pred: [B, T, D] — 3D — never touched
        noise = torch.randn_like(model_output) * CURRENT_NOISE_LEVEL
        model_output = model_output + noise
    return _ORIGINAL_SCHEDULER_STEP(self, model_output, timestep, sample,
                                     step_index, return_dict)


def patch_scheduler():
    """Install the noisy scheduler step."""
    FlowUniPCMultistepScheduler.step = _noisy_step


def unpatch_scheduler():
    """Restore original scheduler step."""
    FlowUniPCMultistepScheduler.step = _ORIGINAL_SCHEDULER_STEP


def _reset_model_state(ah):
    """Reset inference state for a fresh sample."""
    ah.current_start_frame = 0
    ah.language = None
    ah.clip_feas = None
    ah.ys = None
    ah.kv_cache1 = None
    ah.kv_cache_neg = None
    ah.crossattn_cache = None
    ah.crossattn_cache_neg = None


def find_config(checkpoint_dir):
    """Find config file near checkpoint directory."""
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
    global CURRENT_NOISE_LEVEL

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="eval/interference")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Noise levels to test (including AO as special case)
    noise_levels = [0.0, 0.05, 0.1, 0.3, 0.5, 1.0]

    # --- Load config ---
    cfg_file = find_config(args.checkpoint)
    print(f"Config: {cfg_file}")
    cfg = OmegaConf.load(cfg_file)

    # --- Load model ---
    print("Loading model...")
    model = instantiate(cfg.model)
    model.eval()
    model.to(DEVICE)
    ah = model.action_head
    ah.post_initialize()
    print("post_initialize done")

    # --- Load LoRA ---
    adapter_file = os.path.join(args.checkpoint, "adapter_model.safetensors")
    sd = load_file(adapter_file)
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
    torch.manual_seed(args.seed)
    ds = instantiate(cfg.train_dataset)
    it = iter(ds)
    samples = [next(it) for _ in range(args.num_samples)]
    print(f"Loaded {len(samples)} samples")

    # --- Preprocess samples once ---
    from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer

    preprocessed = []
    for s in samples:
        tk = HuggingfaceTokenizer(name="/checkpoints/umt5-xxl", seq_len=512, clean='whitespace')
        txt = s.get("text", "") or ""
        ids, amask = tk([txt], return_mask=True)
        neg_ids, neg_mask = tk([""], return_mask=True)

        inp = {
            "images": s["images"], "state": s["state"], "action": s["action"],
            "text": ids, "text_attention_mask": amask,
            "text_negative": neg_ids, "text_attention_mask_negative": neg_mask,
            "embodiment_id": s["embodiment_id"],
            "has_real_action": s.get("has_real_action", True),
            "action_mask": s.get("action_mask", torch.ones(len(s["action"]), 32)),
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

        gt = s["action"].float().numpy() if hasattr(s["action"], 'numpy') else np.asarray(s["action"])
        preprocessed.append((inp, gt))

    # --- Run experiment for each noise level ---
    all_results = {}

    # AO baseline (special: no video denoising at all)
    print(f"\n{'='*60}")
    print(f"AO baseline (skip video denoising)")
    print(f"{'='*60}")
    ao_mses = []
    ao_times = []
    saved_decouple = ah.config.decouple_inference_noise
    saved_final_noise = ah.config.video_inference_final_noise
    ah.config.decouple_inference_noise = True
    ah.config.video_inference_final_noise = 1.0

    for i, (inp, gt) in enumerate(preprocessed):
        _reset_model_state(ah)
        try:
            with torch.no_grad():
                t0 = time.time()
                out = model.lazy_joint_video_action_causal(inp)
                dt = time.time() - t0
            pred = out["action_pred"][0].cpu().float().numpy()
            mse = float(np.mean((gt[:len(pred)] - pred) ** 2))
            ao_mses.append(mse)
            ao_times.append(dt)
            print(f"  AO[{i}]: MSE={mse:.4f}, t={dt:.1f}s")
        except Exception as e:
            print(f"  AO[{i}] ERROR: {e}")

    ah.config.decouple_inference_noise = saved_decouple
    ah.config.video_inference_final_noise = saved_final_noise

    ao_mean = float(np.mean(ao_mses)) if ao_mses else None
    all_results["AO"] = {"mse_per_sample": ao_mses, "mse_mean": ao_mean, "time_mean": float(np.mean(ao_times)) if ao_times else None}
    print(f"  AO mean MSE: {ao_mean:.4f}\n")

    # Full mode with noise injection
    for nl in noise_levels:
        label = f"Full+noise={nl}" if nl > 0 else "Full (baseline)"
        print(f"{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")

        CURRENT_NOISE_LEVEL = nl
        patch_scheduler()

        mses = []
        times = []
        for i, (inp, gt) in enumerate(preprocessed):
            _reset_model_state(ah)
            try:
                with torch.no_grad():
                    t0 = time.time()
                    out = model.lazy_joint_video_action_causal(inp)
                    dt = time.time() - t0
                pred = out["action_pred"][0].cpu().float().numpy()
                mse = float(np.mean((gt[:len(pred)] - pred) ** 2))
                mses.append(mse)
                times.append(dt)
                print(f"  [{i}]: MSE={mse:.4f}, t={dt:.1f}s")
            except Exception as e:
                print(f"  [{i}] ERROR: {e}")
                import traceback
                traceback.print_exc()

        unpatch_scheduler()
        mean_mse = float(np.mean(mses)) if mses else None
        all_results[f"noise={nl}"] = {"mse_per_sample": mses, "mse_mean": mean_mse, "time_mean": float(np.mean(times)) if times else None}
        print(f"  Mean MSE: {mean_mse:.4f}\n")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("  DOSE-RESPONSE: Action MSE vs Video Noise Injection Level")
    print(f"{'='*70}")
    print(f"{'Condition':<25} {'Action MSE':>12} {'Delta vs Full':>14} {'Delta vs AO':>14}")
    print("-" * 70)

    baseline_full = all_results.get("noise=0.0", {}).get("mse_mean", None)
    baseline_ao = all_results.get("AO", {}).get("mse_mean", None)

    for key in ["noise=0.0", "noise=0.05", "noise=0.1", "noise=0.3", "noise=0.5", "noise=1.0"]:
        r = all_results.get(key, {})
        mse = r.get("mse_mean")
        if mse is not None:
            delta_vs_full = mse - baseline_full if baseline_full is not None else float('nan')
            delta_vs_ao = mse - baseline_ao if baseline_ao is not None else float('nan')
            print(f"{key:<25} {mse:>12.4f} {delta_vs_full:>+14.4f} {delta_vs_ao:>+14.4f}")

    r = all_results.get("AO", {})
    if r.get("mse_mean") is not None:
        delta_full = r["mse_mean"] - baseline_full if baseline_full is not None else float('nan')
        print(f"{'AO (skip video)':<25} {r['mse_mean']:>12.4f} {delta_full:>+14.4f} {'--':>14}")

    # Interpretation
    print(f"\n{'='*70}")
    print("  INTERPRETATION")
    print(f"{'='*70}")
    print(f"  Baseline Full:  {baseline_full:.4f}")
    print(f"  Baseline AO:    {baseline_ao:.4f}")
    print(f"  AO-Full gap:    {baseline_ao - baseline_full:.4f}")

    # Check which noise levels exceed AO
    for key in ["noise=0.05", "noise=0.1", "noise=0.3", "noise=0.5", "noise=1.0"]:
        r = all_results.get(key, {})
        mse = r.get("mse_mean")
        if mse is not None and baseline_ao is not None:
            if mse > baseline_ao:
                print(f"  {key}: MSE={mse:.4f} EXCEEDS AO ({baseline_ao:.4f}) → noise IS worse than skipping video")
            elif mse > baseline_full * 1.1:
                print(f"  {key}: MSE={mse:.4f} > 1.1× baseline → noise modestly degrades action")
            else:
                print(f"  {key}: MSE={mse:.4f} ≈ baseline → noise has NO effect on action")

    # Save
    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "results": all_results,
    }
    out_file = os.path.join(args.output_dir, "interference_results.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_file}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
