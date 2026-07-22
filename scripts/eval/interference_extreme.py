#!/usr/bin/env python3
"""
Video Denoising Interference Experiment — EXTREME Edition

Tests: what if video latents are completely replaced with random noise at each step?
This is MORE extreme than corrupting flow_pred — it corrupts the DiT INPUT, which
affects attention Q/K/V computation for video tokens.

Conditions:
  Full (baseline):    normal joint denoising
  AO:                 skip video denoising entirely
  Full+random_latents: each step, replace video latents with fresh N(0,1) before DiT
  Full+zero_latents:  each step, replace video latents with zeros before DiT
"""
import os, sys, json, time, argparse
import numpy as np

os.environ["DREAMZERO_DEVICE"] = "npu"
sys.path.insert(0, "/workspace/dreamzero")

import torch, torch.distributed as dist
torch._dynamo.config.disable = True

for k, v in [("MASTER_ADDR", "localhost"), ("MASTER_PORT", "29760"),
             ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]:
    os.environ.setdefault(k, v)
if not dist.is_initialized():
    dist.init_process_group(backend="hccl")

from omegaconf import OmegaConf
from hydra.utils import instantiate
from safetensors.torch import load_file
from groot.vla.common.utils.device import DEVICE
from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler

# Global mode for current experiment run
CURRENT_MODE = "normal"  # "normal" | "random_latents" | "zero_latents"
_ORIGINAL_SCHEDULER_STEP = FlowUniPCMultistepScheduler.step


def _corrupt_step(self, model_output, timestep, sample, step_index, return_dict=False):
    """Monkey-patched scheduler step: corrupt video latents after scheduler update."""
    result = _ORIGINAL_SCHEDULER_STEP(self, model_output, timestep, sample, step_index, return_dict)
    if CURRENT_MODE != "normal" and model_output.ndim == 5:
        # Only touch video scheduler (5D: [B,C,T,H,W]), NOT action scheduler (3D)
        if isinstance(result, tuple):
            if CURRENT_MODE == "random_latents":
                result = (torch.randn_like(result[0]),) + result[1:]
            elif CURRENT_MODE == "zero_latents":
                result = (torch.zeros_like(result[0]),) + result[1:]
        else:
            if CURRENT_MODE == "random_latents":
                result = torch.randn_like(result)
            elif CURRENT_MODE == "zero_latents":
                result = torch.zeros_like(result)
    return result


def install_patch():
    FlowUniPCMultistepScheduler.step = _corrupt_step


def uninstall_patch():
    FlowUniPCMultistepScheduler.step = _ORIGINAL_SCHEDULER_STEP


def _reset_model_state(ah):
    ah.current_start_frame = 0
    ah.language = None
    ah.clip_feas = None
    ah.ys = None
    ah.kv_cache1 = None
    ah.kv_cache_neg = None
    ah.crossattn_cache = None
    ah.crossattn_cache_neg = None


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
    global CURRENT_MODE

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="eval/interference_extreme")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    cfg_file = find_config(args.checkpoint)
    print(f"Config: {cfg_file}")
    cfg = OmegaConf.load(cfg_file)

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
    lora_n = sum(1 for k in stripped_sd if "lora_" in k)
    print(f"Loaded LoRA: {lora_n} lora keys")

    # --- Load samples ---
    print(f"Loading {args.num_samples} samples...")
    torch.manual_seed(args.seed)
    ds = instantiate(cfg.train_dataset)
    it = iter(ds)
    samples = [next(it) for _ in range(args.num_samples)]

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
        _skip = {"text", "text_attention_mask", "text_negative", "text_attention_mask_negative"}
        for k in list(inp.keys()):
            v = inp[k]
            if isinstance(v, torch.Tensor):
                inp[k] = v.to(DEVICE) if k in _skip else v.unsqueeze(0).to(DEVICE)
            elif isinstance(v, np.ndarray):
                inp[k] = torch.from_numpy(v).unsqueeze(0).to(DEVICE) if v.ndim > 0 else torch.tensor([v.item()], device=DEVICE)
            elif isinstance(v, (int, float, bool)):
                inp[k] = torch.tensor([int(v) if isinstance(v, bool) else v], device=DEVICE)
        if inp["state"].shape[1] > ah.model.num_state_per_block:
            inp["state"] = inp["state"][:, -ah.model.num_state_per_block:]
        gt = s["action"].float().numpy() if hasattr(s["action"], 'numpy') else np.asarray(s["action"])
        preprocessed.append((inp, gt))

    # --- Run conditions ---
    conditions = [
        ("Full (normal)", "normal"),
        ("Full+random_latents", "random_latents"),
        ("Full+zero_latents", "zero_latents"),
    ]

    all_results = {}

    for label, mode in conditions:
        CURRENT_MODE = mode
        install_patch()
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        mses, times = [], []
        for i, (inp, gt) in enumerate(preprocessed):
            _reset_model_state(ah)
            try:
                with torch.no_grad():
                    t0 = time.time()
                    out = model.lazy_joint_video_action_causal(inp)
                    dt = time.time() - t0
                pred = out["action_pred"][0].cpu().float().numpy()
                mse = float(np.mean((gt[:len(pred)] - pred) ** 2))
                mses.append(mse); times.append(dt)
                print(f"  [{i}]: MSE={mse:.4f}, t={dt:.1f}s")
            except Exception as e:
                print(f"  [{i}] ERROR: {e}")

        uninstall_patch()
        all_results[label] = {"mse_per_sample": mses, "mse_mean": float(np.mean(mses)) if mses else None,
                              "time_mean": float(np.mean(times)) if times else None}
        print(f"  Mean MSE: {all_results[label]['mse_mean']:.4f}")

    # --- AO baseline ---
    print(f"\n{'='*60}")
    print(f"  AO (skip video)")
    print(f"{'='*60}")
    saved_d, saved_f = ah.config.decouple_inference_noise, ah.config.video_inference_final_noise
    ah.config.decouple_inference_noise = True
    ah.config.video_inference_final_noise = 1.0
    ao_mses = []
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
            print(f"  AO[{i}]: MSE={mse:.4f}, t={dt:.1f}s")
        except Exception as e:
            print(f"  AO[{i}] ERROR: {e}")
    ah.config.decouple_inference_noise = saved_d
    ah.config.video_inference_final_noise = saved_f
    all_results["AO"] = {"mse_per_sample": ao_mses, "mse_mean": float(np.mean(ao_mses)) if ao_mses else None}

    # --- Summary ---
    baseline_full = all_results.get("Full (normal)", {}).get("mse_mean")
    baseline_ao = all_results.get("AO", {}).get("mse_mean")
    gap = baseline_ao - baseline_full if baseline_ao and baseline_full else 0

    print(f"\n{'='*70}")
    print(f"  EXTREME INTERFERENCE RESULTS")
    print(f"{'='*70}")
    print(f"{'Condition':<25} {'Action MSE':>12} {'vs Full':>10} {'% of gap':>12}")
    print("-" * 65)
    for label in ["Full (normal)", "Full+random_latents", "Full+zero_latents", "AO"]:
        r = all_results.get(label, {})
        mse = r.get("mse_mean")
        if mse is not None:
            vs = mse - baseline_full if baseline_full else 0
            pct = f"{vs / gap * 100:.0f}%" if gap else "--"
            print(f"{label:<25} {mse:>12.4f} {vs:>+10.4f} {pct:>12}")

    print(f"\n  AO-Full gap: {gap:.4f}")
    rl = all_results.get("Full+random_latents", {}).get("mse_mean")
    zl = all_results.get("Full+zero_latents", {}).get("mse_mean")
    if rl is not None and baseline_full is not None:
        print(f"  random_latents vs Full: {rl - baseline_full:+.4f}")
        if rl < baseline_ao:
            print(f"  → Random video latents STILL help action (better than AO)!")
        else:
            print(f"  → Random video latents DEGRADE action below AO!")

    summary = {"checkpoint": args.checkpoint, "num_samples": args.num_samples, "results": all_results}
    out_file = os.path.join(args.output_dir, "interference_extreme_results.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_file}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
