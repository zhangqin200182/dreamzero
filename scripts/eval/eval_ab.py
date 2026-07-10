#!/usr/bin/env python3
"""A/B Eval: Full vs Action-Only inference for DreamZero LoRA checkpoints.

Full mode:     joint video+action denoising (normal)
Action-Only:   skip video denoising, predict action only (faster)
"""
import os, sys, json, time, argparse
import numpy as np
os.environ["DREAMZERO_DEVICE"] = "npu"
sys.path.insert(0, "/workspace/dreamzero")
import torch, torch.distributed as dist
# torch.compile requires triton drivers only available under torchrun multi-process.
# In single-process eval, disable it globally so decorator-based @torch.compile
# on scheduler/encoder forwards become no-ops.
torch._dynamo.config.disable = True
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="eval/step_200")
    parser.add_argument("--num_samples", type=int, default=3)
    args = parser.parse_args()

    # --- Load config ---
    exp_dir = os.path.dirname(args.checkpoint)
    cfg_dir = os.path.join(exp_dir, "experiment_cfg")
    cfg_file = None
    for root, dirs, files in os.walk(cfg_dir):
        for f in files:
            if f in ("conf.yaml", "config.yaml"):
                cfg_file = os.path.join(root, f)
                break
    if cfg_file is None:
        raise FileNotFoundError(f"No config found in {cfg_dir}")

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(cfg_file)
    from hydra.utils import instantiate
    from groot.vla.common.utils.device import DEVICE

    # --- Instantiate model ---
    model = instantiate(cfg.model)
    model.eval()
    model.to(DEVICE)

    ah = model.action_head
    ah.post_initialize()
    print("post_initialize done")

    # --- Load LoRA weights ---
    from safetensors.torch import load_file
    adapter_file = os.path.join(args.checkpoint, "adapter_model.safetensors")
    sd = load_file(adapter_file)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded LoRA: {len(sd)} keys, {len(missing)} missing, {len(unexpected)} unexpected")

    # --- Load samples ---
    ds = instantiate(cfg.train_dataset)
    it = iter(ds)
    samples = [next(it) for _ in range(args.num_samples)]
    print(f"Loaded {len(samples)} samples")

    os.makedirs(f"{args.output_dir}/full", exist_ok=True)
    os.makedirs(f"{args.output_dir}/action_only", exist_ok=True)
    results = []

    for i, s in enumerate(samples):
        gt = s["action"].float().numpy() if hasattr(s["action"], 'numpy') else np.asarray(s["action"])

        # Build input dict from transformed sample
        from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer
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

        # The inference path concatenates state tokens with action tokens into a
        # register and asserts register length == num_action_per_block + num_state_per_block.
        # The dataset may provide more state tokens than needed; keep only the most recent.
        if inp["state"].shape[1] > ah.model.num_state_per_block:
            inp["state"] = inp["state"][:, -ah.model.num_state_per_block:]

        # --- Full mode ---
        fm, ft_val = None, None
        try:
            _reset_model_state(ah)
            with torch.no_grad():
                t0 = time.time()
                out = model.lazy_joint_video_action_causal(inp)
                ft_val = time.time() - t0
            pred = out["action_pred"][0].cpu().float().numpy()
            fm = float(np.mean((gt[:len(pred)] - pred) ** 2))
            save_plot(gt, pred, f"{args.output_dir}/full/ep_{i:02d}.png", f"Full {i}")
            print(f"Full[{i}]: MSE={fm:.6f} t={ft_val:.1f}s")
        except Exception:
            import traceback
            traceback.print_exc()

        # --- Action-Only mode ---
        am, at_val = None, None
        try:
            _reset_model_state(ah)
            saved_decouple = ah.config.decouple_inference_noise
            saved_final_noise = ah.config.video_inference_final_noise
            ah.config.decouple_inference_noise = True
            ah.config.video_inference_final_noise = 1.0
            with torch.no_grad():
                t0 = time.time()
                out = model.lazy_joint_video_action_causal(inp)
                at_val = time.time() - t0
            ah.config.decouple_inference_noise = saved_decouple
            ah.config.video_inference_final_noise = saved_final_noise
            pred = out["action_pred"][0].cpu().float().numpy()
            am = float(np.mean((gt[:len(pred)] - pred) ** 2))
            save_plot(gt, pred, f"{args.output_dir}/action_only/ep_{i:02d}.png", f"AO {i}")
            print(f"AO[{i}]: MSE={am:.6f} t={at_val:.1f}s")
        except Exception:
            import traceback
            traceback.print_exc()
            if 'saved_decouple' in dir():
                ah.config.decouple_inference_noise = saved_decouple
                ah.config.video_inference_final_noise = saved_final_noise

        results.append({"i": i, "full_mse": fm, "ao_mse": am})

    # Summary
    fm_all = [r["full_mse"] for r in results if r["full_mse"] is not None]
    am_all = [r["ao_mse"] for r in results if r["ao_mse"] is not None]
    summary = {
        "checkpoint": args.checkpoint,
        "full_mse": float(np.mean(fm_all)) if fm_all else None,
        "ao_mse": float(np.mean(am_all)) if am_all else None,
    }
    with open(f"{args.output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSUMMARY: Full MSE={summary['full_mse']}, AO MSE={summary['ao_mse']}")


def save_plot(gt, pred, path, title):
    T = min(len(gt), len(pred))
    D = min(gt.shape[1], pred.shape[1], 7)
    fig, axes = plt.subplots(D, 1, figsize=(10, 2 * D), sharex=True)
    if D == 1:
        axes = [axes]
    for d in range(D):
        axes[d].plot(range(T), gt[:T, d], 'b-', label='GT', lw=2, alpha=.7)
        axes[d].plot(range(T), pred[:T, d], 'r--', label='Pred', lw=2, alpha=.7)
        axes[d].set_ylabel(f"J{d}")
        axes[d].legend(fontsize=7)
        axes[d].grid(alpha=.3)
    axes[-1].set_xlabel("Step")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
