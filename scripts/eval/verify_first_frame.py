#!/usr/bin/env python3
"""
Rigorous verification of the first-frame semantic independence hypothesis.

Tests across 2 checkpoints × 20 samples with more perturbation types:
  Full, black 1st frame, white 1st frame, random 1st frame,
  wrong-ep 1st frame, half-gray 1st frame, AO

If the hypothesis is correct:
  - wrong-ep ≈ Full (same structural scene, different semantics)
  - black/white/random cause similar modest degradation
  - all perturbed conditions >> AO (process benefit dominates)
"""
import os, sys, json, time, argparse
import numpy as np

os.environ["DREAMZERO_DEVICE"] = "npu"
sys.path.insert(0, "/workspace/dreamzero")
import torch, torch.distributed as dist
torch._dynamo.config.disable = True
# Port will be set per-checkpoint to avoid TIME_WAIT conflicts
if not dist.is_initialized():
    _port = os.environ.get("VERIFY_PORT", "29780")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = _port
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    dist.init_process_group(backend="hccl")
from omegaconf import OmegaConf
from hydra.utils import instantiate
from safetensors.torch import load_file
from groot.vla.common.utils.device import DEVICE

def _reset_model_state(ah):
    ah.current_start_frame = 0; ah.language = None; ah.clip_feas = None
    ah.ys = None; ah.kv_cache1 = None; ah.kv_cache_neg = None
    ah.crossattn_cache = None; ah.crossattn_cache_neg = None

def find_config(checkpoint_dir):
    for cfg_dir in [
        os.path.join(checkpoint_dir, "experiment_cfg"),
        os.path.join(os.path.dirname(checkpoint_dir), "experiment_cfg"),
        "/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v3/experiment_cfg",
        "/workspace/dreamzero/groot/vla/configs",
    ]:
        if os.path.isdir(cfg_dir):
            for root, dirs, files in os.walk(cfg_dir):
                for f in files:
                    if f in ("conf.yaml", "config.yaml"):
                        return os.path.join(root, f)
    raise FileNotFoundError(f"No config near {checkpoint_dir}")

def perturb_1st_frame(video, mode, other_video=None):
    """Perturb the first frame of a video. video shape: [T, H, W, C] (numpy or torch)."""
    is_np = isinstance(video, np.ndarray)
    v = video.copy() if is_np else video.clone()
    shape = v[0].shape
    if mode == "black":
        v[0] = np.zeros(shape, dtype=v.dtype) if is_np else torch.zeros(shape, dtype=v.dtype, device=v.device)
    elif mode == "white":
        v[0] = np.ones(shape, dtype=v.dtype) if is_np else torch.ones(shape, dtype=v.dtype, device=v.device)
    elif mode == "random":
        v[0] = np.random.randn(*shape).astype(v.dtype) if is_np else torch.randn(shape, dtype=v.dtype, device=v.device)
    elif mode == "gray":
        v[0] = (np.ones(shape, dtype=v.dtype) * 0.5) if is_np else torch.ones(shape, dtype=v.dtype, device=v.device) * 0.5
    elif mode == "wrong_ep" and other_video is not None:
        v[0] = other_video[0].copy() if is_np else other_video[0].clone()
    return v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output_dir", default="/tmp/verify_results")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_checkpoint_results = {}

    for ckpt_path in args.checkpoints:
        ckpt_name = os.path.basename(ckpt_path.rstrip("/"))
        print(f"\n{'#'*70}\n#  CHECKPOINT: {ckpt_name}\n{'#'*70}")

        # --- Load model ---
        cfg_file = find_config(ckpt_path)
        cfg = OmegaConf.load(cfg_file)
        model = instantiate(cfg.model); model.eval(); model.to(DEVICE)
        ah = model.action_head; ah.post_initialize()
        adapter_file = os.path.join(ckpt_path, "adapter_model.safetensors")
        sd = load_file(adapter_file)
        stripped = {k.replace(".base_layer.", ".") if ".base_layer." in k else k: v for k, v in sd.items()}
        model.load_state_dict(stripped, strict=False)
        print(f"Loaded LoRA: {sum(1 for k in stripped if 'lora_' in k)} keys")

        # --- Load samples ---
        torch.manual_seed(args.seed)
        ds = instantiate(cfg.train_dataset)
        it = iter(ds)
        samples = [next(it) for _ in range(args.num_samples)]

        from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer
        all_videos, all_inps, all_gts = [], [], []
        for s in samples:
            raw = s["images"]; all_videos.append(raw)
            tk = HuggingfaceTokenizer(name="/checkpoints/umt5-xxl", seq_len=512, clean='whitespace')
            txt = s.get("text","") or ""
            ids, amask = tk([txt], return_mask=True); neg_ids, neg_mask = tk([""], return_mask=True)
            inp = {"images": raw, "state": s["state"], "action": s["action"],
                   "text": ids, "text_attention_mask": amask,
                   "text_negative": neg_ids, "text_attention_mask_negative": neg_mask,
                   "embodiment_id": s["embodiment_id"],
                   "has_real_action": s.get("has_real_action", True),
                   "action_mask": s.get("action_mask", torch.ones(len(s["action"]), 32))}
            all_inps.append(inp)
            all_gts.append(s["action"].float().numpy() if hasattr(s["action"],'numpy') else np.asarray(s["action"]))

        def prepare_input(template, video):
            inp = dict(template); inp["images"] = video
            _skip = {"text","text_attention_mask","text_negative","text_attention_mask_negative"}
            for k in list(inp.keys()):
                v = inp[k]
                if isinstance(v, torch.Tensor): inp[k] = v.to(DEVICE) if k in _skip else v.unsqueeze(0).to(DEVICE)
                elif isinstance(v, np.ndarray):
                    inp[k] = torch.from_numpy(v).unsqueeze(0).to(DEVICE) if v.ndim>0 else torch.tensor([v.item()],device=DEVICE)
                elif isinstance(v, (int,float,bool)): inp[k] = torch.tensor([int(v)if isinstance(v,bool)else v],device=DEVICE)
            if inp["state"].shape[1] > ah.model.num_state_per_block: inp["state"] = inp["state"][:,-ah.model.num_state_per_block:]
            return inp

        # --- Run conditions ---
        conditions = [
            ("Full (correct)",    lambda i,v: v),
            ("1st=black",         lambda i,v: perturb_1st_frame(v, "black")),
            ("1st=white",         lambda i,v: perturb_1st_frame(v, "white")),
            ("1st=gray(0.5)",     lambda i,v: perturb_1st_frame(v, "gray")),
            ("1st=random",        lambda i,v: perturb_1st_frame(v, "random")),
            ("1st=wrong_ep",      lambda i,v: perturb_1st_frame(v, "wrong_ep", all_videos[(i+1)%len(all_videos)])),
        ]
        results = {}

        for label, perturb_fn in conditions:
            print(f"\n  --- {label} ---")
            mses = []
            for i in range(len(all_inps)):
                _reset_model_state(ah)
                video = perturb_fn(i, all_videos[i])
                inp = prepare_input(all_inps[i], video)
                with torch.no_grad():
                    out = model.lazy_joint_video_action_causal(inp)
                pred = out["action_pred"][0].cpu().float().numpy()
                mse = float(np.mean((all_gts[i][:len(pred)] - pred)**2))
                mses.append(mse)
            mean_mse = float(np.mean(mses))
            results[label] = {"mse_per_sample": mses, "mse_mean": mean_mse}
            # Print inline: Full vs this condition per sample
            if "Full" in results:
                diffs = [f"{m - f:+.1f}" for m, f in zip(mses, results["Full (correct)"]["mse_per_sample"])]
                print(f"  MSE={mean_mse:.2f}  diffs vs Full: {diffs}")
            else:
                print(f"  MSE={mean_mse:.2f}  samples: {[f'{x:.1f}' for x in mses]}")

        # --- AO baseline ---
        print(f"\n  --- AO ---")
        saved_d, saved_f = ah.config.decouple_inference_noise, ah.config.video_inference_final_noise
        ah.config.decouple_inference_noise = True; ah.config.video_inference_final_noise = 1.0
        ao_mses = []
        for i in range(len(all_inps)):
            _reset_model_state(ah)
            inp = prepare_input(all_inps[i], all_videos[i])
            with torch.no_grad():
                out = model.lazy_joint_video_action_causal(inp)
            pred = out["action_pred"][0].cpu().float().numpy()
            ao_mses.append(float(np.mean((all_gts[i][:len(pred)] - pred)**2)))
        ah.config.decouple_inference_noise = saved_d; ah.config.video_inference_final_noise = saved_f
        results["AO"] = {"mse_per_sample": ao_mses, "mse_mean": float(np.mean(ao_mses))}
        diff_ao = [f"{a - f:+.1f}" for a, f in zip(ao_mses, results["Full (correct)"]["mse_per_sample"])]
        print(f"  MSE={results['AO']['mse_mean']:.2f}  diffs vs Full: {diff_ao}")

        all_checkpoint_results[ckpt_name] = results
        # Clean up NPU memory before next checkpoint
        del model
        import gc; gc.collect()
        torch.npu.empty_cache()
        dist.destroy_process_group()

    # --- Cross-checkpoint summary ---
    print(f"\n\n{'='*85}")
    print(f"  CROSS-CHECKPOINT SUMMARY")
    print(f"{'='*85}")
    for ckpt_name, results in all_checkpoint_results.items():
        full = results["Full (correct)"]["mse_mean"]
        ao = results["AO"]["mse_mean"]
        gap = ao - full
        print(f"\n  {ckpt_name}:  Full={full:.2f}  AO={ao:.2f}  gap={gap:.2f}")
        print(f"  {'Condition':<22} {'MSE':>8} {'vs Full':>9} {'%gap':>8}")
        print(f"  {'-'*47}")
        for label in ["Full (correct)","1st=black","1st=white","1st=gray(0.5)","1st=random","1st=wrong_ep","AO"]:
            r = results.get(label, {})
            mse = r.get("mse_mean")
            if mse is not None:
                vs = mse - full
                pct = f"{vs/gap*100:.0f}%" if gap else "--"
                print(f"  {label:<22} {mse:>8.2f} {vs:>+9.2f} {pct:>8}")

    # Save
    out_file = os.path.join(args.output_dir, "verify_results.json")
    with open(out_file, "w") as f:
        json.dump(all_checkpoint_results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
