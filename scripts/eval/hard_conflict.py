#!/usr/bin/env python3
"""
Hard Semantic Conflict Experiment

Tests whether the model actually uses first-frame visual content by creating
GENUINE conflicts between what the camera sees and what the state says.

Previous "wrong_ep" = swap with another DROID episode (same kitchen, similar robot).
This test: more extreme perturbations that should confuse the model IF it uses vision.

Conditions:
  Full:              correct first frame
  Natural image:     replace 1st frame with an ImageNet-style natural photo
  Rotated 1st frame: rotate 180 degrees (same content, wrong orientation)
  Right half zeroed: zero out the RIGHT half (where robot arm usually is)
  Left half zeroed:  zero out the LEFT half
  High contrast:     max-min normalize (destroys natural appearance)
  Wrong ep (control): swap with another DROID first frame
  AO:                skip video
"""
import os, sys, json, time, argparse
import numpy as np

os.environ["DREAMZERO_DEVICE"] = "npu"
sys.path.insert(0, "/workspace/dreamzero")
import torch, torch.distributed as dist
torch._dynamo.config.disable = True

_port = os.environ.get("CONFLICT_PORT", "29810")
for k, v in [("MASTER_ADDR", "localhost"), ("MASTER_PORT", _port),
             ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]:
    os.environ[k] = v
if not dist.is_initialized():
    dist.init_process_group(backend="hccl")
from omegaconf import OmegaConf
from hydra.utils import instantiate
from safetensors.torch import load_file
from groot.vla.common.utils.device import DEVICE

def _reset_model_state(ah):
    ah.current_start_frame = 0; ah.language = None; ah.clip_feas = None
    ah.ys = None; ah.kv_cache1 = None; ah.kv_cache_neg = None
    ah.crossattn_cache = None; ah.crossattn_cache_neg = None

def find_config(ckpt_dir):
    for d in [os.path.join(ckpt_dir, "experiment_cfg"),
              os.path.join(os.path.dirname(ckpt_dir), "experiment_cfg"),
              "/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v3/experiment_cfg"]:
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d):
                for f in files:
                    if f in ("conf.yaml","config.yaml"):
                        return os.path.join(root, f)
    raise FileNotFoundError

def perturb(video, mode, all_videos=None, idx=None):
    """Perturb first frame. video: [T,H,W,C] numpy/torch."""
    is_np = isinstance(video, np.ndarray)
    v = video.copy() if is_np else video.clone()
    shape = v[0].shape  # [H, W, C]

    if mode == "full": return v

    if mode == "wrong_ep":
        v[0] = all_videos[(idx+1)%len(all_videos)][0].copy() if is_np else all_videos[(idx+1)%len(all_videos)][0].clone()

    elif mode == "natural_img":
        # Create a checkerboard pattern with natural-like statistics
        # (We can't download real images, so use a structured pattern)
        h, w, c = shape
        x = np.linspace(0, 4*np.pi, w)
        y = np.linspace(0, 4*np.pi, h)
        xx, yy = np.meshgrid(x, y)
        img = 0.5 * (np.sin(xx) * np.cos(yy) + 1)  # smooth wave pattern
        img = np.stack([img]*c, axis=-1).astype(np.float32)
        v[0] = img if is_np else torch.from_numpy(img).to(v.device).to(v.dtype)

    elif mode == "rotate180":
        v[0] = np.rot90(v[0], 2) if is_np else torch.rot90(v[0], 2, dims=[0,1])

    elif mode == "right_half_zero":
        h, w, c = shape
        vc = v[0].copy() if is_np else v[0].clone()
        vc[:, w//2:, :] = 0  # zero right half
        v[0] = vc

    elif mode == "left_half_zero":
        h, w, c = shape
        vc = v[0].copy() if is_np else v[0].clone()
        vc[:, :w//2, :] = 0  # zero left half
        v[0] = vc

    elif mode == "high_contrast":
        vc = v[0].astype(np.float32) if is_np else v[0].float()
        vmin, vmax = vc.min(), vc.max()
        if vmax > vmin:
            vc = (vc - vmin) / (vmax - vmin)
        else:
            vc = vc * 0
        v[0] = vc.astype(v.dtype) if is_np else vc.to(v.dtype)

    return v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = "/tmp/conflict_results"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    cfg_file = find_config(args.checkpoint)
    cfg = OmegaConf.load(cfg_file)
    model = instantiate(cfg.model); model.eval(); model.to(DEVICE)
    ah = model.action_head; ah.post_initialize()
    adapter = os.path.join(args.checkpoint, "adapter_model.safetensors")
    sd = load_file(adapter)
    stripped = {k.replace(".base_layer.", ".") if ".base_layer." in k else k: v for k, v in sd.items()}
    model.load_state_dict(stripped, strict=False)
    print(f"Model loaded. LoRA keys: {sum(1 for k in stripped if 'lora_' in k)}")

    # Load samples
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
        ids, amask = tk([txt], return_mask=True)
        neg_ids, neg_mask = tk([""], return_mask=True)
        inp = {"images": raw, "state": s["state"], "action": s["action"],
               "text": ids, "text_attention_mask": amask,
               "text_negative": neg_ids, "text_attention_mask_negative": neg_mask,
               "embodiment_id": s["embodiment_id"],
               "has_real_action": s.get("has_real_action", True),
               "action_mask": s.get("action_mask", torch.ones(len(s["action"]),32))}
        all_inps.append(inp)
        all_gts.append(s["action"].float().numpy() if hasattr(s["action"],'numpy') else np.asarray(s["action"]))

    def prepare(template, video):
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

    conditions = [
        ("Full (correct)",      "full"),
        ("Wrong ep (control)",  "wrong_ep"),
        ("Natural pattern",     "natural_img"),
        ("Rotated 180°",        "rotate180"),
        ("Right half zeroed",   "right_half_zero"),
        ("Left half zeroed",    "left_half_zero"),
        ("High contrast",       "high_contrast"),
    ]
    results = {}

    for label, mode in conditions:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
        mses = []
        for i in range(len(all_inps)):
            _reset_model_state(ah)
            video = perturb(all_videos[i], mode, all_videos, i)
            inp = prepare(all_inps[i], video)
            with torch.no_grad():
                out = model.lazy_joint_video_action_causal(inp)
            pred = out["action_pred"][0].cpu().float().numpy()
            mse = float(np.mean((all_gts[i][:len(pred)] - pred)**2))
            mses.append(mse)
        results[label] = {"mse": mses, "mean": float(np.mean(mses))}
        if "Full (correct)" in results:
            diffs = [f"{m-f:+.1f}" for m,f in zip(mses, results["Full (correct)"]["mse"])]
            print(f"  mean={results[label]['mean']:.2f}  per-sample diff vs Full: {diffs}")
        else:
            print(f"  mean={results[label]['mean']:.2f}")

    # AO
    print(f"\n{'='*60}\n  AO\n{'='*60}")
    saved_d, saved_f = ah.config.decouple_inference_noise, ah.config.video_inference_final_noise
    ah.config.decouple_inference_noise = True; ah.config.video_inference_final_noise = 1.0
    ao_mses = []
    for i in range(len(all_inps)):
        _reset_model_state(ah)
        inp = prepare(all_inps[i], all_videos[i])
        with torch.no_grad():
            out = model.lazy_joint_video_action_causal(inp)
        pred = out["action_pred"][0].cpu().float().numpy()
        ao_mses.append(float(np.mean((all_gts[i][:len(pred)] - pred)**2)))
    ah.config.decouple_inference_noise = saved_d; ah.config.video_inference_final_noise = saved_f
    results["AO"] = {"mse": ao_mses, "mean": float(np.mean(ao_mses))}
    diffs = [f"{a-f:+.1f}" for a,f in zip(ao_mses, results["Full (correct)"]["mse"])]
    print(f"  mean={results['AO']['mean']:.2f}  per-sample diff vs Full: {diffs}")

    # Summary
    full = results["Full (correct)"]["mean"]
    ao = results["AO"]["mean"]
    gap = ao - full
    print(f"\n\n{'='*70}")
    print(f"  HARD CONFLICT RESULTS")
    print(f"  Full={full:.2f}  AO={ao:.2f}  gap={gap:.2f}")
    print(f"{'='*70}")
    print(f"  {'Condition':<25} {'MSE':>8} {'vs Full':>9} {'%gap':>8}")
    print(f"  {'-'*52}")
    for label in ["Full (correct)","Wrong ep (control)","Natural pattern",
                   "Rotated 180°","Right half zeroed","Left half zeroed",
                   "High contrast","AO"]:
        r = results.get(label, {})
        mse = r.get("mean")
        if mse is not None:
            vs = mse - full; pct = f"{vs/gap*100:.0f}%"
            print(f"  {label:<25} {mse:>8.2f} {vs:>+9.2f} {pct:>8}")

    out_file = os.path.join(output_dir, "conflict_results.json")
    with open(out_file, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "results": {
            k: {"mse": v["mse"], "mean": v["mean"]} for k,v in results.items()
        }}, f, indent=2)
    print(f"\nSaved: {out_file}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
