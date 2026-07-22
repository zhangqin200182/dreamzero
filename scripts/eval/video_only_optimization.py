#!/usr/bin/env python3
"""
Video-Only Optimization → Does Action Improve?

Train LoRA weights using ONLY video loss (dynamics_loss) for N steps.
Measure action_loss on a fixed validation set every K steps.

If action improves → video optimization transfers to action via shared DiT.
If action degrades → need mixing with action loss (catastrophic forgetting).
If action unchanged → shared representations are decoupled.

This is a multi-step version of the gradient propagation test.
"""
import os, sys, json, time, argparse
import numpy as np

os.environ["DREAMZERO_DEVICE"] = "npu"
sys.path.insert(0, "/workspace/dreamzero")
import torch, torch.distributed as dist
torch._dynamo.config.disable = True
_port = os.environ.get("VIDOPT_PORT", "29940")
for k, v in [("MASTER_ADDR", "localhost"), ("MASTER_PORT", _port),
             ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]:
    os.environ[k] = v
if not dist.is_initialized():
    dist.init_process_group(backend="hccl")
from omegaconf import OmegaConf
from hydra.utils import instantiate
from safetensors.torch import load_file, save_file
from groot.vla.common.utils.device import DEVICE

def find_config(ckpt_dir):
    for d in [os.path.join(ckpt_dir, "experiment_cfg"),
              os.path.join(os.path.dirname(ckpt_dir), "experiment_cfg"),
              "/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v3/experiment_cfg"]:
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d):
                for f in files:
                    if f in ("conf.yaml","config.yaml"): return os.path.join(root, f)
    raise FileNotFoundError

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = "/tmp/video_opt_results"
    os.makedirs(output_dir, exist_ok=True)

    # --- Load model ---
    cfg_file = find_config(args.checkpoint)
    cfg = OmegaConf.load(cfg_file)
    print(f"Config: {cfg_file}")

    model = instantiate(cfg.model); model.eval(); model.to(DEVICE)
    ah = model.action_head; ah.post_initialize()

    adapter = os.path.join(args.checkpoint, "adapter_model.safetensors")
    sd = load_file(adapter)
    stripped = {k.replace(".base_layer.", ".") if ".base_layer." in k else k: v for k, v in sd.items()}
    model.load_state_dict(stripped, strict=False)
    n_lora = sum(1 for k in stripped if 'lora_' in k)
    n_lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)
    print(f"LoRA: {n_lora} keys, {n_lora_params:,} params")

    # --- Prepare LoRA optimizer ---
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n]
    for p in lora_params: p.requires_grad = True
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
    print(f"Optimizer: AdamW, lr={args.lr}, {len(lora_params)} param groups")

    # --- Load dataset ---
    torch.manual_seed(args.seed)
    ds = instantiate(cfg.train_dataset)
    it = iter(ds)

    # --- Validation set (fixed 10 samples, same throughout) ---
    print("Loading validation set...")
    val_samples = [next(it) for _ in range(10)]
    from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer

    def prepare_sample(s):
        tk = HuggingfaceTokenizer(name="/checkpoints/umt5-xxl", seq_len=512, clean='whitespace')
        txt = s.get("text","") or ""
        ids, amask = tk([txt], return_mask=True)
        neg_ids, neg_mask = tk([""], return_mask=True)
        inp = {"images": s["images"], "state": s["state"], "action": s["action"],
               "text": ids, "text_attention_mask": amask,
               "text_negative": neg_ids, "text_attention_mask_negative": neg_mask,
               "embodiment_id": s["embodiment_id"],
               "has_real_action": s.get("has_real_action", True),
               "action_mask": s.get("action_mask", torch.ones(len(s["action"]),32))}
        _text_keys = {"text","text_attention_mask","text_negative","text_attention_mask_negative"}
        for k in list(inp.keys()):
            v = inp[k]
            if k in _text_keys:
                inp[k] = v.long().to(DEVICE) if isinstance(v, torch.Tensor) else torch.from_numpy(v).long().to(DEVICE)
            elif isinstance(v, torch.Tensor): inp[k] = v.to(DEVICE)
            elif isinstance(v, np.ndarray): inp[k] = torch.from_numpy(v).to(DEVICE) if v.ndim>0 else torch.tensor([v.item()],device=DEVICE)
            elif isinstance(v, (int,float,bool)): inp[k] = torch.tensor([int(v)if isinstance(v,bool)else v],device=DEVICE)
        return inp

    val_ins = [prepare_sample(s) for s in val_samples]

    def make_batch(inp):
        _text_keys = {"text","text_attention_mask","text_negative","text_attention_mask_negative"}
        batch = {}
        for k, v in inp.items():
            if k in _text_keys: batch[k] = v
            elif v.dim() == 0: batch[k] = v.unsqueeze(0)
            else: batch[k] = v.unsqueeze(0)
        return batch

    def evaluate():
        """Evaluate video_loss and action_loss on validation set."""
        model.eval()
        v_losses, a_losses, t_losses = [], [], []
        for inp in val_ins:
            with torch.no_grad():
                out = model.forward(make_batch(inp))
            v_losses.append(float(out["dynamics_loss"].cpu()))
            a_losses.append(float(out["action_loss"].cpu()))
            t_losses.append(float(out["loss"].cpu()))
        return float(np.mean(v_losses)), float(np.mean(a_losses)), float(np.mean(t_losses))

    # --- Baseline evaluation ---
    v0, a0, t0 = evaluate()
    print(f"\nStep   0: video_loss={v0:.4f}  action_loss={a0:.4f}  total_loss={t0:.4f}")

    # Save initial LoRA
    lora_before = {n: p.detach().clone() for n, p in model.named_parameters() if 'lora_' in n}

    # --- Training loop (video-only) ---
    history = [{"step": 0, "video_loss": v0, "action_loss": a0, "total_loss": t0}]
    train_v_losses = []

    print(f"\nTraining {args.train_steps} steps (video loss only)...")
    t_start = time.time()

    for step in range(1, args.train_steps + 1):
        # Get next training sample
        try:
            train_sample = next(it)
        except StopIteration:
            it = iter(ds)
            train_sample = next(it)

        train_inp = prepare_sample(train_sample)

        model.train()
        optimizer.zero_grad()
        out = model.forward(make_batch(train_inp))

        video_loss = out["dynamics_loss"]
        video_loss.backward()
        optimizer.step()

        train_v_losses.append(float(video_loss.cpu()))

        # Evaluate periodically
        if step % args.eval_interval == 0:
            model.eval()
            v_val, a_val, t_val = evaluate()
            history.append({"step": step, "video_loss": v_val, "action_loss": a_val, "total_loss": t_val})
            train_v_mean = float(np.mean(train_v_losses[-args.eval_interval:]))
            elapsed = time.time() - t_start
            print(f"Step {step:>4d}: video_loss={v_val:.4f}  action_loss={a_val:.4f}  total_loss={t_val:.4f}  "
                  f"train_v_loss={train_v_mean:.4f}  ({elapsed:.1f}s)")

    # --- Final ---
    elapsed = time.time() - t_start
    print(f"\nCompleted {args.train_steps} steps in {elapsed:.1f}s ({args.train_steps/elapsed:.1f} steps/s)")

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"{'Step':<8} {'video_loss':>12} {'action_loss':>12} {'total_loss':>12} {'Δ action %':>12}")
    print(f"{'-'*56}")
    for h in history:
        delta_pct = (h["action_loss"] - a0) / a0 * 100 if a0 > 0 else 0
        print(f"{h['step']:<8} {h['video_loss']:>12.4f} {h['action_loss']:>12.4f} {h['total_loss']:>12.4f} {delta_pct:>+11.2f}%")

    final_a = history[-1]["action_loss"]
    delta_a = final_a - a0
    pct = delta_a / a0 * 100 if a0 > 0 else 0

    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")
    print(f"  action_loss: {a0:.4f} → {final_a:.4f} (Δ={delta_a:+.4f}, {pct:+.2f}%)")
    print()

    if pct < -5:
        print(f"  ✅ ACTION IMPROVED by {abs(pct):.1f}%")
        print(f"  Video-only RL transfers to action via shared DiT.")
    elif abs(pct) < 5:
        print(f"  ≈ ACTION STABLE (change within 5%)")
        print(f"  Video optimization is safe — no catastrophic forgetting.")
        print(f"  But also no clear benefit from shared representation.")
    else:
        print(f"  ⚠️ ACTION DEGRADED by {pct:.1f}%")
        print(f"  Video-only optimization hurts action.")
        print(f"  Need mixing with action loss (λ constraint).")

    # --- Save ---
    summary = {
        "checkpoint": args.checkpoint, "train_steps": args.train_steps,
        "lr": args.lr,
        "action_before": a0, "action_after": final_a, "delta": delta_a, "delta_pct": pct,
        "history": history,
    }
    with open(os.path.join(output_dir, "video_opt_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {output_dir}/video_opt_results.json")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
