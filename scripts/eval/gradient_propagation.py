#!/usr/bin/env python3
"""
Gradient Propagation Test: Does video-only optimization affect action?

1. Load checkpoint + a batch of samples
2. Forward pass → record action_pred_BEFORE
3. Compute video_loss only (ignore action_loss) → backward → update LoRA
4. Forward pass again → record action_pred_AFTER
5. If action changed → gradients propagate through shared DiT → RL CAN work
"""
import os, sys, json, time, argparse
import numpy as np

os.environ["DREAMZERO_DEVICE"] = "npu"
sys.path.insert(0, "/workspace/dreamzero")
import torch, torch.distributed as dist
torch._dynamo.config.disable = True
_port = os.environ.get("GRAD_PORT", "29910")
for k, v in [("MASTER_ADDR", "localhost"), ("MASTER_PORT", _port),
             ("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")]:
    os.environ[k] = v
if not dist.is_initialized():
    dist.init_process_group(backend="hccl")
from omegaconf import OmegaConf
from hydra.utils import instantiate
from safetensors.torch import load_file
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = "/tmp/gradient_test"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    cfg_file = find_config(args.checkpoint)
    cfg = OmegaConf.load(cfg_file)
    print(f"Config: {cfg_file}")

    model = instantiate(cfg.model); model.eval(); model.to(DEVICE)
    ah = model.action_head; ah.post_initialize()

    adapter = os.path.join(args.checkpoint, "adapter_model.safetensors")
    sd = load_file(adapter)
    stripped = {k.replace(".base_layer.", ".") if ".base_layer." in k else k: v for k, v in sd.items()}
    model.load_state_dict(stripped, strict=False)
    print(f"Loaded LoRA: {sum(1 for k in stripped if 'lora_' in k)} keys")

    # Load samples
    torch.manual_seed(args.seed)
    ds = instantiate(cfg.train_dataset)
    it = iter(ds)
    samples = [next(it) for _ in range(args.batch_size)]

    from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer
    inps, gts = [], []
    for s in samples:
        raw = s["images"]
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
        # Prepare for training forward: the model.forward() calls prepare_input()
        # which expects raw dataset format. Just add batch dim to all tensors.
        _text_keys = {"text","text_attention_mask","text_negative","text_attention_mask_negative"}
        for k in list(inp.keys()):
            v = inp[k]
            if k in _text_keys:
                # Text tokens: keep as [1, seq_len] tensor, ensure long dtype
                if isinstance(v, np.ndarray):
                    inp[k] = torch.from_numpy(v).long().to(DEVICE)
                else:
                    inp[k] = v.long().to(DEVICE)
            elif isinstance(v, torch.Tensor):
                inp[k] = v.to(DEVICE)
            elif isinstance(v, np.ndarray):
                inp[k] = torch.from_numpy(v).to(DEVICE) if v.ndim>0 else torch.tensor([v.item()],device=DEVICE)
            elif isinstance(v, (int,float,bool)):
                inp[k] = torch.tensor([int(v)if isinstance(v,bool)else v],device=DEVICE)
        inps.append(inp)
        gts.append(s["action"].float().numpy() if hasattr(s["action"],'numpy') else np.asarray(s["action"]))

    # Find all LoRA parameters
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
            param.requires_grad = True
    print(f"LoRA parameters: {len(lora_params)} total, {sum(p.numel() for p in lora_params):,} elements")

    _text_keys = {"text","text_attention_mask","text_negative","text_attention_mask_negative"}
    def make_batch(inp):
        batch = {}
        for k, v in inp.items():
            if k in _text_keys:
                batch[k] = v  # already [1, seq], keep as-is
            elif v.dim() == 0:
                batch[k] = v.unsqueeze(0)
            else:
                batch[k] = v.unsqueeze(0) if v.dim() > 0 else v.unsqueeze(0)
        return batch

    # Record BEFORE action predictions
    before_mses = []
    for i, inp in enumerate(inps):
        inp_batch = make_batch(inp)
        with torch.no_grad():
            out = model.forward(inp_batch)
        before_mses.append(float(out["action_loss"].cpu()))
    before_mean = float(np.mean(before_mses))
    print(f"\nBEFORE: action_loss mean = {before_mean:.6f}")

    # Save LoRA weights before update
    lora_before = {name: param.detach().clone() for name, param in model.named_parameters() if 'lora_' in name}

    # Now: compute video_loss ONLY, update LoRA
    print(f"\nOptimizing VIDEO LOSS ONLY for {len(inps)} batches...")
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)

    video_losses_before = []
    video_losses_after = []
    total_grad_norm = 0.0

    for i, inp in enumerate(inps):
        optimizer.zero_grad()
        inp_batch = make_batch(inp)

        # Forward pass (training mode)
        model.train()
        out = model.forward(inp_batch)
        model.eval()

        video_loss_before = float(out["dynamics_loss"].cpu())
        video_losses_before.append(video_loss_before)

        # Backward on VIDEO loss only
        out["dynamics_loss"].backward()

        # Track grad norm
        grad_norm = sum(p.grad.norm().item()**2 for p in lora_params if p.grad is not None)**0.5
        total_grad_norm += grad_norm

        optimizer.step()
        video_losses_after.append(video_loss_before)  # same batch, loss would have changed if we re-ran

        print(f"  batch {i}: video_loss={video_loss_before:.6f}, grad_norm={grad_norm:.6f}")

    avg_grad_norm = total_grad_norm / len(inps)
    print(f"\n  Total LoRA updates: {len(inps)} steps, avg grad_norm: {avg_grad_norm:.6f}, lr: {args.lr}")
    print(f"  Effective weight change magnitude: {avg_grad_norm * args.lr:.6f}")

    # Record AFTER action predictions
    after_mses = []
    for i, inp in enumerate(inps):
        inp_batch = make_batch(inp)
        with torch.no_grad():
            out = model.forward(inp_batch)
        after_mses.append(float(out["action_loss"].cpu()))
    after_mean = float(np.mean(after_mses))
    print(f"\nAFTER:  action_loss mean = {after_mean:.6f}")

    # Also measure per-parameter change
    lora_changes = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            delta = (param - lora_before[name]).norm().item()
            lora_changes[name] = delta

    top_changes = sorted(lora_changes.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 changed LoRA params:")
    for name, delta in top_changes:
        print(f"  {name}: Δ={delta:.6f}")

    # -----------------------------------------------------------------
    # FINAL VERDICT
    # -----------------------------------------------------------------
    delta_action = after_mean - before_mean
    pct_change = abs(delta_action) / before_mean * 100 if before_mean > 0 else 0

    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")
    print(f"  action_loss BEFORE: {before_mean:.6f}")
    print(f"  action_loss AFTER:  {after_mean:.6f}")
    print(f"  Δ action_loss:      {delta_action:+.6f} ({pct_change:.2f}%)")
    print(f"  Avg LoRA change:    {avg_grad_norm * args.lr:.6f}")
    print()

    if pct_change > 1.0:
        print(f"  ✓ SIGNIFICANT: video-only optimization changes action by {pct_change:.1f}%")
        print(f"    → Gradients propagate through shared DiT → RL CAN work!")
    elif pct_change > 0.1:
        print(f"  ~ WEAK: video-only optimization changes action by {pct_change:.2f}%")
        print(f"    → Some coupling exists, RL may work with more steps")
    else:
        print(f"  ✗ NEGLIGIBLE: video-only optimization does NOT change action")
        print(f"    → Shared DiT representations are decoupled → RL unlikely to work")

    # Save
    summary = {
        "checkpoint": args.checkpoint, "batch_size": args.batch_size, "lr": args.lr,
        "before_mean": before_mean, "after_mean": after_mean,
        "delta": delta_action, "pct_change": pct_change,
        "avg_grad_norm": avg_grad_norm,
        "top_changes": top_changes[:5],
    }
    with open(os.path.join(output_dir, "gradient_test.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {output_dir}/gradient_test.json")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
