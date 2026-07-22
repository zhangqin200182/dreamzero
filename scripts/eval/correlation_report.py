#!/usr/bin/env python3
"""
Aggregate and visualize Video-Action correlation results from batch analysis.

Usage: python3 correlation_report.py <results_dir>
  results_dir should contain per-checkpoint *_correlation.json files
  and optionally an all_correlations.jsonl summary file.
"""
import json, sys, os
from pathlib import Path
import numpy as np

def load_results(results_dir):
    """Load all correlation JSON files from a directory."""
    results = []
    for f in sorted(Path(results_dir).glob("*_correlation.json")):
        with open(f) as fp:
            data = json.load(fp)
            results.append(data)
    return results

def load_summary(summary_file):
    """Load all_correlations.jsonl file."""
    results = []
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    print(f"Loading results from {results_dir}...")

    # Try all_correlations.jsonl first (aggregate summary)
    summary_file = os.path.join(os.path.dirname(results_dir), "all_correlations.jsonl")
    summary = load_summary(summary_file)

    if summary:
        results = summary
        print(f"Loaded {len(results)} checkpoints from summary file")
    else:
        results = load_results(results_dir)
        print(f"Loaded {len(results)} checkpoints from individual files")

    if not results:
        print("ERROR: No results found!")
        sys.exit(1)

    # Print detailed table
    print(f"\n{'Step':<15} {'N':<6} {'Pearson r':<12} {'p-val':<10} {'Spearman r':<12} {'p-val':<10} {'Dyn Mean':<12} {'Act Mean':<12}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: (x.get('step', ''), x.get('checkpoint', ''))):
        s = r.get('step', os.path.basename(r.get('checkpoint', '')))
        n = r.get('n', 0)
        pr = r.get('pearson_r')
        pp = r.get('pearson_p')
        sr = r.get('spearman_r')
        sp = r.get('spearman_p')
        dm = r.get('dynamics_loss_mean')
        am = r.get('action_loss_mean')
        print(f"{s:<15} {n:<6} {pr or 'N/A':<12} {pp or 'N/A':<10} {sr or 'N/A':<12} {sp or 'N/A':<10} {dm or 'N/A':<12} {am or 'N/A':<12}")

    # Significance summary
    significant = [r for r in results if r.get('pearson_p') is not None and r['pearson_p'] < 0.05]
    non_sig = [r for r in results if r.get('pearson_p') is not None and r['pearson_p'] >= 0.05]

    print(f"\n=== Significance Summary ===")
    print(f"  Significant (p<0.05): {len(significant)}/{len(results)}")
    print(f"  Not significant:      {len(non_sig)}/{len(results)}")

    if significant:
        pearson_rs = [r['pearson_r'] for r in significant if r.get('pearson_r') is not None]
        spearman_rs = [r['spearman_r'] for r in significant if r.get('spearman_r') is not None]
        print(f"  Significant Pearson r range:  [{min(pearson_rs):.3f}, {max(pearson_rs):.3f}]")
        print(f"  Significant Spearman r range: [{min(spearman_rs):.3f}, {max(spearman_rs):.3f}]")

    # Meta-analysis
    pearson_rs = [r['pearson_r'] for r in results if r.get('pearson_r') is not None]
    spearman_rs = [r['spearman_r'] for r in results if r.get('spearman_r') is not None]
    total_samples = sum(r.get('n', 0) for r in results)

    if pearson_rs:
        print(f"\n=== Meta-Analysis ===")
        print(f"  Checkpoints:  {len(results)}")
        print(f"  Total samples: {total_samples}")
        print(f"  Pearson r:  mean={np.mean(pearson_rs):.4f}, std={np.std(pearson_rs):.4f}, "
              f"min={np.min(pearson_rs):.4f}, max={np.max(pearson_rs):.4f}")
        if spearman_rs:
            print(f"  Spearman r: mean={np.mean(spearman_rs):.4f}, std={np.std(spearman_rs):.4f}, "
                  f"min={np.min(spearman_rs):.4f}, max={np.max(spearman_rs):.4f}")

        # Overall verdict
        mean_r = np.mean(pearson_rs)
        sig_count = len(significant)
        total = len(results)

        print(f"\n{'='*60}")
        print("OVERALL VERDICT:")
        if sig_count >= total * 0.5 and mean_r > 0.2:
            print(f"✓ SIGNIFICANT POSITIVE: {sig_count}/{total} checkpoints show significant")
            print(f"  positive correlation (mean Pearson r={mean_r:.3f}).")
            print(f"  → Video prediction quality is LINKED to action prediction quality.")
            print(f"  → RL on video generation via shared DiT SHOULD improve action prediction.")
        elif sig_count >= total * 0.5 and mean_r < -0.2:
            print(f"✗ SIGNIFICANT NEGATIVE: {sig_count}/{total} checkpoints show significant")
            print(f"  negative correlation (mean Pearson r={mean_r:.3f}).")
            print(f"  → Better video = WORSE action? This is surprising and needs investigation.")
        elif mean_r > 0.1:
            print(f"~ WEAK POSITIVE: mean Pearson r={mean_r:.3f}, {sig_count}/{total} significant.")
            print(f"  → Marginal relationship. RL on video MAY help action, but effect is small.")
            print(f"  → Consider: test with more checkpoints or different video quality metrics.")
        elif mean_r < -0.1:
            print(f"~ WEAK NEGATIVE: mean Pearson r={mean_r:.3f}, {sig_count}/{total} significant.")
            print(f"  → Video and action prediction may TRADE OFF against each other in shared DiT.")
            print(f"  → RL on video generation could DEGRADE action prediction.")
        else:
            print(f"≈ NO CORRELATION: mean Pearson r={mean_r:.3f}, {sig_count}/{total} significant.")
            print(f"  → Video and action prediction quality appear INDEPENDENT.")
            print(f"  → RL on video generation is UNLIKELY to improve action prediction.")
            print(f"  → Action quality depends on factors OTHER than shared visual representation quality.")
        print(f"{'='*60}")

    # Load per-sample data for scatter plot (if available)
    per_sample_data = []
    for r in results:
        checkpoint_path = r.get('checkpoint', '')
        step = r.get('step', os.path.basename(checkpoint_path))
        result_file = os.path.join(results_dir, f"{os.path.basename(checkpoint_path)}_correlation.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                detailed = json.load(f)
                per_sample = detailed.get('per_sample', [])
                for ps in per_sample:
                    ps['step'] = step
                per_sample_data.extend(per_sample)

    if per_sample_data:
        dyn_losses = [d['dynamics_loss'] for d in per_sample_data]
        act_losses = [d['action_loss'] for d in per_sample_data]
        steps = [d['step'] for d in per_sample_data]

        # Print per-checkpoint stats
        unique_steps = sorted(set(steps))
        print(f"\n=== Per-Checkpoint Detail ===")
        for step in unique_steps:
            step_data = [d for d in per_sample_data if d['step'] == step]
            dyn = [d['dynamics_loss'] for d in step_data]
            act = [d['action_loss'] for d in step_data]
            print(f"  {step}: n={len(step_data)}, dyn_loss={np.mean(dyn):.6f}±{np.std(dyn):.6f}, "
                  f"act_loss={np.mean(act):.6f}±{np.std(act):.6f}")


if __name__ == "__main__":
    main()
