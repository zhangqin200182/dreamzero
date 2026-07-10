#!/usr/bin/env python3
"""
Trend analysis: aggregate batch_eval results and produce comparison plots.

Reads all_summaries.jsonl (produced by batch_eval.py) and generates:
  - trend_mse.png       : MSE vs training step (Full vs Action-Only)
  - trend_time.png      : Inference time vs step
  - trend_delta.png     : Relative MSE delta (%) vs step

Usage:
  python scripts/eval/trend_analysis.py --eval_root eval
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_summaries(eval_root):
    """Load all step_N/summary.json files, return sorted list."""
    summaries = []
    for entry in sorted(os.listdir(eval_root)):
        if not entry.startswith("step_"):
            continue
        path = os.path.join(eval_root, entry, "summary.json")
        if os.path.isfile(path):
            with open(path) as f:
                summaries.append(json.load(f))
    summaries.sort(key=lambda s: s["step"])
    return summaries


def plot_mse_trend(summaries, output_path):
    """MSE vs training step for Full and Action-Only modes."""
    steps = [s["step"] for s in summaries]
    full_mse = [s["full"]["action_mse_mean"] for s in summaries]
    ao_mse = [s["action_only"]["action_mse_mean"] for s in summaries]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, full_mse, 'b-o', label='Full (video+action)', linewidth=2, markersize=6)
    ax.plot(steps, ao_mse, 'r-s', label='Action-Only', linewidth=2, markersize=6)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Action MSE', fontsize=12)
    ax.set_title('Action Prediction MSE vs Training Progress', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate best Full MSE
    if full_mse and any(m is not None for m in full_mse):
        valid = [(s, m) for s, m in zip(steps, full_mse) if m is not None]
        if valid:
            best_step, best_mse = min(valid, key=lambda x: x[1])
            ax.annotate(f'Best Full: {best_mse:.2f} @ step {best_step}',
                        xy=(best_step, best_mse), fontsize=9, color='blue',
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_time_trend(summaries, output_path):
    """Inference time vs step."""
    steps = [s["step"] for s in summaries]
    full_time = [s["full"]["time_mean_sec"] for s in summaries]
    ao_time = [s["action_only"]["time_mean_sec"] for s in summaries]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, full_time, 'b-o', label='Full (video+action)', linewidth=2, markersize=6)
    ax.plot(steps, ao_time, 'r-s', label='Action-Only', linewidth=2, markersize=6)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Inference Time (s)', fontsize=12)
    ax.set_title('Inference Time vs Training Progress', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if full_time and ao_time:
        valid_pairs = [(ft, at) for ft, at in zip(full_time, ao_time) if ft is not None and at is not None]
        if valid_pairs:
            avg_speedup = np.mean([ft / at for ft, at in valid_pairs])
            ax.text(0.98, 0.05, f'Avg speedup (AO/Full): {avg_speedup:.2f}x',
                    transform=ax.transAxes, ha='right', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_delta_trend(summaries, output_path):
    """Relative MSE delta (%) between Full and Action-Only."""
    steps = []
    deltas = []
    for s in summaries:
        fm = s["full"]["action_mse_mean"]
        am = s["action_only"]["action_mse_mean"]
        if fm and am and fm > 0:
            steps.append(s["step"])
            deltas.append((am - fm) / fm * 100)

    if not steps:
        print("No valid delta data")
        return

    colors = ['green' if d <= 50 else 'orange' if d <= 100 else 'red' for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(steps, deltas, color=colors, width=max(20, (max(steps)-min(steps))/len(steps)*0.6))
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('MSE Increase (%)', fontsize=12)
    ax.set_title('Action-Only vs Full: MSE Increase % (lower=better for AO)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{d:.1f}%', ha='center', va='bottom', fontsize=8)

    # Legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='≤50% — AO competitive'),
        Patch(facecolor='orange', label='50-100% — moderate gap'),
        Patch(facecolor='red', label='>100% — large gap'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_table(summaries):
    """Print a formatted summary table."""
    print(f"\n{'='*80}")
    print(f"{'Step':<8} {'Full MSE':>12} {'AO MSE':>12} {'Δ MSE %':>10} {'Full Time':>11} {'AO Time':>11} {'Speedup':>8}")
    print(f"{'-'*80}")
    for s in summaries:
        fm = s["full"]["action_mse_mean"]
        am = s["action_only"]["action_mse_mean"]
        ft = s["full"]["time_mean_sec"]
        at = s["action_only"]["time_mean_sec"]

        fm_str = f"{fm:.4f}" if fm else "N/A"
        am_str = f"{am:.4f}" if am else "N/A"
        ft_str = f"{ft:.1f}s" if ft else "N/A"
        at_str = f"{at:.1f}s" if at else "N/A"

        if fm and am and fm > 0:
            delta = f"{(am-fm)/fm*100:+.1f}%"
        else:
            delta = "N/A"

        if ft and at and at > 0:
            speedup = f"{ft/at:.2f}x"
        else:
            speedup = "N/A"

        print(f"{s['step']:<8} {fm_str:>12} {am_str:>12} {delta:>10} {ft_str:>11} {at_str:>11} {speedup:>8}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", default="eval",
                        help="Root directory containing step_N/ subdirs with summary.json")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plots, print table only")
    args = parser.parse_args()

    summaries = load_summaries(args.eval_root)
    if not summaries:
        print(f"No summaries found in {args.eval_root}")
        return

    print(f"Loaded {len(summaries)} checkpoint summaries")

    # Print table
    print_table(summaries)

    if args.no_plot:
        return

    # Generate plots
    plot_mse_trend(summaries, os.path.join(args.eval_root, "trend_mse.png"))
    plot_time_trend(summaries, os.path.join(args.eval_root, "trend_time.png"))
    plot_delta_trend(summaries, os.path.join(args.eval_root, "trend_delta.png"))

    # Final verdict
    steps_with_data = [s for s in summaries
                       if s["full"]["action_mse_mean"] and s["action_only"]["action_mse_mean"]]
    if steps_with_data:
        best = min(steps_with_data, key=lambda s: s["full"]["action_mse_mean"])
        print(f"\n★ Best checkpoint: step {best['step']} (Full MSE={best['full']['action_mse_mean']:.4f})")

        ao_competitive = [s for s in steps_with_data
                          if s["full"]["action_mse_mean"] and s["full"]["action_mse_mean"] > 0
                          and (s["action_only"]["action_mse_mean"] - s["full"]["action_mse_mean"]) / s["full"]["action_mse_mean"] < 0.50]
        if ao_competitive:
            print(f"★ Action-Only is competitive (Δ<50%) at {len(ao_competitive)}/{len(steps_with_data)} checkpoints")


if __name__ == "__main__":
    main()
