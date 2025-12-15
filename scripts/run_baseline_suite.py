#!/usr/bin/env python3
"""Curated end-to-end suite: baseline + ablations with strong attack settings.

Runs a concise set of experiments in an efficient order and produces a
summary CSV and simple plots suitable for diagrams and comparison studies.
"""

import argparse
import csv
import os
import subprocess
from pathlib import Path

import itertools
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Run baseline + curated ablations")
    p.add_argument('--output-root', type=str, default='results/suite')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-rounds', type=int, default=5)
    p.add_argument('--data-subset', type=int, default=200)
    p.add_argument('--client-lr', type=float, default=0.01)
    p.add_argument('--local-epochs', type=int, default=1)
    p.add_argument('--no-augment', action='store_true')
    # Attack defaults tuned for visual quality + speed
    p.add_argument('--attack-iterations', type=int, default=2000)
    p.add_argument('--attack-lr', type=float, default=0.1)
    p.add_argument('--tv-weight', type=float, default=1e-3)
    p.add_argument('--attack-restarts', type=int, default=3)
    p.add_argument('--label-strategy', type=str, default='auto')
    p.add_argument('--attack-optimizer', type=str, default='adam')
    p.add_argument('--attack-batch', type=int, default=1)
    # Enhancements
    p.add_argument('--lr-schedule', type=str, default='cosine')
    p.add_argument('--early-stop', action='store_true')
    p.add_argument('--patience', type=int, default=600)
    p.add_argument('--min-delta', type=float, default=1e-4)
    p.add_argument('--fft-init', action='store_true')
    p.add_argument('--preset', type=str, default='soft')
    p.add_argument('--match-metric', type=str, default='both')
    p.add_argument('--l2-weight', type=float, default=1e-2)
    p.add_argument('--cos-weight', type=float, default=1.0)
    p.add_argument('--select-by-name', type=str, nargs='+', default=None)
    p.add_argument('--layer-weights', type=str, default='auto')
    p.add_argument('--compute-lpips', action='store_true')
    p.add_argument('--no-heatmap', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def run_case(name, out_dir, extra_cmd):
    cmd = ['python', 'run_experiment.py', '--out-dir', str(out_dir), '--save-config'] + extra_cmd
    print(f"\n[SUITE] {name}\n Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def metrics_from(out_dir):
    mpath = Path(out_dir) / 'metrics.txt'
    out = {}
    if not mpath.exists():
        return out
    for line in mpath.read_text().splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        out[k.strip()] = v.strip()
    return out


def plot_series(rows, metric, key_fields, output_root):
    # key_fields: tuple of (x_field, group_field)
    x_field, group_field = key_fields
    groups = sorted({row[group_field] for row in rows})
    plt.figure(figsize=(6, 4))
    plotted = False
    for g in groups:
        series = [(row[x_field], float(row.get(metric))) for row in rows if row[group_field] == g and metric in row]
        if not series:
            continue
        series.sort(key=lambda x: x[0])
        xs, ys = zip(*series)
        plt.plot(xs, ys, marker='o', label=f"{group_field}={g}")
        plotted = True
    if plotted:
        plt.title(f"{metric} vs {x_field}")
        plt.xlabel(x_field)
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        outp = Path(output_root) / f"suite_{metric}_vs_{x_field}.png"
        plt.tight_layout(); plt.savefig(outp, dpi=150); plt.close()
        print(f"Saved plot: {outp}")


def main():
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    base_flags = [
        '--seed', str(args.seed),
        '--num-rounds', str(args.num_rounds),
        '--data-subset', str(args.data_subset),
        '--client-lr', str(args.client_lr),
        '--local-epochs', str(args.local_epochs),
        '--attack-iterations', str(args.attack_iterations),
        '--attack-lr', str(args.attack_lr),
        '--tv-weight', str(args.tv_weight),
        '--attack-restarts', str(args.attack_restarts),
        '--label-strategy', args.label_strategy,
        '--attack-optimizer', args.attack_optimizer,
        '--attack-batch', str(args.attack_batch),
        '--lr-schedule', args.lr_schedule,
        '--patience', str(args.patience),
        '--min-delta', str(args.min_delta),
        '--preset', str(args.preset),
        '--match-metric', args.match_metric,
        '--l2-weight', str(args.l2_weight),
        '--cos-weight', str(args.cos_weight),
    ]
    if args.device:
        base_flags += ['--device', args.device]
    if args.no_augment:
        base_flags.append('--no-augment')
    if args.early_stop:
        base_flags.append('--early-stop')
    if args.fft_init:
        base_flags.append('--fft-init')
    if args.select_by_name:
        base_flags += ['--select-by-name'] + list(args.select_by_name)
    if args.layer_weights:
        base_flags += ['--layer-weights', str(args.layer_weights)]
    if args.compute_lpips:
        base_flags.append('--compute-lpips')
    if args.no_heatmap:
        base_flags.append('--no-heatmap')

    rows = []

    # 1) Baseline: gradients, bs=1, mom=0.0
    name = 'baseline_grad_bs1_mom0.0'
    out_dir = root / name
    flags = base_flags + ['--batch-size', '1', '--client-momentum', '0.0', '--attack-source', 'gradients']
    if not args.dry_run:
        run_case(name, out_dir, flags)
    m = metrics_from(out_dir); m.update({'case': name, 'batch_size': 1, 'momentum': 0.0, 'attack_source': 'gradients', 'output_dir': str(out_dir)})
    rows.append(m)

    # 2) Ablation: batch size impact (gradients), momentum in {0.0, 0.9}
    for bs, mom in itertools.product([1, 2, 4], [0.0, 0.9]):
        name = f'abl_grad_bs{bs}_mom{mom}'
        out_dir = root / name
        flags = base_flags + ['--batch-size', str(bs), '--client-momentum', str(mom), '--attack-source', 'gradients']
        if not args.dry_run:
            run_case(name, out_dir, flags)
        m = metrics_from(out_dir); m.update({'case': name, 'batch_size': bs, 'momentum': mom, 'attack_source': 'gradients', 'output_dir': str(out_dir)})
        rows.append(m)

    # 3) Ablation: attack source variants at bs=1, mom=0.0
    for src in ['one_step_update', 'agg_update']:
        name = f'abl_src_{src}_bs1_mom0.0'
        out_dir = root / name
        flags = base_flags + ['--batch-size', '1', '--client-momentum', '0.0', '--attack-source', src]
        if not args.dry_run:
            run_case(name, out_dir, flags)
        m = metrics_from(out_dir); m.update({'case': name, 'batch_size': 1, 'momentum': 0.0, 'attack_source': src, 'output_dir': str(out_dir)})
        rows.append(m)

    # Save summary
    summary_path = Path(args.output_root) / 'suite_summary.csv'
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"Saved summary: {summary_path}")

    # Plots for diagrams: MSE and LPIPS vs batch_size for gradients
    rows_grad = [r for r in rows if r.get('attack_source') == 'gradients']
    plot_series(rows_grad, 'MSE', ('batch_size', 'momentum'), args.output_root)
    if any('LPIPS' in r for r in rows_grad):
        plot_series(rows_grad, 'LPIPS', ('batch_size', 'momentum'), args.output_root)


if __name__ == '__main__':
    main()

