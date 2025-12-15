#!/usr/bin/env python3
"""Simple sweep runner for baseline gradient inversion experiments."""

import argparse
import csv
import itertools
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep baseline FL attack settings")
    parser.add_argument('--output-root', type=str, default='results/sweeps',
                        help='Root directory to store per-run outputs and summary csv')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--momenta', type=float, nargs='+', default=[0.0, 0.9])
    parser.add_argument('--attack-sources', type=str, nargs='+',
                        default=['gradients', 'one_step_update', 'agg_update'])
    parser.add_argument('--attack-batch', type=int, default=1)
    parser.add_argument('--num-rounds', type=int, default=5)
    parser.add_argument('--capture-round', type=int, default=None,
                        help='Round index to capture on (defaults to last)')
    parser.add_argument('--data-subset', type=int, default=200)
    parser.add_argument('--client-lr', type=float, default=0.01)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--attack-iterations', type=int, default=2000)
    parser.add_argument('--attack-lr', type=float, default=0.1)
    parser.add_argument('--tv-weight', type=float, default=0.001)
    parser.add_argument('--attack-restarts', type=int, default=1)
    parser.add_argument('--label-strategy', type=str, default='auto')
    # Enhancements
    parser.add_argument('--lr-schedule', type=str, default='cosine')
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--patience', type=int, default=600)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--fft-init', action='store_true')
    parser.add_argument('--preset', type=str, default='soft')
    parser.add_argument('--match-metric', type=str, default='both')
    parser.add_argument('--l2-weight', type=float, default=1e-2)
    parser.add_argument('--cos-weight', type=float, default=1.0)
    parser.add_argument('--use-layers', type=int, nargs='+', default=None)
    parser.add_argument('--select-by-name', type=str, nargs='+', default=None)
    parser.add_argument('--layer-weights', type=str, default='auto')
    parser.add_argument('--compute-lpips', action='store_true')
    parser.add_argument('--no-heatmap', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--plot-metric', type=str, default='MSE',
                        help='If set, generate simple line plots for this metric (set to "none" to disable)')
    parser.add_argument('--extra-args', nargs=argparse.REMAINDER,
                        help='Additional args passed to run_experiment.py after "--"')
    return parser.parse_args()


def read_metrics(metrics_path):
    metrics = {}
    if not os.path.exists(metrics_path):
        return metrics
    with open(metrics_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            metrics[key.strip()] = value.strip()
    return metrics


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(args.batch_sizes, args.momenta, args.attack_sources))
    rows = []

    for batch_size, momentum, attack_source in combos:
        combo_name = f"bs{batch_size}_mom{momentum:.2f}_src{attack_source}"
        run_dir = output_root / combo_name
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            'python', 'run_experiment.py',
            '--batch-size', str(batch_size),
            '--client-momentum', str(momentum),
            '--attack-source', attack_source,
            '--attack-batch', str(args.attack_batch),
            '--num-rounds', str(args.num_rounds),
            '--data-subset', str(args.data_subset),
            '--client-lr', str(args.client_lr),
            '--local-epochs', str(args.local_epochs),
            '--attack-iterations', str(args.attack_iterations),
            '--attack-lr', str(args.attack_lr),
            '--tv-weight', str(args.tv_weight),
            '--attack-restarts', str(args.attack_restarts),
            '--label-strategy', args.label_strategy,
            '--out-dir', str(run_dir),
            '--save-config',
            '--lr-schedule', args.lr_schedule,
            '--patience', str(args.patience),
            '--min-delta', str(args.min_delta),
            '--preset', str(args.preset),
            '--match-metric', args.match_metric,
            '--l2-weight', str(args.l2_weight),
            '--cos-weight', str(args.cos_weight),
        ]

        if args.capture_round is not None:
            cmd += ['--capture-round', str(args.capture_round)]
        if args.device:
            cmd += ['--device', args.device]
        if args.no_augment:
            cmd.append('--no-augment')
        if args.early_stop:
            cmd.append('--early-stop')
        if args.fft_init:
            cmd.append('--fft-init')
        if args.use_layers:
            cmd += ['--use-layers'] + [str(x) for x in args.use_layers]
        if args.select_by_name:
            cmd += ['--select-by-name'] + list(args.select_by_name)
        if args.layer_weights:
            cmd += ['--layer-weights', str(args.layer_weights)]
        if args.compute_lpips:
            cmd.append('--compute-lpips')
        if args.no_heatmap:
            cmd.append('--no-heatmap')
        if args.extra_args:
            cmd += args.extra_args

        print(f"Running sweep case: {combo_name}")
        print('Command:', ' '.join(cmd))

        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)

        metrics_path = run_dir / 'metrics.txt'
        metrics = read_metrics(metrics_path)
        row = {
            'batch_size': batch_size,
            'momentum': momentum,
            'attack_source': attack_source,
            'attack_batch': args.attack_batch,
            'output_dir': str(run_dir),
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    summary_path = output_root / 'sweep_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary to {summary_path}")

    metric_name = args.plot_metric
    if metric_name and metric_name.lower() != 'none':
        momenta = sorted({row['momentum'] for row in rows})
        attack_sources = sorted({row['attack_source'] for row in rows})
        for mom in momenta:
            plt.figure(figsize=(6, 4))
            plotted = False
            for attack_source in attack_sources:
                series = []
                for row in rows:
                    if row['momentum'] == mom and row['attack_source'] == attack_source:
                        metric_value = row.get(metric_name)
                        if metric_value is None:
                            continue
                        try:
                            metric_float = float(metric_value)
                        except ValueError:
                            continue
                        series.append((row['batch_size'], metric_float))
                if not series:
                    continue
                series.sort(key=lambda x: x[0])
                xs, ys = zip(*series)
                plt.plot(xs, ys, marker='o', label=attack_source)
                plotted = True
            if not plotted:
                plt.close()
                continue
            plt.title(f"{metric_name} vs batch size (momentum={mom:.2f})")
            plt.xlabel('Client batch size')
            plt.ylabel(metric_name)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plot_path = output_root / f"plot_{metric_name}_mom{mom:.2f}.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved plot: {plot_path}")


if __name__ == '__main__':
    main()
