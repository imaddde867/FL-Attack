#!/usr/bin/env python3
"""
make_report.py - Generate a comprehensive research outcome report from FL experiment artifacts.

Usage: 
  python scripts/make_report.py           # Standard report
  python scripts/make_report.py --poster  # Standard report + 4K poster

Outputs:
  results/report/summary.csv       - Full dataframe of all parsed metrics
  results/report/summary.md        - Human-readable markdown report
  results/report/figs/             - Publication-quality PNG figures
  results/report/montages/         - Image montages for slides
  results/report/poster_4k.png     - 3840x2160 UHD showcase poster (with --poster flag)

Dependencies: stdlib + pandas + matplotlib + pillow
"""

import argparse
import io
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================================
# Configuration
# ============================================================================

RESULTS_ROOT = Path("results")
REPORT_DIR = RESULTS_ROOT / "report"
FIGS_DIR = REPORT_DIR / "figs"
MONTAGES_DIR = REPORT_DIR / "montages"

# Input folder patterns
INPUT_FOLDERS = {
    "showcase": RESULTS_ROOT / "showcase",
    "multi_client": RESULTS_ROOT / "multi_client",
    "ablation": RESULTS_ROOT / "ablation",
    "defenses": RESULTS_ROOT / "defenses",
}

# Expected metric keys (will discover others dynamically)
KNOWN_METRICS = ["MSE", "PSNR", "SSIM", "LPIPS", "LabelMatch"]

# Montage settings
MONTAGE_THUMB_SIZE = (300, 300)
MONTAGE_TOP_K = 12
FONT_SIZE = 14

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Colors for plots
COLORS = plt.cm.Set2.colors

# ============================================================================
# Parsing Functions
# ============================================================================

def parse_metrics_file(filepath: Path) -> Dict[str, float]:
    """Parse a metrics.txt file into a dict of metric_name -> value."""
    metrics = {}
    if not filepath.exists():
        return metrics
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                try:
                    metrics[key] = float(val)
                except ValueError:
                    metrics[key] = float('nan')
    except Exception as e:
        print(f"  Warning: Could not parse {filepath}: {e}")
    
    return metrics


def find_png_in_dir(dirpath: Path) -> Optional[Path]:
    """Find the attack result PNG in a directory."""
    candidates = [
        "baseline_attack_result.png",
        "attack_result.png",
        "result.png",
    ]
    for name in candidates:
        p = dirpath / name
        if p.exists():
            return p
    # Fallback: any PNG
    pngs = list(dirpath.glob("*.png"))
    return pngs[0] if pngs else None


def extract_client_from_path(path: Path) -> Optional[str]:
    """Extract client identifier from path if present."""
    # Patterns: bmk_c0, c0, client_0, etc.
    name = path.name
    match = re.search(r'(?:bmk_)?c(\d+)', name, re.IGNORECASE)
    if match:
        return f"c{match.group(1)}"
    
    # Check parent
    match = re.search(r'(?:bmk_)?c(\d+)', path.parent.name, re.IGNORECASE)
    if match:
        return f"c{match.group(1)}"
    
    return None


def derive_setting_name(path: Path, group: str) -> str:
    """Derive a human-readable setting name from path."""
    parts = path.relative_to(RESULTS_ROOT).parts
    
    if group == "showcase":
        return "showcase"
    elif group == "multi_client":
        # e.g., multi_client/bmk_c0 -> bmk_c0
        return parts[-1] if len(parts) > 1 else "multi_client"
    elif group == "ablation":
        # e.g., ablation/baseline/c0 -> baseline
        if len(parts) >= 2:
            return parts[1]
        return "ablation"
    elif group == "defenses":
        # e.g., defenses/dp_eps1 -> dp_eps1
        if len(parts) >= 2:
            return parts[1]
        return "defenses"
    
    return path.name


def scan_group(group: str, base_path: Path) -> List[Dict[str, Any]]:
    """Scan a group folder for all runs and extract data."""
    runs = []
    
    if not base_path.exists():
        return runs
    
    # Find all directories that contain metrics.txt
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        metrics_file = root_path / "metrics.txt"
        
        if metrics_file.exists():
            metrics = parse_metrics_file(metrics_file)
            png_path = find_png_in_dir(root_path)
            client = extract_client_from_path(root_path)
            setting = derive_setting_name(root_path, group)
            
            run_data = {
                "group": group,
                "setting": setting,
                "client": client,
                "run_dir": str(root_path),
                "png_path": str(png_path) if png_path else None,
            }
            run_data.update(metrics)
            runs.append(run_data)
    
    return runs


def build_dataframe() -> pd.DataFrame:
    """Scan all input folders and build a unified DataFrame."""
    all_runs = []
    
    for group, base_path in INPUT_FOLDERS.items():
        runs = scan_group(group, base_path)
        all_runs.extend(runs)
        print(f"  Scanned {group}: found {len(runs)} runs")
    
    if not all_runs:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_runs)
    
    # Ensure known metric columns exist
    for metric in KNOWN_METRICS:
        if metric not in df.columns:
            df[metric] = float('nan')
    
    # Reorder columns
    base_cols = ["group", "setting", "client", "run_dir", "png_path"]
    metric_cols = KNOWN_METRICS + [c for c in df.columns if c not in base_cols + KNOWN_METRICS]
    df = df[base_cols + [c for c in metric_cols if c in df.columns]]
    
    # Sort deterministically
    df = df.sort_values(["group", "setting", "client", "run_dir"]).reset_index(drop=True)
    
    return df


# ============================================================================
# Leaderboard & Statistics
# ============================================================================

def compute_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Sort runs by quality: lowest LPIPS, then highest SSIM, then highest PSNR."""
    df_sorted = df.copy()
    df_sorted = df_sorted.sort_values(
        by=["LPIPS", "SSIM", "PSNR"],
        ascending=[True, False, False],
        na_position='last'
    ).reset_index(drop=True)
    return df_sorted


def compute_per_setting_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean±std for each setting within each group."""
    stats = []
    
    for (group, setting), grp in df.groupby(["group", "setting"]):
        row = {"group": group, "setting": setting, "n_runs": len(grp)}
        for metric in KNOWN_METRICS:
            if metric in grp.columns:
                vals = grp[metric].dropna()
                if len(vals) > 0:
                    row[f"{metric}_mean"] = vals.mean()
                    row[f"{metric}_std"] = vals.std() if len(vals) > 1 else 0.0
                else:
                    row[f"{metric}_mean"] = float('nan')
                    row[f"{metric}_std"] = float('nan')
        stats.append(row)
    
    return pd.DataFrame(stats)


def compute_defense_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deltas between baseline and each defense."""
    defenses_df = df[df["group"] == "defenses"].copy()
    
    if defenses_df.empty:
        return pd.DataFrame()
    
    # Find baseline
    baseline = defenses_df[defenses_df["setting"] == "baseline"]
    if baseline.empty:
        # Try to use first run as baseline
        baseline = defenses_df.iloc[:1]
    
    baseline_metrics = {
        "PSNR": baseline["PSNR"].mean(),
        "SSIM": baseline["SSIM"].mean(),
        "LPIPS": baseline["LPIPS"].mean(),
        "LabelMatch": baseline["LabelMatch"].mean() if "LabelMatch" in baseline else float('nan'),
    }
    
    deltas = []
    for setting, grp in defenses_df.groupby("setting"):
        row = {
            "setting": setting,
            "n_runs": len(grp),
            "PSNR": grp["PSNR"].mean(),
            "SSIM": grp["SSIM"].mean(),
            "LPIPS": grp["LPIPS"].mean(),
            "LabelMatch": grp["LabelMatch"].mean() if "LabelMatch" in grp else float('nan'),
            "ΔPSNR": grp["PSNR"].mean() - baseline_metrics["PSNR"],
            "ΔSSIM": grp["SSIM"].mean() - baseline_metrics["SSIM"],
            "ΔLPIPS": grp["LPIPS"].mean() - baseline_metrics["LPIPS"],
        }
        deltas.append(row)
    
    return pd.DataFrame(deltas).sort_values("LPIPS", ascending=False)


# ============================================================================
# Markdown Report Generation
# ============================================================================

def generate_markdown_report(
    df: pd.DataFrame,
    leaderboard: pd.DataFrame,
    per_setting_stats: pd.DataFrame,
    defense_deltas: pd.DataFrame
) -> str:
    """Generate a comprehensive markdown report."""
    lines = []
    lines.append("# Federated Learning Gradient Attack - Experiment Report")
    lines.append("")
    lines.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Runs Parsed:** {len(df)}")
    lines.append("")
    
    # Summary counts
    lines.append("## Summary")
    lines.append("")
    for group in df["group"].unique():
        count = len(df[df["group"] == group])
        lines.append(f"- **{group}**: {count} runs")
    lines.append("")
    
    # Top 10 Leaderboard
    lines.append("## Top 10 Overall (by LPIPS ↓, SSIM ↑, PSNR ↑)")
    lines.append("")
    top10 = leaderboard.head(10)[["group", "setting", "client", "LPIPS", "SSIM", "PSNR", "MSE"]].copy()
    top10 = top10.round(4)
    lines.append(top10.to_markdown(index=False))
    lines.append("")
    
    # Per-Group Statistics
    lines.append("## Per-Setting Aggregates")
    lines.append("")
    
    for group in per_setting_stats["group"].unique():
        lines.append(f"### {group.title()}")
        lines.append("")
        grp_stats = per_setting_stats[per_setting_stats["group"] == group].copy()
        
        # Format mean±std
        display_cols = ["setting", "n_runs"]
        for metric in ["PSNR", "SSIM", "LPIPS"]:
            if f"{metric}_mean" in grp_stats.columns:
                grp_stats[metric] = grp_stats.apply(
                    lambda r: f"{r[f'{metric}_mean']:.3f}±{r[f'{metric}_std']:.3f}" 
                    if pd.notna(r[f'{metric}_mean']) else "N/A",
                    axis=1
                )
                display_cols.append(metric)
        
        lines.append(grp_stats[display_cols].to_markdown(index=False))
        lines.append("")
    
    # Multi-client details
    mc_df = df[df["group"] == "multi_client"]
    if not mc_df.empty:
        lines.append("## Multi-Client Details")
        lines.append("")
        mc_display = mc_df[["setting", "client", "PSNR", "SSIM", "LPIPS", "LabelMatch"]].copy()
        mc_display = mc_display.round(4)
        lines.append(mc_display.to_markdown(index=False))
        lines.append("")
        
        # Aggregate stats
        lines.append("**Aggregate Statistics:**")
        lines.append("")
        for metric in ["PSNR", "SSIM", "LPIPS"]:
            if metric in mc_df.columns:
                vals = mc_df[metric].dropna()
                if len(vals) > 0:
                    lines.append(f"- {metric}: {vals.mean():.4f} ± {vals.std():.4f} (min: {vals.min():.4f}, max: {vals.max():.4f})")
        lines.append("")
    
    # Defense comparison
    if not defense_deltas.empty:
        lines.append("## Defense Comparison")
        lines.append("")
        lines.append("Δ values are relative to baseline (positive ΔLPIPS = worse reconstruction = better defense).")
        lines.append("")
        defense_display = defense_deltas[["setting", "n_runs", "PSNR", "SSIM", "LPIPS", "ΔPSNR", "ΔSSIM", "ΔLPIPS", "LabelMatch"]].copy()
        defense_display = defense_display.round(4)
        lines.append(defense_display.to_markdown(index=False))
        lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("*Report generated by `python scripts/make_report.py`*")
    lines.append("")
    
    return "\n".join(lines)


# ============================================================================
# Figure Generation
# ============================================================================

def plot_ablation_bars(df: pd.DataFrame, stats: pd.DataFrame, output_dir: Path):
    """Create bar charts for ablation study."""
    ablation_stats = stats[stats["group"] == "ablation"].copy()
    
    if ablation_stats.empty:
        print("  No ablation data for bar chart")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Ablation Study: Attack Performance by Setting", fontsize=14, fontweight='bold')
    
    metrics = ["PSNR", "SSIM", "LPIPS"]
    titles = ["PSNR (↑ better)", "SSIM (↑ better)", "LPIPS (↓ better)"]
    
    settings = ablation_stats["setting"].tolist()
    x = np.arange(len(settings))
    width = 0.6
    
    for ax, metric, title in zip(axes, metrics, titles):
        means = ablation_stats[f"{metric}_mean"].fillna(0).values
        stds = ablation_stats[f"{metric}_std"].fillna(0).values
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=3, 
                     color=COLORS[:len(settings)], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel("Setting")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(settings, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_bars.png")
    plt.close()
    print(f"  Saved: {output_dir / 'ablation_bars.png'}")


def plot_multiclient_boxplots(df: pd.DataFrame, output_dir: Path):
    """Create boxplots for multi-client results."""
    mc_df = df[df["group"] == "multi_client"].copy()
    
    if mc_df.empty:
        print("  No multi-client data for boxplots")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Multi-Client Attack Performance Distribution", fontsize=14, fontweight='bold')
    
    metrics = ["PSNR", "SSIM", "LPIPS"]
    titles = ["PSNR (↑ better)", "SSIM (↑ better)", "LPIPS (↓ better)"]
    
    for ax, metric, title in zip(axes, metrics, titles):
        if metric not in mc_df.columns:
            continue
        
        data = mc_df[metric].dropna().values
        if len(data) == 0:
            continue
        
        bp = ax.boxplot(data, patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS[0])
        bp['boxes'][0].set_alpha(0.7)
        
        # Add individual points
        x = np.ones(len(data)) + np.random.uniform(-0.1, 0.1, len(data))
        ax.scatter(x, data, alpha=0.6, color=COLORS[1], edgecolor='black', linewidth=0.5, s=50)
        
        # Mark showcase if available
        showcase = df[(df["group"] == "showcase") & df[metric].notna()]
        if not showcase.empty:
            showcase_val = showcase[metric].iloc[0]
            ax.axhline(y=showcase_val, color='red', linestyle='--', linewidth=2, label='Showcase')
            ax.legend(loc='upper right')
        
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticklabels(['All Clients'])
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "multiclient_boxplots.png")
    plt.close()
    print(f"  Saved: {output_dir / 'multiclient_boxplots.png'}")


def plot_defense_comparison(df: pd.DataFrame, output_dir: Path):
    """Create defense comparison plots."""
    defense_df = df[df["group"] == "defenses"].copy()
    
    if defense_df.empty:
        print("  No defense data for comparison plots")
        return
    
    # (a) Grouped bar chart: LPIPS + LabelMatch
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Defense Effectiveness Comparison", fontsize=14, fontweight='bold')
    
    settings = defense_df.groupby("setting").first().reset_index()["setting"].tolist()
    
    # Sort to put baseline first
    if "baseline" in settings:
        settings.remove("baseline")
        settings = ["baseline"] + sorted(settings)
    
    x = np.arange(len(settings))
    
    # LPIPS bar chart
    ax = axes[0]
    lpips_vals = [defense_df[defense_df["setting"] == s]["LPIPS"].mean() for s in settings]
    colors_list = ['#2ecc71' if s == 'baseline' else COLORS[i % len(COLORS)] for i, s in enumerate(settings)]
    
    bars = ax.bar(x, lpips_vals, color=colors_list, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Defense")
    ax.set_ylabel("LPIPS")
    ax.set_title("LPIPS by Defense (↑ = better protection)")
    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # LabelMatch bar chart
    ax = axes[1]
    if "LabelMatch" in defense_df.columns:
        lm_vals = [defense_df[defense_df["setting"] == s]["LabelMatch"].mean() for s in settings]
        bars = ax.bar(x, lm_vals, color=colors_list, edgecolor='black', linewidth=0.5)
        ax.set_xlabel("Defense")
        ax.set_ylabel("LabelMatch Rate")
        ax.set_title("Label Match Rate (↓ = better protection)")
        ax.set_xticks(x)
        ax.set_xticklabels(settings, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "defense_bars.png")
    plt.close()
    print(f"  Saved: {output_dir / 'defense_bars.png'}")
    
    # (b) Scatter: PSNR vs LPIPS
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, setting in enumerate(settings):
        subset = defense_df[defense_df["setting"] == setting]
        color = '#2ecc71' if setting == 'baseline' else COLORS[i % len(COLORS)]
        ax.scatter(subset["LPIPS"], subset["PSNR"], 
                  c=[color], s=100, alpha=0.8, edgecolor='black', linewidth=0.5,
                  label=setting)
    
    ax.set_xlabel("LPIPS (↑ = worse reconstruction)")
    ax.set_ylabel("PSNR (↑ = better reconstruction)")
    ax.set_title("Defense Effect: PSNR vs LPIPS Trade-off")
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "defense_scatter.png")
    plt.close()
    print(f"  Saved: {output_dir / 'defense_scatter.png'}")


# ============================================================================
# Montage Generation
# ============================================================================

def get_font(size: int = FONT_SIZE):
    """Get a font for PIL, with fallback."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except:
            return ImageFont.load_default()


def load_and_resize(png_path: str, size: Tuple[int, int] = MONTAGE_THUMB_SIZE) -> Optional[Image.Image]:
    """Load a PNG and resize it."""
    if not png_path or not Path(png_path).exists():
        return None
    try:
        img = Image.open(png_path).convert("RGB")
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"  Warning: Could not load {png_path}: {e}")
        return None


def create_labeled_tile(img: Image.Image, label: str, tile_size: Tuple[int, int]) -> Image.Image:
    """Create a tile with image and label."""
    tile = Image.new("RGB", tile_size, color=(255, 255, 255))
    
    # Paste image centered
    x_offset = (tile_size[0] - img.width) // 2
    y_offset = 5
    tile.paste(img, (x_offset, y_offset))
    
    # Add label
    draw = ImageDraw.Draw(tile)
    font = get_font(12)
    
    # Wrap text if needed
    label_y = y_offset + img.height + 5
    lines = label.split('\n')
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (tile_size[0] - text_width) // 2
        draw.text((text_x, label_y), line, fill=(0, 0, 0), font=font)
        label_y += 15
    
    return tile


def create_montage_grid(
    items: List[Tuple[Image.Image, str]],
    cols: int,
    tile_size: Tuple[int, int] = (320, 360),
    title: str = ""
) -> Image.Image:
    """Create a grid montage from (image, label) pairs."""
    if not items:
        # Return placeholder
        placeholder = Image.new("RGB", (400, 200), color=(240, 240, 240))
        draw = ImageDraw.Draw(placeholder)
        draw.text((100, 90), "No images available", fill=(100, 100, 100), font=get_font(14))
        return placeholder
    
    rows = (len(items) + cols - 1) // cols
    title_height = 40 if title else 0
    
    montage = Image.new("RGB", (cols * tile_size[0], rows * tile_size[1] + title_height), color=(255, 255, 255))
    
    # Add title
    if title:
        draw = ImageDraw.Draw(montage)
        font = get_font(18)
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text(((cols * tile_size[0] - text_width) // 2, 10), title, fill=(0, 0, 0), font=font)
    
    for idx, (img, label) in enumerate(items):
        row, col = divmod(idx, cols)
        tile = create_labeled_tile(img, label, tile_size)
        montage.paste(tile, (col * tile_size[0], row * tile_size[1] + title_height))
    
    return montage


def generate_montage_best_overall(df: pd.DataFrame, output_dir: Path, k: int = MONTAGE_TOP_K):
    """Create montage of top-K best reconstructions."""
    leaderboard = compute_leaderboard(df)
    top_k = leaderboard.head(k)
    
    items = []
    for _, row in top_k.iterrows():
        img = load_and_resize(row["png_path"])
        if img:
            label = f"{row['setting']}"
            if row['client']:
                label += f" ({row['client']})"
            label += f"\nLPIPS={row['LPIPS']:.3f}"
            items.append((img, label))
    
    montage = create_montage_grid(items, cols=4, title=f"Top {k} Best Reconstructions (by LPIPS)")
    montage.save(output_dir / "montage_best_overall.png")
    print(f"  Saved: {output_dir / 'montage_best_overall.png'}")


def generate_montage_ablation_best(df: pd.DataFrame, output_dir: Path):
    """Create montage of best run per ablation setting."""
    ablation_df = df[df["group"] == "ablation"].copy()
    
    if ablation_df.empty:
        print("  No ablation data for montage")
        return
    
    items = []
    for setting in ablation_df["setting"].unique():
        subset = ablation_df[ablation_df["setting"] == setting]
        # Sort by LPIPS
        subset = subset.sort_values("LPIPS", ascending=True)
        best = subset.iloc[0]
        
        img = load_and_resize(best["png_path"])
        if img:
            label = f"{setting}"
            if best['client']:
                label += f" ({best['client']})"
            label += f"\nLPIPS={best['LPIPS']:.3f}"
            items.append((img, label))
    
    cols = min(len(items), 4)
    montage = create_montage_grid(items, cols=cols, title="Ablation: Best Run per Setting")
    montage.save(output_dir / "montage_ablation_best_per_setting.png")
    print(f"  Saved: {output_dir / 'montage_ablation_best_per_setting.png'}")


def generate_montage_multiclient(df: pd.DataFrame, output_dir: Path):
    """Create montage for multi-client results (c0..c9)."""
    mc_df = df[df["group"] == "multi_client"].copy()
    
    if mc_df.empty:
        print("  No multi-client data for montage")
        return
    
    # Sort by client number
    mc_df["client_num"] = mc_df["setting"].str.extract(r'c(\d+)').astype(float)
    mc_df = mc_df.sort_values("client_num")
    
    items = []
    for _, row in mc_df.iterrows():
        img = load_and_resize(row["png_path"])
        if img:
            label = f"{row['setting']}\nLPIPS={row['LPIPS']:.3f}"
            items.append((img, label))
    
    cols = min(len(items), 5)
    montage = create_montage_grid(items, cols=cols, title="Multi-Client Attack Results")
    montage.save(output_dir / "montage_multiclient.png")
    print(f"  Saved: {output_dir / 'montage_multiclient.png'}")


def generate_montage_defenses(df: pd.DataFrame, output_dir: Path):
    """Create side-by-side defense comparison montage."""
    defense_df = df[df["group"] == "defenses"].copy()
    
    if defense_df.empty:
        print("  No defense data for montage")
        return
    
    # Get all settings, put baseline first
    settings = defense_df["setting"].unique().tolist()
    if "baseline" in settings:
        settings.remove("baseline")
        settings = ["baseline"] + sorted(settings)
    
    items = []
    for setting in settings:
        subset = defense_df[defense_df["setting"] == setting]
        if subset.empty:
            continue
        
        # Pick first/best run
        best = subset.sort_values("LPIPS", ascending=True).iloc[0]
        img = load_and_resize(best["png_path"])
        if img:
            label = f"{setting}\nLPIPS={best['LPIPS']:.3f}"
            if pd.notna(best.get("SSIM")):
                label += f"\nSSIM={best['SSIM']:.3f}"
            items.append((img, label))
    
    cols = min(len(items), 4)
    montage = create_montage_grid(items, cols=cols, tile_size=(320, 380), 
                                  title="Defense Comparison: Baseline vs Protections")
    montage.save(output_dir / "montage_defenses_side_by_side.png")
    print(f"  Saved: {output_dir / 'montage_defenses_side_by_side.png'}")


# ============================================================================
# 4K Poster Generation (3840x2160 UHD Showcase)
# ============================================================================

POSTER_WIDTH = 3840
POSTER_HEIGHT = 2160
POSTER_BG_COLOR = (255, 255, 255)
POSTER_ACCENT_COLOR = (41, 128, 185)  # Blue accent
POSTER_DARK_COLOR = (44, 62, 80)  # Dark text
POSTER_LIGHT_GRAY = (245, 245, 245)
POSTER_SUCCESS_COLOR = (39, 174, 96)  # Green badge
POSTER_WARNING_COLOR = (243, 156, 18)  # Orange


def get_poster_font(size: int = 24, bold: bool = False):
    """Get a font for poster, with fallback."""
    # Try system fonts with bold variant
    font_candidates = [
        ("/System/Library/Fonts/Helvetica.ttc", 0),  # Regular
        ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
    ]
    if bold:
        font_candidates = [
            ("/System/Library/Fonts/Helvetica.ttc", 1),  # Bold index
            ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
        ] + font_candidates
    
    for path, idx in font_candidates:
        try:
            if path.endswith('.ttc'):
                return ImageFont.truetype(path, size, index=idx)
            else:
                return ImageFont.truetype(path, size)
        except:
            continue
    return ImageFont.load_default()


def find_anchor_client(df: pd.DataFrame) -> str:
    """
    Find the best anchor client that exists in the most defense folders.
    Returns the client identifier (e.g., 'c0').
    """
    defense_df = df[df["group"] == "defenses"].copy()
    if defense_df.empty:
        # Fallback to multi-client or ablation
        for group in ["multi_client", "ablation"]:
            grp_df = df[df["group"] == group]
            if not grp_df.empty:
                clients = grp_df["client"].dropna().unique()
                if len(clients) > 0:
                    return clients[0]
        return None
    
    # Count how many defense settings each client appears in
    # Priority settings: baseline, dp_eps*, he, dp_he
    priority_settings = ["baseline", "dp_eps01", "dp_eps1", "dp_eps8", "he", "dp_he"]
    
    client_counts = defaultdict(int)
    for setting in priority_settings:
        setting_df = defense_df[defense_df["setting"] == setting]
        for client in setting_df["client"].dropna().unique():
            client_counts[client] += 1
    
    # If no clients found in defense data, check if there's client info
    if not client_counts:
        # Defense data might not have client column, use "N/A" or check for single runs
        if len(defense_df) > 0:
            return None  # Single client scenario
    
    if client_counts:
        # Return client with highest count
        return max(client_counts, key=client_counts.get)
    
    return None


def load_and_resize_poster(png_path: str, max_size: Tuple[int, int], 
                            maintain_aspect: bool = True) -> Optional[Image.Image]:
    """Load a PNG and resize for poster, maintaining aspect ratio."""
    if not png_path or not Path(png_path).exists():
        return None
    try:
        img = Image.open(png_path).convert("RGB")
        if maintain_aspect:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        else:
            img = img.resize(max_size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"  Warning: Could not load {png_path}: {e}")
        return None


def create_missing_placeholder(size: Tuple[int, int], text: str = "Missing") -> Image.Image:
    """Create a tasteful placeholder for missing images."""
    placeholder = Image.new("RGB", size, color=POSTER_LIGHT_GRAY)
    draw = ImageDraw.Draw(placeholder)
    
    # Draw diagonal lines
    for i in range(-size[1], size[0], 20):
        draw.line([(i, 0), (i + size[1], size[1])], fill=(220, 220, 220), width=1)
    
    # Draw text in center
    font = get_poster_font(24)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size[0] - text_w) // 2
    y = (size[1] - text_h) // 2
    
    # Text background
    padding = 10
    draw.rectangle([x - padding, y - padding, x + text_w + padding, y + text_h + padding], 
                   fill=(255, 255, 255, 200))
    draw.text((x, y), text, fill=(150, 150, 150), font=font)
    
    return placeholder


def create_metrics_card(metrics: Dict[str, float], size: Tuple[int, int], 
                        title: str = "", highlight_best: bool = False) -> Image.Image:
    """Create a metrics card with PSNR/SSIM/LPIPS/LabelMatch."""
    card = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(card)
    
    # Border
    draw.rectangle([0, 0, size[0]-1, size[1]-1], outline=POSTER_ACCENT_COLOR, width=2)
    
    y_pos = 8
    
    if title:
        font_title = get_poster_font(18, bold=True)
        draw.text((10, y_pos), title, fill=POSTER_DARK_COLOR, font=font_title)
        y_pos += 28
    
    font_metric = get_poster_font(16)
    font_value = get_poster_font(18, bold=True)
    
    metric_order = ["PSNR", "SSIM", "LPIPS", "LabelMatch"]
    
    for metric in metric_order:
        if metric in metrics and not pd.isna(metrics.get(metric)):
            val = metrics[metric]
            label = f"{metric}:"
            value_str = f"{val:.4f}" if metric != "LabelMatch" else f"{val:.2f}"
            
            draw.text((10, y_pos), label, fill=(100, 100, 100), font=font_metric)
            draw.text((90, y_pos), value_str, fill=POSTER_DARK_COLOR, font=font_value)
            y_pos += 24
    
    # Best defense badge
    if highlight_best:
        badge_text = "★ BEST"
        font_badge = get_poster_font(14, bold=True)
        bbox = draw.textbbox((0, 0), badge_text, font=font_badge)
        badge_w = bbox[2] - bbox[0] + 16
        badge_h = bbox[3] - bbox[1] + 8
        badge_x = size[0] - badge_w - 8
        badge_y = 8
        draw.rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h], 
                       fill=POSTER_SUCCESS_COLOR)
        draw.text((badge_x + 8, badge_y + 4), badge_text, fill=(255, 255, 255), font=font_badge)
    
    return card


def render_dp_trend_plot(df: pd.DataFrame, size: Tuple[int, int]) -> Image.Image:
    """Render a small DP trend plot: epsilon vs LPIPS."""
    defense_df = df[df["group"] == "defenses"].copy()
    
    # Extract epsilon values from setting names
    dp_data = []
    for _, row in defense_df.iterrows():
        setting = row["setting"]
        if "dp_eps" in setting and "he" not in setting:
            # Extract epsilon: dp_eps01 -> 0.1, dp_eps1 -> 1, dp_eps8 -> 8
            match = re.search(r'dp_eps(\d+)', setting)
            if match:
                eps_str = match.group(1)
                if eps_str == "01":
                    eps = 0.1
                else:
                    eps = float(eps_str)
                dp_data.append({"epsilon": eps, "LPIPS": row["LPIPS"], "PSNR": row["PSNR"]})
    
    if len(dp_data) < 2:
        # Not enough data for trend
        return create_missing_placeholder(size, "DP Trend\n(insufficient data)")
    
    dp_df = pd.DataFrame(dp_data).groupby("epsilon").mean().reset_index()
    dp_df = dp_df.sort_values("epsilon")
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    
    # Convert color to 0-1 range for matplotlib
    accent_color = tuple(c/255 for c in POSTER_ACCENT_COLOR)
    
    ax.plot(dp_df["epsilon"], dp_df["LPIPS"], 'o-', color=accent_color, 
            linewidth=2, markersize=8, label="LPIPS")
    
    ax.set_xlabel("ε (privacy budget)", fontsize=10)
    ax.set_ylabel("LPIPS (↑ = better defense)", fontsize=10)
    ax.set_title("DP Privacy-Utility Trade-off", fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    plot_img = Image.open(buf).convert("RGB")
    
    # Resize to exact size
    plot_img = plot_img.resize(size, Image.Resampling.LANCZOS)
    
    return plot_img


def render_comparison_sparkbars(baseline_lpips: float, dp_lpips: float, 
                                 he_lpips: float, dphe_lpips: float,
                                 size: Tuple[int, int]) -> Image.Image:
    """Create mini comparison bars for LPIPS across defense types."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    font = get_poster_font(12)
    font_small = get_poster_font(10)
    
    labels = ["Base", "DP", "HE", "DP+HE"]
    values = [baseline_lpips, dp_lpips, he_lpips, dphe_lpips]
    colors = [(231, 76, 60), (243, 156, 18), (52, 152, 219), (39, 174, 96)]
    
    # Filter valid values
    valid_data = [(l, v, c) for l, v, c in zip(labels, values, colors) if not pd.isna(v)]
    
    if not valid_data:
        return create_missing_placeholder(size, "No data")
    
    max_val = max(v for _, v, _ in valid_data) * 1.1
    bar_height = 18
    bar_start_x = 50
    bar_max_width = size[0] - bar_start_x - 60
    y = 10
    
    draw.text((5, y - 5), "LPIPS Comparison", fill=POSTER_DARK_COLOR, font=font)
    y += 20
    
    for label, val, color in valid_data:
        bar_width = int((val / max_val) * bar_max_width) if max_val > 0 else 0
        
        draw.text((5, y + 2), label, fill=POSTER_DARK_COLOR, font=font_small)
        draw.rectangle([bar_start_x, y, bar_start_x + bar_width, y + bar_height], fill=color)
        draw.text((bar_start_x + bar_width + 5, y + 2), f"{val:.3f}", 
                  fill=POSTER_DARK_COLOR, font=font_small)
        y += bar_height + 8
    
    return img


def create_ablation_summary_table(df: pd.DataFrame, stats: pd.DataFrame, 
                                   size: Tuple[int, int]) -> Image.Image:
    """Create a compact ablation summary table image."""
    ablation_stats = stats[stats["group"] == "ablation"].copy()
    
    if ablation_stats.empty:
        return create_missing_placeholder(size, "No ablation data")
    
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    font_header = get_poster_font(14, bold=True)
    font_cell = get_poster_font(12)
    
    # Table headers
    headers = ["Setting", "PSNR", "SSIM", "LPIPS"]
    col_widths = [size[0] * 0.35, size[0] * 0.2, size[0] * 0.2, size[0] * 0.2]
    
    y = 5
    x = 5
    
    # Header row
    draw.rectangle([0, 0, size[0], 25], fill=POSTER_ACCENT_COLOR)
    for header, w in zip(headers, col_widths):
        draw.text((x, y), header, fill=(255, 255, 255), font=font_header)
        x += int(w)
    
    y = 28
    
    # Find best setting (lowest LPIPS)
    best_setting = ablation_stats.loc[ablation_stats["LPIPS_mean"].idxmin(), "setting"] \
                   if "LPIPS_mean" in ablation_stats.columns else None
    
    # Data rows
    for _, row in ablation_stats.iterrows():
        x = 5
        setting = row["setting"]
        is_best = setting == best_setting
        
        if is_best:
            draw.rectangle([0, y - 2, size[0], y + 18], fill=(232, 245, 233))
        
        # Setting name (truncate if needed)
        setting_display = setting[:12] + "..." if len(setting) > 15 else setting
        draw.text((x, y), setting_display, fill=POSTER_DARK_COLOR, font=font_cell)
        x += int(col_widths[0])
        
        for metric, w in zip(["PSNR", "SSIM", "LPIPS"], col_widths[1:]):
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            if mean_key in row and not pd.isna(row[mean_key]):
                val_str = f"{row[mean_key]:.2f}"
                if std_key in row and not pd.isna(row[std_key]):
                    val_str += f"±{row[std_key]:.2f}"
                draw.text((x, y), val_str, fill=POSTER_DARK_COLOR, font=font_cell)
            x += int(w)
        
        y += 20
        if y > size[1] - 20:
            break
    
    return img


def derive_ablation_conclusions(df: pd.DataFrame, stats: pd.DataFrame) -> List[str]:
    """Derive 2-3 factual conclusions from ablation data."""
    conclusions = []
    ablation_stats = stats[stats["group"] == "ablation"].copy()
    
    if ablation_stats.empty:
        return ["No ablation data available for analysis."]
    
    # Find best and worst settings by LPIPS
    if "LPIPS_mean" in ablation_stats.columns:
        best_idx = ablation_stats["LPIPS_mean"].idxmin()
        worst_idx = ablation_stats["LPIPS_mean"].idxmax()
        
        best = ablation_stats.loc[best_idx]
        worst = ablation_stats.loc[worst_idx]
        
        conclusions.append(
            f"Best setting: '{best['setting']}' (LPIPS: {best['LPIPS_mean']:.3f})"
        )
        
        if best["setting"] != worst["setting"]:
            lpips_diff = worst["LPIPS_mean"] - best["LPIPS_mean"]
            conclusions.append(
                f"'{worst['setting']}' shows {lpips_diff:.3f} higher LPIPS (worse attack)"
            )
    
    # Compare SSIM trends
    if "SSIM_mean" in ablation_stats.columns:
        ssim_range = ablation_stats["SSIM_mean"].max() - ablation_stats["SSIM_mean"].min()
        if ssim_range > 0.05:
            conclusions.append(f"SSIM varies by {ssim_range:.3f} across ablation settings")
        else:
            conclusions.append(f"SSIM relatively stable (range: {ssim_range:.3f})")
    
    return conclusions[:3]


def generate_4k_poster(df: pd.DataFrame, stats: pd.DataFrame, output_path: Path):
    """
    Generate a 4K (3840x2160) showcase poster.
    
    Layout:
    - Top row: Title bar
    - Middle: 4 columns (Baseline | DP | HE | DP+HE)
    - Bottom strip: Ablation summary
    """
    print("\n[POSTER] Generating 4K showcase poster...")
    
    # Create canvas
    poster = Image.new("RGB", (POSTER_WIDTH, POSTER_HEIGHT), color=POSTER_BG_COLOR)
    draw = ImageDraw.Draw(poster)
    
    # === FIND ANCHOR CLIENT ===
    anchor_client = find_anchor_client(df)
    print(f"  Anchor client: {anchor_client if anchor_client else 'N/A (single-client mode)'}")
    
    # === LAYOUT CONSTANTS ===
    TITLE_HEIGHT = 100
    BOTTOM_STRIP_HEIGHT = 450
    CONTENT_HEIGHT = POSTER_HEIGHT - TITLE_HEIGHT - BOTTOM_STRIP_HEIGHT
    COLUMN_WIDTH = POSTER_WIDTH // 4
    MARGIN = 20
    
    # === TITLE ROW ===
    draw.rectangle([0, 0, POSTER_WIDTH, TITLE_HEIGHT], fill=POSTER_ACCENT_COLOR)
    
    title_font = get_poster_font(48, bold=True)
    subtitle_font = get_poster_font(24)
    
    title_text = "Federated Learning Privacy Attack Analysis"
    draw.text((MARGIN, 20), title_text, fill=(255, 255, 255), font=title_font)
    
    subtitle_text = f"Defense Comparison • Anchor Client: {anchor_client if anchor_client else 'Single Run'}"
    draw.text((MARGIN, 65), subtitle_text, fill=(220, 220, 220), font=subtitle_font)
    
    # Timestamp on right
    timestamp_text = pd.Timestamp.now().strftime('%Y-%m-%d')
    bbox = draw.textbbox((0, 0), timestamp_text, font=subtitle_font)
    draw.text((POSTER_WIDTH - bbox[2] - MARGIN, 65), timestamp_text, 
              fill=(220, 220, 220), font=subtitle_font)
    
    # === HELPER: Get defense data for anchor client ===
    def get_defense_run(setting: str):
        """Get the run data for a specific defense setting (optionally for anchor client)."""
        defense_df = df[df["group"] == "defenses"]
        setting_df = defense_df[defense_df["setting"] == setting]
        
        if setting_df.empty:
            return None
        
        if anchor_client:
            client_df = setting_df[setting_df["client"] == anchor_client]
            if not client_df.empty:
                return client_df.iloc[0]
        
        # Fallback to first run
        return setting_df.iloc[0]
    
    # === COLUMN 1: BASELINE ===
    col1_x = 0
    col_y = TITLE_HEIGHT
    
    # Column header
    header_font = get_poster_font(28, bold=True)
    draw.rectangle([col1_x, col_y, col1_x + COLUMN_WIDTH, col_y + 50], fill=POSTER_DARK_COLOR)
    draw.text((col1_x + MARGIN, col_y + 10), "BASELINE", fill=(255, 255, 255), font=header_font)
    
    baseline_run = get_defense_run("baseline")
    baseline_img_y = col_y + 60
    baseline_img_size = (COLUMN_WIDTH - 2*MARGIN, int(CONTENT_HEIGHT * 0.55))
    
    if baseline_run is not None and baseline_run["png_path"]:
        baseline_img = load_and_resize_poster(baseline_run["png_path"], baseline_img_size)
        if baseline_img:
            img_x = col1_x + (COLUMN_WIDTH - baseline_img.width) // 2
            poster.paste(baseline_img, (img_x, baseline_img_y))
    else:
        placeholder = create_missing_placeholder(baseline_img_size, "Baseline Image\nMissing")
        poster.paste(placeholder, (col1_x + MARGIN, baseline_img_y))
    
    # Metrics card
    metrics_y = baseline_img_y + baseline_img_size[1] + 15
    metrics_card_size = (COLUMN_WIDTH - 2*MARGIN, 140)
    
    if baseline_run is not None:
        baseline_metrics = {
            "PSNR": baseline_run.get("PSNR"),
            "SSIM": baseline_run.get("SSIM"),
            "LPIPS": baseline_run.get("LPIPS"),
            "LabelMatch": baseline_run.get("LabelMatch"),
        }
        metrics_card = create_metrics_card(baseline_metrics, metrics_card_size, "Attack Metrics")
        poster.paste(metrics_card, (col1_x + MARGIN, metrics_y))
        baseline_lpips = baseline_run.get("LPIPS", float('nan'))
    else:
        baseline_lpips = float('nan')
        placeholder = create_missing_placeholder(metrics_card_size, "No metrics")
        poster.paste(placeholder, (col1_x + MARGIN, metrics_y))
    
    # Comparison sparkbars
    sparkbar_y = metrics_y + metrics_card_size[1] + 15
    sparkbar_size = (COLUMN_WIDTH - 2*MARGIN, 150)
    
    # Get LPIPS for each defense type
    dp_run = get_defense_run("dp_eps1")  # Use eps1 as representative
    he_run = get_defense_run("he")
    dphe_run = get_defense_run("dp_he")
    
    dp_lpips = dp_run.get("LPIPS", float('nan')) if dp_run is not None else float('nan')
    he_lpips = he_run.get("LPIPS", float('nan')) if he_run is not None else float('nan')
    dphe_lpips = dphe_run.get("LPIPS", float('nan')) if dphe_run is not None else float('nan')
    
    sparkbars = render_comparison_sparkbars(baseline_lpips, dp_lpips, he_lpips, dphe_lpips, sparkbar_size)
    poster.paste(sparkbars, (col1_x + MARGIN, sparkbar_y))
    
    # === COLUMN 2: DP (3 epsilon values) ===
    col2_x = COLUMN_WIDTH
    
    # Column header
    draw.rectangle([col2_x, col_y, col2_x + COLUMN_WIDTH, col_y + 50], fill=(243, 156, 18))
    draw.text((col2_x + MARGIN, col_y + 10), "DIFFERENTIAL PRIVACY", fill=(255, 255, 255), font=header_font)
    
    dp_settings = ["dp_eps8", "dp_eps1", "dp_eps01"]
    dp_labels = ["ε=8 (weak)", "ε=1 (moderate)", "ε=0.1 (strong)"]
    
    dp_panel_height = int((CONTENT_HEIGHT - 200) / 3)
    dp_y = col_y + 60
    
    for setting, label in zip(dp_settings, dp_labels):
        run = get_defense_run(setting)
        
        # Image panel
        panel_img_size = (int(COLUMN_WIDTH * 0.55), dp_panel_height - 20)
        
        if run is not None and run["png_path"]:
            panel_img = load_and_resize_poster(run["png_path"], panel_img_size)
            if panel_img:
                poster.paste(panel_img, (col2_x + MARGIN, dp_y))
        else:
            placeholder = create_missing_placeholder(panel_img_size, f"{setting}\nMissing")
            poster.paste(placeholder, (col2_x + MARGIN, dp_y))
        
        # Mini metrics next to image
        mini_metrics_x = col2_x + MARGIN + panel_img_size[0] + 10
        mini_metrics_w = COLUMN_WIDTH - panel_img_size[0] - 3*MARGIN
        
        label_font = get_poster_font(14, bold=True)
        value_font = get_poster_font(13)
        
        draw.text((mini_metrics_x, dp_y), label, fill=POSTER_DARK_COLOR, font=label_font)
        
        if run is not None:
            metrics_text = [
                f"LPIPS: {run.get('LPIPS', 0):.3f}",
                f"SSIM: {run.get('SSIM', 0):.3f}",
                f"PSNR: {run.get('PSNR', 0):.1f}",
            ]
            for i, txt in enumerate(metrics_text):
                draw.text((mini_metrics_x, dp_y + 20 + i*18), txt, 
                          fill=(80, 80, 80), font=value_font)
        
        dp_y += dp_panel_height
    
    # DP trend plot
    trend_y = col_y + 60 + 3 * dp_panel_height
    trend_size = (COLUMN_WIDTH - 2*MARGIN, 180)
    trend_plot = render_dp_trend_plot(df, trend_size)
    poster.paste(trend_plot, (col2_x + MARGIN, trend_y))
    
    # === COLUMN 3: HE ===
    col3_x = 2 * COLUMN_WIDTH
    
    # Column header
    draw.rectangle([col3_x, col_y, col3_x + COLUMN_WIDTH, col_y + 50], fill=(52, 152, 219))
    draw.text((col3_x + MARGIN, col_y + 10), "HOMOMORPHIC ENCRYPTION", 
              fill=(255, 255, 255), font=get_poster_font(24, bold=True))
    
    he_run = get_defense_run("he")
    he_img_y = col_y + 60
    he_img_size = (COLUMN_WIDTH - 2*MARGIN, int(CONTENT_HEIGHT * 0.5))
    
    if he_run is not None and he_run["png_path"]:
        he_img = load_and_resize_poster(he_run["png_path"], he_img_size)
        if he_img:
            img_x = col3_x + (COLUMN_WIDTH - he_img.width) // 2
            poster.paste(he_img, (img_x, he_img_y))
    else:
        placeholder = create_missing_placeholder(he_img_size, "HE Image\nMissing")
        poster.paste(placeholder, (col3_x + MARGIN, he_img_y))
    
    # HE metrics card
    he_metrics_y = he_img_y + he_img_size[1] + 15
    he_metrics_size = (COLUMN_WIDTH - 2*MARGIN, 140)
    
    if he_run is not None:
        he_metrics = {
            "PSNR": he_run.get("PSNR"),
            "SSIM": he_run.get("SSIM"),
            "LPIPS": he_run.get("LPIPS"),
            "LabelMatch": he_run.get("LabelMatch"),
        }
        he_card = create_metrics_card(he_metrics, he_metrics_size, "HE Defense Metrics")
        poster.paste(he_card, (col3_x + MARGIN, he_metrics_y))
    else:
        placeholder = create_missing_placeholder(he_metrics_size, "No metrics")
        poster.paste(placeholder, (col3_x + MARGIN, he_metrics_y))
    
    # Takeaway box
    takeaway_y = he_metrics_y + he_metrics_size[1] + 15
    takeaway_h = 120
    
    draw.rectangle([col3_x + MARGIN, takeaway_y, col3_x + COLUMN_WIDTH - MARGIN, takeaway_y + takeaway_h],
                   fill=POSTER_LIGHT_GRAY, outline=POSTER_ACCENT_COLOR, width=2)
    
    takeaway_font = get_poster_font(14)
    takeaway_title_font = get_poster_font(16, bold=True)
    
    draw.text((col3_x + MARGIN + 10, takeaway_y + 10), "Key Insight:", 
              fill=POSTER_DARK_COLOR, font=takeaway_title_font)
    
    if he_run is not None:
        he_lpips = he_run.get("LPIPS", 0)
        if baseline_run is not None:
            base_lpips = baseline_run.get("LPIPS", 0)
            if he_lpips > base_lpips:
                insight = f"HE increases LPIPS by {he_lpips - base_lpips:.3f},\ndegrading attack quality."
            else:
                insight = "HE maintains comparable\nreconstruction quality."
        else:
            insight = f"LPIPS: {he_lpips:.3f}\n(no baseline for comparison)"
    else:
        insight = "HE data not available."
    
    draw.text((col3_x + MARGIN + 10, takeaway_y + 35), insight, 
              fill=(80, 80, 80), font=takeaway_font)
    
    # === COLUMN 4: DP+HE ===
    col4_x = 3 * COLUMN_WIDTH
    
    # Column header
    draw.rectangle([col4_x, col_y, col4_x + COLUMN_WIDTH, col_y + 50], fill=POSTER_SUCCESS_COLOR)
    draw.text((col4_x + MARGIN, col_y + 10), "DP + HE COMBINED", 
              fill=(255, 255, 255), font=header_font)
    
    dphe_run = get_defense_run("dp_he")
    dphe_img_y = col_y + 60
    dphe_img_size = (COLUMN_WIDTH - 2*MARGIN, int(CONTENT_HEIGHT * 0.5))
    
    if dphe_run is not None and dphe_run["png_path"]:
        dphe_img = load_and_resize_poster(dphe_run["png_path"], dphe_img_size)
        if dphe_img:
            img_x = col4_x + (COLUMN_WIDTH - dphe_img.width) // 2
            poster.paste(dphe_img, (img_x, dphe_img_y))
    else:
        placeholder = create_missing_placeholder(dphe_img_size, "DP+HE Image\nMissing")
        poster.paste(placeholder, (col4_x + MARGIN, dphe_img_y))
    
    # Determine if DP+HE is best defense
    defense_df = df[df["group"] == "defenses"]
    is_best = False
    if dphe_run is not None and not defense_df.empty:
        dphe_lpips_val = dphe_run.get("LPIPS", float('nan'))
        if not pd.isna(dphe_lpips_val):
            # Check if highest LPIPS among defenses
            max_lpips = defense_df["LPIPS"].max()
            is_best = (dphe_lpips_val >= max_lpips * 0.99)  # Allow small tolerance
    
    # DP+HE metrics card
    dphe_metrics_y = dphe_img_y + dphe_img_size[1] + 15
    dphe_metrics_size = (COLUMN_WIDTH - 2*MARGIN, 140)
    
    if dphe_run is not None:
        dphe_metrics = {
            "PSNR": dphe_run.get("PSNR"),
            "SSIM": dphe_run.get("SSIM"),
            "LPIPS": dphe_run.get("LPIPS"),
            "LabelMatch": dphe_run.get("LabelMatch"),
        }
        dphe_card = create_metrics_card(dphe_metrics, dphe_metrics_size, 
                                        "Combined Defense Metrics", highlight_best=is_best)
        poster.paste(dphe_card, (col4_x + MARGIN, dphe_metrics_y))
    else:
        placeholder = create_missing_placeholder(dphe_metrics_size, "No metrics")
        poster.paste(placeholder, (col4_x + MARGIN, dphe_metrics_y))
    
    # Summary box for DP+HE
    summary_y = dphe_metrics_y + dphe_metrics_size[1] + 15
    summary_h = 120
    
    if is_best:
        box_fill = (232, 245, 233)  # Light green
        box_outline = POSTER_SUCCESS_COLOR
    else:
        box_fill = POSTER_LIGHT_GRAY
        box_outline = POSTER_ACCENT_COLOR
    
    draw.rectangle([col4_x + MARGIN, summary_y, col4_x + COLUMN_WIDTH - MARGIN, summary_y + summary_h],
                   fill=box_fill, outline=box_outline, width=2)
    
    draw.text((col4_x + MARGIN + 10, summary_y + 10), "Defense Summary:", 
              fill=POSTER_DARK_COLOR, font=takeaway_title_font)
    
    if dphe_run is not None:
        if is_best:
            summary = "Combined DP+HE achieves\nbest protection (highest LPIPS)."
        else:
            summary = "DP+HE provides layered\nprivacy protection."
    else:
        summary = "DP+HE data not available."
    
    draw.text((col4_x + MARGIN + 10, summary_y + 35), summary, 
              fill=(80, 80, 80), font=takeaway_font)
    
    # === BOTTOM STRIP: ABLATIONS ===
    strip_y = POSTER_HEIGHT - BOTTOM_STRIP_HEIGHT
    
    # Strip background
    draw.rectangle([0, strip_y, POSTER_WIDTH, POSTER_HEIGHT], fill=(250, 250, 250))
    draw.line([(0, strip_y), (POSTER_WIDTH, strip_y)], fill=POSTER_ACCENT_COLOR, width=3)
    
    # Strip title
    strip_title_font = get_poster_font(32, bold=True)
    draw.text((MARGIN, strip_y + 15), "ABLATION STUDY", fill=POSTER_DARK_COLOR, font=strip_title_font)
    
    ablation_content_y = strip_y + 70
    
    # Left: Ablation bars image
    ablation_bars_path = FIGS_DIR / "ablation_bars.png"
    ablation_bars_size = (int(POSTER_WIDTH * 0.45), BOTTOM_STRIP_HEIGHT - 100)
    
    if ablation_bars_path.exists():
        ablation_bars = load_and_resize_poster(str(ablation_bars_path), ablation_bars_size)
        if ablation_bars:
            poster.paste(ablation_bars, (MARGIN, ablation_content_y))
    else:
        placeholder = create_missing_placeholder(ablation_bars_size, "ablation_bars.png\nMissing")
        poster.paste(placeholder, (MARGIN, ablation_content_y))
    
    # Middle: Ablation summary table
    table_x = int(POSTER_WIDTH * 0.47)
    table_size = (int(POSTER_WIDTH * 0.28), BOTTOM_STRIP_HEIGHT - 120)
    ablation_table = create_ablation_summary_table(df, stats, table_size)
    poster.paste(ablation_table, (table_x, ablation_content_y))
    
    # Right: Key takeaways
    takeaways_x = int(POSTER_WIDTH * 0.77)
    takeaways_w = POSTER_WIDTH - takeaways_x - MARGIN
    
    draw.rectangle([takeaways_x, ablation_content_y, POSTER_WIDTH - MARGIN, POSTER_HEIGHT - MARGIN],
                   fill=(255, 255, 255), outline=POSTER_ACCENT_COLOR, width=2)
    
    draw.text((takeaways_x + 15, ablation_content_y + 10), "Key Takeaways", 
              fill=POSTER_DARK_COLOR, font=get_poster_font(20, bold=True))
    
    conclusions = derive_ablation_conclusions(df, stats)
    bullet_y = ablation_content_y + 45
    bullet_font = get_poster_font(14)
    
    for i, conclusion in enumerate(conclusions):
        # Wrap text if too long
        wrapped = conclusion
        if len(conclusion) > 40:
            mid = len(conclusion) // 2
            space_idx = conclusion.rfind(' ', 0, mid + 10)
            if space_idx > 10:
                wrapped = conclusion[:space_idx] + "\n  " + conclusion[space_idx+1:]
        
        bullet_text = f"• {wrapped}"
        draw.text((takeaways_x + 15, bullet_y), bullet_text, 
                  fill=(60, 60, 60), font=bullet_font)
        
        # Count newlines in wrapped text
        bullet_y += 25 + (wrapped.count('\n') * 18)
    
    # === SAVE POSTER ===
    poster.save(output_path, "PNG", quality=95)
    print(f"  Poster saved to: {output_path}")
    print(f"  Anchor client used: {anchor_client if anchor_client else 'N/A'}")
    
    # Optionally try PDF
    pdf_path = output_path.with_suffix('.pdf')
    try:
        poster.save(pdf_path, "PDF", resolution=100.0)
        print(f"  PDF also saved to: {pdf_path}")
    except Exception as e:
        print(f"  (PDF export skipped: {e})")
    
    return poster


# ============================================================================
# Main
# ============================================================================

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Generate FL experiment report and optional 4K poster."
    )
    parser.add_argument(
        "--poster", 
        action="store_true",
        help="Also generate a 4K (3840x2160) showcase poster"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Federated Learning Experiment Report Generator")
    print("=" * 60)
    print()
    
    # Create output directories
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    MONTAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Parse all metrics
    print("[1/5] Parsing experiment artifacts...")
    df = build_dataframe()
    
    if df.empty:
        print("ERROR: No runs found! Check that results/ folder exists with experiment outputs.")
        sys.exit(1)
    
    print(f"  Total runs parsed: {len(df)}")
    print()
    
    # Step 2: Compute statistics
    print("[2/5] Computing statistics...")
    leaderboard = compute_leaderboard(df)
    per_setting_stats = compute_per_setting_stats(df)
    defense_deltas = compute_defense_deltas(df)
    print(f"  Settings analyzed: {len(per_setting_stats)}")
    print()
    
    # Step 3: Save CSV and Markdown
    print("[3/5] Saving summary files...")
    df.to_csv(REPORT_DIR / "summary.csv", index=False)
    print(f"  Saved: {REPORT_DIR / 'summary.csv'}")
    
    md_report = generate_markdown_report(df, leaderboard, per_setting_stats, defense_deltas)
    with open(REPORT_DIR / "summary.md", "w") as f:
        f.write(md_report)
    print(f"  Saved: {REPORT_DIR / 'summary.md'}")
    print()
    
    # Step 4: Generate figures
    print("[4/5] Generating figures...")
    plot_ablation_bars(df, per_setting_stats, FIGS_DIR)
    plot_multiclient_boxplots(df, FIGS_DIR)
    plot_defense_comparison(df, FIGS_DIR)
    print()
    
    # Step 5: Generate montages
    print("[5/5] Generating montages...")
    generate_montage_best_overall(df, MONTAGES_DIR)
    generate_montage_ablation_best(df, MONTAGES_DIR)
    generate_montage_multiclient(df, MONTAGES_DIR)
    generate_montage_defenses(df, MONTAGES_DIR)
    print()
    
    # Step 6: Generate 4K poster (if requested)
    if args.poster:
        print("[6/6] Generating 4K showcase poster...")
        poster_path = REPORT_DIR / "poster_4k.png"
        generate_4k_poster(df, per_setting_stats, poster_path)
        print()
    
    # Final summary
    print("=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)
    print()
    print(f"Total runs parsed: {len(df)}")
    print(f"  - showcase: {len(df[df['group'] == 'showcase'])}")
    print(f"  - multi_client: {len(df[df['group'] == 'multi_client'])}")
    print(f"  - ablation: {len(df[df['group'] == 'ablation'])}")
    print(f"  - defenses: {len(df[df['group'] == 'defenses'])}")
    print()
    
    # Best run
    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        print(f"Best overall run: {best['setting']}", end="")
        if best['client']:
            print(f" ({best['client']})", end="")
        print(f" - LPIPS={best['LPIPS']:.4f}, SSIM={best['SSIM']:.4f}, PSNR={best['PSNR']:.4f}")
    print()
    
    print("Output locations:")
    print(f"  Summary CSV:    {REPORT_DIR / 'summary.csv'}")
    print(f"  Summary MD:     {REPORT_DIR / 'summary.md'}")
    print(f"  Figures:        {FIGS_DIR}/")
    print(f"  Montages:       {MONTAGES_DIR}/")
    if args.poster:
        print(f"  4K Poster:      {REPORT_DIR / 'poster_4k.png'}")
    print()
    print("Run command: python scripts/make_report.py [--poster]")
    print()


if __name__ == "__main__":
    main()
