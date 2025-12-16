#!/usr/bin/env python3
"""
make_report.py - Generate a comprehensive research outcome report from FL experiment artifacts.

Usage: python scripts/make_report.py

Outputs:
  results/report/summary.csv       - Full dataframe of all parsed metrics
  results/report/summary.md        - Human-readable markdown report
  results/report/figs/             - Publication-quality PNG figures
  results/report/montages/         - Image montages for slides

Dependencies: stdlib + pandas + matplotlib + pillow
"""

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
# Main
# ============================================================================

def main():
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
    print()
    print("Run command: python scripts/make_report.py")
    print()


if __name__ == "__main__":
    main()
