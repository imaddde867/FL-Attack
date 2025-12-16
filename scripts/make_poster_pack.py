
import os
import glob
import json
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Constants
METRICS_FILE = 'metrics.txt'
CONFIG_FILE = 'config.json'
IMAGE_FILE = 'baseline_attack_result.png'
REPORT_DIR = os.path.join('results', 'report')

def setup_fonts():
    # Try to find a decent font, fallback to default
    try:
        # MacOS/Linux common paths
        possible_fonts = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ]
        for font_path in possible_fonts:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, 20)
    except:
        pass
    return ImageFont.load_default()

FONT = setup_fonts()

def parse_metrics(file_path):
    """Parses metrics.txt into a dictionary."""
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, val = line.strip().split(':', 1)
                    try:
                        metrics[key.strip()] = float(val.strip())
                    except ValueError:
                        metrics[key.strip()] = val.strip()
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
    return metrics

def infer_defense_info(config, path):
    """Infers defense type and key parameters from config or path."""
    info = {
        'defense': 'Baseline',
        'params': '',
        'param_val': 0.0,
        'client': config.get('capture_client', 0),
        'round': config.get('capture_round', 'last')
    }
    
    # 1. Check Config
    dp = config.get('dp_epsilon')
    he = config.get('use_he')
    
    if dp is not None and he:
        info['defense'] = 'DP+HE'
        info['params'] = f"ε={dp}, HE={config.get('he_bits')}b"
        info['param_val'] = dp # Primary sort key
    elif dp is not None:
        info['defense'] = 'DP'
        info['params'] = f"ε={dp}"
        info['param_val'] = dp
    elif he:
        info['defense'] = 'HE'
        info['params'] = f"HE={config.get('he_bits')}b"
        info['param_val'] = config.get('he_bits')
    else:
        # Check path for keywords if config missing specific flags (fallback)
        path_lower = path.lower()
        if 'dp_' in path_lower and 'he_' in path_lower:
             info['defense'] = 'DP+HE'
        elif 'dp_' in path_lower:
             info['defense'] = 'DP'
        elif 'he_' in path_lower:
             info['defense'] = 'HE'
    
    # Ablation detection (e.g. if defense is baseline but path has 'ablation')
    if info['defense'] == 'Baseline' and 'ablation' in path.lower():
         info['defense'] = 'Ablation'
         # Try to extract what is being ablated
         if 'tv_' in path.lower(): info['params'] = 'TV'
    
    return info

def scan_results(root_dirs):
    """Recursively scans directories for experiment runs."""
    records = []
    
    # Allow passing a single string or list
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
        
    for root_dir in root_dirs:
        logger.info(f"Scanning {root_dir}...")
        for root, dirs, files in os.walk(root_dir):
            if METRICS_FILE in files:
                # Found a run
                run_path = root
                metrics_path = os.path.join(root, METRICS_FILE)
                config_path = os.path.join(root, CONFIG_FILE)
                img_path = os.path.join(root, IMAGE_FILE)
                
                # Parse
                metrics = parse_metrics(metrics_path)
                
                config = {}
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                    except:
                        pass
                
                defense_info = infer_defense_info(config, run_path)
                
                record = {
                    'path': run_path,
                    'image_path': img_path if os.path.exists(img_path) else None,
                    **metrics,
                    **defense_info,
                    **config # Flatten config for advanced analysis if needed
                }
                records.append(record)
    
    logger.info(f"Found {len(records)} runs.")
    return pd.DataFrame(records)

def crop_image_panel(img_path, section='recon'):
    """
    Crops specific section from the vertical stack output.
    Assumes standard 3-row layout: Orig, Recon, Diff.
    """
    if not img_path: return None
    try:
        img = Image.open(img_path)
        w, h = img.size
        # Assume 3 rows
        row_h = h // 3
        
        if section == 'orig':
            return img.crop((0, 0, w, row_h))
        elif section == 'recon':
            return img.crop((0, row_h, w, row_h * 2))
        elif section == 'diff':
            return img.crop((0, row_h * 2, w, h))
        else:
            return img
    except Exception as e:
        logger.error(f"Error cropping {img_path}: {e}")
        return None

def make_hero_panel(df, out_dir):
    """Creates side-by-side comparison of Baseline vs Defenses for the same client."""
    # Group by client to find a common one
    if 'capture_client' not in df.columns:
        return
        
    # Prefer client 0 or the most frequent one
    client_id = df['capture_client'].mode()[0] if not df['capture_client'].empty else 0
    subset = df[df['capture_client'] == client_id]
    
    # We want one representative per Defense type
    # Baseline, DP (best/mid), HE, DP+HE
    defenses = ['Baseline', 'DP', 'HE', 'DP+HE']
    selected_runs = {}
    
    for d in defenses:
        matches = subset[subset['defense'] == d]
        if not matches.empty:
            # Pick the one with median LPIPS or best PSNR? Let's pick 'best' PSNR for visual clarity
            # or 'worst' for defense efficacy? 
            # For "Vulnerability", Baseline should look good (high PSNR). 
            # Defenses should look bad (low PSNR).
            # So let's pick the run with the *median* PSNR to be representative.
            median_idx = matches['PSNR'].argsort()[len(matches)//2]
            selected_runs[d] = matches.iloc[median_idx]
            
    if not selected_runs:
        logger.warning("No runs found for Hero Panel.")
        return

    # Create composite
    # Width = sum of widths, Height = max height
    # Layout: [ Label ]
    #         [ Image ]
    #         [ PSNR  ]
    
    images = []
    
    # First, get the 'Original' from the Baseline run
    if 'Baseline' in selected_runs and selected_runs['Baseline']['image_path']:
        orig_img = crop_image_panel(selected_runs['Baseline']['image_path'], 'orig')
        if orig_img:
            images.append(('Original', '', orig_img))
            
    for d in defenses:
        if d in selected_runs and selected_runs[d]['image_path']:
            row = selected_runs[d]
            img = crop_image_panel(row['image_path'], 'recon')
            if img:
                meta = f"PSNR: {row.get('PSNR',0):.2f}\nSSIM: {row.get('SSIM',0):.2f}"
                images.append((d, meta, img))
    
    if not images:
        return

    # Stitch
    w, h = images[0][2].size
    total_w = w * len(images)
    total_h = h + 60 # Space for text
    
    final_img = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(final_img)
    
    for i, (label, meta, img) in enumerate(images):
        x = i * w
        # Resize if mismatch (robustness)
        if img.size != (w, h): img = img.resize((w, h))
            
        final_img.paste(img, (x, 40))
        
        # Draw Label (Top)
        text_bbox = draw.textbbox((0,0), label, font=FONT)
        text_w = text_bbox[2] - text_bbox[0]
        draw.text((x + (w - text_w)/2, 10), label, fill='black', font=FONT)
        
        # Draw Meta (Bottom overlay or below? Below is cleaner but I allocated header space only)
        # Let's write inside the image at bottom or just below text
        # Actually I added 60px height but pasted at y=40. So 20px bottom margin.
        # Let's put meta text ON the image (white with black border) or just keep it simple.
        # User requested "readable labels".
        
    final_path = os.path.join(out_dir, 'hero_panel.png')
    final_img.save(final_path)
    logger.info(f"Generated {final_path}")

def make_vulnerability_plot(df, out_dir):
    """Boxplot of metrics by Defense."""
    if df.empty: return
    
    plt.figure(figsize=(10, 6))
    
    # Filter for main metric
    metric = 'LPIPS' if 'LPIPS' in df and not df['LPIPS'].isna().all() else 'SSIM'
    
    # Prepare data
    data_to_plot = []
    labels = []
    
    defenses = ['Baseline', 'DP', 'HE', 'DP+HE']
    for d in defenses:
        subset = df[df['defense'] == d]
        if not subset.empty:
            vals = subset[metric].dropna()
            if len(vals) > 0:
                data_to_plot.append(vals)
                labels.append(f"{d}\n(N={len(vals)})")
    
    if not data_to_plot: return
        
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    plt.title(f"Vulnerability Analysis: {metric} Distribution")
    plt.ylabel(f"{metric} (Lower is Better)" if metric == 'LPIPS' else f"{metric} (Higher is Better)")
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.savefig(os.path.join(out_dir, 'vulnerability_plot.png'), dpi=300)
    plt.close()

def make_tradeoff_plot(df, out_dir):
    """Scatter plot: Privacy Param vs Reconstruction Quality."""
    # Focus on DP epsilon
    dp_runs = df[df['defense'].isin(['DP', 'DP+HE'])].copy()
    if dp_runs.empty or 'dp_epsilon' not in dp_runs.columns:
        return
        
    plt.figure(figsize=(8, 6))
    
    metric = 'LPIPS' if 'LPIPS' in dp_runs and not dp_runs['LPIPS'].isna().all() else 'SSIM'
    
    # Plot DP
    subset = dp_runs[dp_runs['defense'] == 'DP']
    if not subset.empty:
        plt.scatter(subset['dp_epsilon'], subset[metric], label='DP Only', alpha=0.7, s=100)
        
    # Plot DP+HE
    subset_he = dp_runs[dp_runs['defense'] == 'DP+HE']
    if not subset_he.empty:
        plt.scatter(subset_he['dp_epsilon'], subset_he[metric], label='DP + HE', marker='^', alpha=0.7, s=100)
        
    plt.xlabel('Privacy Budget (\epsilon)')
    plt.ylabel(f"Reconstruction Quality ({metric})")
    plt.title("Privacy-Utility Trade-off")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(out_dir, 'privacy_tradeoff.png'), dpi=300)
    plt.close()

def make_montage(df, out_dir, filename, sort_by='LPIPS', ascending=True, top_n=10, title="Montage"):
    """Grid of reconstruction images."""
    if df.empty: return
    
    # Sort
    if sort_by in df.columns:
        df_sorted = df.sort_values(by=sort_by, ascending=ascending).head(top_n)
    else:
        df_sorted = df.head(top_n)
        
    images = []
    for _, row in df_sorted.iterrows():
        if row['image_path']:
            img = crop_image_panel(row['image_path'], 'recon')
            if img:
                # Add label
                draw = ImageDraw.Draw(img)
                label = f"{row['defense']} {row.get(sort_by, 0):.2f}"
                draw.text((5, 5), label, fill=(255, 0, 0), font=FONT)
                images.append(img)
                
    if not images: return
    
    # Grid calculation
    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    w, h = images[0].size
    grid_img = Image.new('RGB', (w * cols, h * rows), (255, 255, 255))
    
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        if img.size != (w, h): img = img.resize((w, h))
        grid_img.paste(img, (c * w, r * h))
        
    grid_img.save(os.path.join(out_dir, filename))

def main():
    parser = argparse.ArgumentParser(description="Generate Publication-Ready Report")
    parser.add_argument('--input', nargs='+', default=['results'], help='Input directories to scan')
    args = parser.parse_args()
    
    # 1. Setup
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
        
    # 2. Scan
    df = scan_results(args.input)
    if df.empty:
        print("No results found. Run experiments first!")
        return

    # 3. CSV Outputs
    df.to_csv(os.path.join(REPORT_DIR, 'summary.csv'), index=False)
    
    # Aggregate
    agg_cols = ['defense']
    if 'dp_epsilon' in df.columns: agg_cols.append('dp_epsilon')
    
    # Select numeric columns only for aggregation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg_df = df.groupby(agg_cols)[numeric_cols].agg(['mean', 'std', 'count'])
    agg_df.to_csv(os.path.join(REPORT_DIR, 'aggregated.csv'))
    
    # 4. Figures
    print("Generating Figures...")
    make_hero_panel(df, REPORT_DIR)
    make_vulnerability_plot(df, REPORT_DIR)
    make_tradeoff_plot(df, REPORT_DIR)
    
    # 5. Montages
    print("Generating Montages...")
    # Best Reconstructions (Lowest LPIPS or Highest PSNR)
    # If LPIPS exists, lower is better. If only PSNR, higher is better.
    if 'LPIPS' in df.columns and not df['LPIPS'].isna().all():
        make_montage(df, REPORT_DIR, 'montage_best_recons.png', sort_by='LPIPS', ascending=True, top_n=9)
        make_montage(df, REPORT_DIR, 'montage_worst_recons.png', sort_by='LPIPS', ascending=False, top_n=9)
    else:
        make_montage(df, REPORT_DIR, 'montage_best_recons.png', sort_by='PSNR', ascending=False, top_n=9)
    
    # 6. Report Markdown
    with open(os.path.join(REPORT_DIR, 'summary.md'), 'w') as f:
        f.write("# Federated Learning Gradient Inversion Report\n\n")
        f.write("## Executive Summary\n")
        f.write(f"Analyzed **{len(df)}** experiment runs.\n\n")
        
        f.write("## Aggregated Metrics\n")
        f.write(agg_df.to_markdown())
        f.write("\n\n")
        
        f.write("## Visualizations\n")
        f.write("### Vulnerability Analysis\n")
        f.write("![Vulnerability](vulnerability_plot.png)\n\n")
        
        f.write("### Hero Comparison\n")
        f.write("![Hero](hero_panel.png)\n\n")
        
    # 7. Manifest
    manifest = {
        'generated_at': str(pd.Timestamp.now()),
        'input_dirs': args.input,
        'run_count': len(df),
        'files': glob.glob(os.path.join(REPORT_DIR, '*'))
    }
    with open(os.path.join(REPORT_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nReport generated in: {REPORT_DIR}")
    print(f"Summary CSV: {os.path.join(REPORT_DIR, 'summary.csv')}")
    print(f"Markdown:    {os.path.join(REPORT_DIR, 'summary.md')}")

if __name__ == "__main__":
    main()
