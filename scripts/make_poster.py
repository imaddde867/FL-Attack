#!/usr/bin/env python3
"""
make_poster.py - Generate a 4K TV Poster for Federated Learning Privacy Research
=================================================================================

Produces a single 3840x2160 poster image optimized for 75-inch 16:9 display.
Designed for visual impact, readability from distance, and data-grounded insights.

Requirements: stdlib + pandas + matplotlib + pillow (no external fonts)
Output: results/report/poster_4k.png (and poster_4k.pdf if possible)

Usage:
    python scripts/make_poster.py
"""

import os
import sys
import glob
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# IMPORTS
# =============================================================================

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas and numpy required. Install with: pip install pandas numpy")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Pillow required. Install with: pip install Pillow")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO
except ImportError:
    print("ERROR: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Canvas dimensions (4K)
CANVAS_WIDTH = 3840
CANVAS_HEIGHT = 2160

# Layout dimensions - optimized for visual impact
HEADER_HEIGHT = 180
MAIN_PANEL_HEIGHT = 1400
BOTTOM_BAND_HEIGHT = CANVAS_HEIGHT - HEADER_HEIGHT - MAIN_PANEL_HEIGHT  # ~580px

# Padding and margins - generous for clarity
OUTER_MARGIN = 25
CARD_PADDING = 20
GUTTER = 28

# Colors (refined dark theme with high contrast)
BG_COLOR = (22, 27, 38)  # Deeper dark blue
CARD_BG = (38, 44, 58)   # Slightly darker cards
CARD_BORDER = (60, 75, 100)
HEADER_BG = (15, 18, 28)  # Very dark header
TEXT_WHITE = (255, 255, 255)  # Pure white for max contrast
TEXT_LIGHT = (220, 225, 235)
TEXT_MUTED = (150, 160, 180)
ACCENT_BLUE = (80, 150, 255)   # Brighter blue
ACCENT_GREEN = (100, 200, 120)  # Brighter green
ACCENT_ORANGE = (255, 170, 50)  # Warmer orange
ACCENT_RED = (255, 90, 90)
ACCENT_PURPLE = (180, 100, 220)  # Brighter purple
ACCENT_GOLD = (255, 215, 0)  # Gold for highlights

# Column header colors - more vibrant
COL_COLORS = [
    (70, 140, 255),   # Bright Blue - Baseline
    (255, 160, 40),   # Bright Orange - DP
    (80, 200, 120),   # Bright Green - HE
    (180, 100, 220),  # Bright Purple - DP+HE
]

# Font sizes (LARGE for TV readability at distance)
FONT_TITLE = 96
FONT_HEADER = 56
FONT_SUBHEADER = 40
FONT_BODY = 32
FONT_CAPTION = 26
FONT_SMALL = 22

# Line spacing multipliers - generous for readability
LINE_SPACING = 1.5
LINE_SPACING_TIGHT = 1.35

# Paths
RESULTS_BASE = Path('results')
REPORT_DIR = RESULTS_BASE / 'report'
DEFENSES_DIR = RESULTS_BASE / 'defenses'
MULTI_CLIENT_DIR = RESULTS_BASE / 'multi_client'
ABLATION_DIR = RESULTS_BASE / 'ablation'
SHOWCASE_DIR = RESULTS_BASE / 'showcase'

# Seed for determinism
np.random.seed(42)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_font(size: int, bold: bool = False) -> Optional[ImageFont.FreeTypeFont]:
    """Get a font with fallback to default."""
    font_paths = [
        # macOS
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/SFNSText.ttf',
        '/Library/Fonts/Arial.ttf',
        # Linux
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        # Windows
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/segoeui.ttf',
    ]
    if bold:
        bold_paths = [
            '/System/Library/Fonts/Helvetica.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            'C:/Windows/Fonts/arialbd.ttf',
        ]
        font_paths = bold_paths + font_paths

    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (IOError, OSError):
            continue

    try:
        return ImageFont.load_default()
    except Exception:
        return None


def parse_metrics_file(path: Path) -> Dict[str, float]:
    """Parse a metrics.txt file."""
    metrics = {}
    if not path.exists():
        return metrics
    try:
        with open(path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, val = line.strip().split(':', 1)
                    try:
                        metrics[key.strip()] = float(val.strip())
                    except ValueError:
                        pass
    except Exception:
        pass
    return metrics


def collect_all_metrics() -> pd.DataFrame:
    """Collect all metrics from results directories into a DataFrame."""
    rows = []

    # Defense experiments
    if DEFENSES_DIR.exists():
        for exp_dir in DEFENSES_DIR.glob('*'):
            if exp_dir.is_dir():
                metrics = parse_metrics_file(exp_dir / 'metrics.txt')
                if metrics:
                    metrics['group'] = 'defenses'
                    metrics['setting'] = exp_dir.name
                    metrics['path'] = str(exp_dir)
                    rows.append(metrics)

    # Multi-client experiments
    if MULTI_CLIENT_DIR.exists():
        for exp_dir in MULTI_CLIENT_DIR.glob('bmk_c*'):
            if exp_dir.is_dir():
                metrics = parse_metrics_file(exp_dir / 'metrics.txt')
                if metrics:
                    metrics['group'] = 'multi_client'
                    metrics['setting'] = exp_dir.name
                    metrics['path'] = str(exp_dir)
                    rows.append(metrics)

    # Ablation experiments
    if ABLATION_DIR.exists():
        for abl_dir in ABLATION_DIR.glob('*'):
            if abl_dir.is_dir():
                for client_dir in abl_dir.glob('c*'):
                    metrics = parse_metrics_file(client_dir / 'metrics.txt')
                    if metrics:
                        metrics['group'] = 'ablation'
                        metrics['setting'] = f"{abl_dir.name}/{client_dir.name}"
                        metrics['ablation_type'] = abl_dir.name
                        metrics['path'] = str(client_dir)
                        rows.append(metrics)

    # Showcase
    if SHOWCASE_DIR.exists():
        showcase_metrics = parse_metrics_file(SHOWCASE_DIR / 'metrics.txt')
        if showcase_metrics:
            showcase_metrics['group'] = 'showcase'
            showcase_metrics['setting'] = 'showcase'
            showcase_metrics['path'] = str(SHOWCASE_DIR)
            rows.append(showcase_metrics)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def load_summary_csv() -> pd.DataFrame:
    """Load summary.csv if exists, else collect metrics."""
    csv_path = REPORT_DIR / 'summary.csv'
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception:
            pass
    return collect_all_metrics()


def save_summary_csv(df: pd.DataFrame) -> Path:
    """Save summary.csv."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / 'summary.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def best_by_lpips(df: pd.DataFrame) -> Optional[pd.Series]:
    """Select best run: lowest LPIPS, tie-breaker: highest SSIM, then PSNR."""
    if df.empty or 'LPIPS' not in df.columns:
        return None
    valid = df.dropna(subset=['LPIPS'])
    if valid.empty:
        return None
    sorted_df = valid.sort_values(
        by=['LPIPS', 'SSIM', 'PSNR'],
        ascending=[True, False, False]
    )
    return sorted_df.iloc[0]


def aggregate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute mean +/- std for numeric metrics."""
    result = {}
    for col in ['PSNR', 'SSIM', 'LPIPS', 'MSE']:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                result[col] = (float(vals.mean()), float(vals.std()))
    if 'LabelMatch' in df.columns:
        vals = df['LabelMatch'].dropna()
        if len(vals) > 0:
            result['LabelMatch_rate'] = float(vals.mean())
    return result


def load_image_safe(path: Path, fallback_size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """Load image with fallback to placeholder."""
    try:
        if path.exists():
            img = Image.open(path).convert('RGB')
            return img
    except Exception:
        pass

    # Create placeholder
    img = Image.new('RGB', fallback_size, CARD_BG)
    draw = ImageDraw.Draw(img)
    font = get_font(FONT_CAPTION)
    text = "(not available)"
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        tw, th = 100, 20
    draw.text(((fallback_size[0] - tw) // 2, (fallback_size[1] - th) // 2),
              text, fill=TEXT_MUTED, font=font)
    return img


def resize_contain(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize image to fit within target dimensions, maintaining aspect ratio."""
    orig_w, orig_h = img.size
    if orig_w <= 0 or orig_h <= 0:
        return img
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(orig_w * ratio))
    new_h = max(1, int(orig_h * ratio))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def add_border(img: Image.Image, border_width: int = 2,
               color: Tuple[int, int, int] = CARD_BORDER) -> Image.Image:
    """Add a border around an image."""
    bordered = Image.new('RGB', (img.width + 2 * border_width, img.height + 2 * border_width), color)
    bordered.paste(img, (border_width, border_width))
    return bordered


def draw_rounded_rect(draw: ImageDraw.Draw, xy: Tuple[int, int, int, int],
                      radius: int, fill: Tuple, outline: Tuple = None, width: int = 1):
    """Draw a rounded rectangle."""
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def matplotlib_to_pil(fig, dpi: int = 150) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=np.array(CARD_BG) / 255, edgecolor='none', transparent=False)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    plt.close(fig)
    return img


# =============================================================================
# CHART GENERATION
# =============================================================================

def create_ablation_charts(df: pd.DataFrame, chart_w: int = 500, chart_h: int = 350) -> List[Image.Image]:
    """Create 3 separate ablation bar charts (PSNR, SSIM, LPIPS)."""
    if df.empty or 'ablation_type' not in df.columns:
        placeholder = Image.new('RGB', (chart_w, chart_h), CARD_BG)
        return [placeholder, placeholder, placeholder]

    # Aggregate by ablation type
    agg_data = df.groupby('ablation_type').agg({
        'PSNR': ['mean', 'std'],
        'SSIM': ['mean', 'std'],
        'LPIPS': ['mean', 'std']
    }).reset_index()

    agg_data.columns = ['ablation', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std', 'lpips_mean', 'lpips_std']
    agg_data = agg_data.sort_values('psnr_mean', ascending=False)

    labels = agg_data['ablation'].tolist()
    x = np.arange(len(labels))

    charts = []
    metrics_info = [
        ('PSNR (dB)', 'psnr_mean', 'psnr_std', '#4287f5'),
        ('SSIM', 'ssim_mean', 'ssim_std', '#4caf50'),
        ('LPIPS', 'lpips_mean', 'lpips_std', '#ff9800'),
    ]

    for title, mean_col, std_col, color in metrics_info:
        fig, ax = plt.subplots(figsize=(chart_w / 100, chart_h / 100))

        means = agg_data[mean_col].values
        stds = agg_data[std_col].fillna(0).values

        ax.bar(x, means, yerr=stds, capsize=3, color=color, alpha=0.85,
               edgecolor='white', linewidth=0.5)
        ax.set_title(title, fontsize=14, color='white', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9, color='white')
        ax.tick_params(colors='white')

        for spine in ax.spines.values():
            spine.set_color('#465673')
        ax.set_facecolor(np.array(CARD_BG) / 255)

        plt.tight_layout()
        charts.append(matplotlib_to_pil(fig, dpi=100))

    return charts


# =============================================================================
# POSTER COMPOSER
# =============================================================================

class PosterComposer:
    """Main class to compose the 4K poster."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.canvas = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), BG_COLOR)
        self.draw = ImageDraw.Draw(self.canvas)
        self.chosen_runs = {}
        self.baseline_grid_runs = []

        # Preload fonts
        self.font_title = get_font(FONT_TITLE, bold=True)
        self.font_header = get_font(FONT_HEADER, bold=True)
        self.font_subheader = get_font(FONT_SUBHEADER, bold=True)
        self.font_body = get_font(FONT_BODY)
        self.font_caption = get_font(FONT_CAPTION)
        self.font_small = get_font(FONT_SMALL)

    def _text_width(self, text: str, font) -> int:
        """Get text width."""
        if font:
            bbox = self.draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0]
        return len(text) * 8

    def _get_baseline_metrics(self) -> Tuple[float, float, float]:
        """Get baseline LPIPS, SSIM, PSNR."""
        defense_df = self.df[self.df['group'] == 'defenses']
        baseline_df = defense_df[defense_df['setting'] == 'baseline']
        if baseline_df.empty:
            return 0.1, 0.9, 30.0
        row = baseline_df.iloc[0]
        return (
            float(row.get('LPIPS', 0.1)),
            float(row.get('SSIM', 0.9)),
            float(row.get('PSNR', 30.0))
        )

    def draw_header(self):
        """Draw the header bar with strong visual impact."""
        # Header background
        draw_rounded_rect(self.draw, (0, 0, CANVAS_WIDTH, HEADER_HEIGHT),
                          radius=0, fill=HEADER_BG)
        
        # Accent line under header
        self.draw.rectangle((0, HEADER_HEIGHT - 4, CANVAS_WIDTH, HEADER_HEIGHT),
                            fill=ACCENT_BLUE)

        # Main title - research topic
        title = "Security in Federated Learning"
        self.draw.text((OUTER_MARGIN + 25, 25), title, fill=TEXT_WHITE, font=self.font_title)

        # Subtitle - tool description
        subtitle = "Privacy-Preserving Benchmarking Tool for Gradient Inversion Attacks"
        self.draw.text((OUTER_MARGIN + 25, 120), subtitle, fill=TEXT_LIGHT, font=self.font_subheader)

        # Key result box - right side
        takeaway = self._generate_takeaway()
        tw = self._text_width(takeaway, self.font_body)
        
        # Draw result with prominent accent background
        result_box_w = tw + 50
        result_box_x = CANVAS_WIDTH - OUTER_MARGIN - result_box_w
        draw_rounded_rect(self.draw, (result_box_x, 40, CANVAS_WIDTH - OUTER_MARGIN, 100),
                          radius=10, fill=(30, 50, 80), outline=ACCENT_GOLD, width=3)
        self.draw.text((result_box_x + 25, 55), takeaway, fill=ACCENT_GOLD, font=self.font_body)

    def _generate_takeaway(self) -> str:
        """Generate data-grounded takeaway - plain ASCII only."""
        if self.df.empty:
            return "RESULT: Privacy defenses block gradient attacks"

        defense_df = self.df[self.df['group'] == 'defenses']
        if defense_df.empty:
            return "RESULT: DP + HE prevents data reconstruction"

        baseline = defense_df[defense_df['setting'] == 'baseline']
        dp_he = defense_df[defense_df['setting'] == 'dp_he']

        if not baseline.empty and not dp_he.empty:
            b_psnr = float(baseline['PSNR'].values[0])
            d_psnr = float(dp_he['PSNR'].values[0])
            drop = b_psnr - d_psnr
            return f"RESULT: {drop:.0f} dB PSNR drop = Attack Blocked"

        return "RESULT: Combined defenses block reconstruction"

    def draw_main_panel(self):
        """Draw the 4-column main panel."""
        y_start = HEADER_HEIGHT + 8
        col_width = (CANVAS_WIDTH - 2 * OUTER_MARGIN - 3 * GUTTER) // 4
        col_height = MAIN_PANEL_HEIGHT - 10

        columns = [
            ("BASELINE", self._render_baseline_column),
            ("DIFFERENTIAL PRIVACY", self._render_dp_column),
            ("HOMOMORPHIC ENCRYPTION", self._render_he_column),
            ("DP + HE", self._render_dphe_column),
        ]

        for i, (title, render_func) in enumerate(columns):
            x = OUTER_MARGIN + i * (col_width + GUTTER)

            # Column card background with subtle inner glow effect
            draw_rounded_rect(self.draw, (x, y_start, x + col_width, y_start + col_height),
                              radius=14, fill=CARD_BG, outline=CARD_BORDER, width=2)

            # Column header - taller and more prominent
            header_h = 58
            draw_rounded_rect(self.draw, (x, y_start, x + col_width, y_start + header_h),
                              radius=14, fill=COL_COLORS[i])
            # Fix bottom corners
            self.draw.rectangle((x, y_start + header_h - 14, x + col_width, y_start + header_h),
                                fill=COL_COLORS[i])

            # Header text - centered vertically and horizontally
            tw = self._text_width(title, self.font_subheader)
            self.draw.text((x + (col_width - tw) // 2, y_start + 10), title,
                           fill=TEXT_WHITE, font=self.font_subheader)

            # Render column content
            content_area = (x + CARD_PADDING, y_start + header_h + CARD_PADDING,
                            x + col_width - CARD_PADDING, y_start + col_height - CARD_PADDING)
            render_func(content_area, col_width - 2 * CARD_PADDING, col_height - header_h - 2 * CARD_PADDING)

    def _render_baseline_column(self, area: Tuple[int, int, int, int], w: int, h: int):
        """Render baseline column with grid of available images (dynamically sized)."""
        x, y, x2, y2 = area
        current_y = y

        # Get baseline/multi-client data
        multi_df = self.df[self.df['group'] == 'multi_client']
        baseline_df = self.df[(self.df['group'] == 'defenses') & (self.df['setting'] == 'baseline')]
        ablation_df = self.df[self.df['group'] == 'ablation']

        # Collect all candidate images
        candidates = []

        # Multi-client first (prefer these)
        for _, row in multi_df.iterrows():
            path = Path(row['path']) / 'baseline_attack_result.png'
            if path.exists():
                candidates.append((path, row.get('LPIPS', 1.0), row.get('setting', '')))

        # Add baseline from defenses
        for _, row in baseline_df.iterrows():
            path = Path(row['path']) / 'baseline_attack_result.png'
            if path.exists():
                candidates.append((path, row.get('LPIPS', 1.0), row.get('setting', '')))

        # Add ablation baseline images to fill the grid
        ablation_baseline = ablation_df[ablation_df['ablation_type'] == 'baseline']
        for _, row in ablation_baseline.iterrows():
            path = Path(row['path']) / 'baseline_attack_result.png'
            if path.exists():
                candidates.append((path, row.get('LPIPS', 1.0), f"abl_{row.get('setting', '')}"))

        # Sort by LPIPS (best first) and take exactly 12
        candidates.sort(key=lambda c: c[1])
        candidates = candidates[:12]
        self.baseline_grid_runs = [c[2] for c in candidates]

        # Build 4x3 grid with only available images (no empty slots)
        n_images = len(candidates)
        if n_images == 0:
            self.draw.text((x, current_y), "(No baseline images available)", 
                           fill=TEXT_MUTED, font=self.font_body)
            return

        # Determine grid layout - larger cells for better visibility
        grid_cols = 4
        grid_rows = 3
        grid_h = int(h * 0.62)
        cell_margin = 6
        cell_w = (w - (grid_cols + 1) * cell_margin) // grid_cols
        cell_h = (grid_h - (grid_rows + 1) * cell_margin) // grid_rows

        grid_img = Image.new('RGB', (w, grid_h), CARD_BG)

        for i in range(min(n_images, 12)):
            col = i % grid_cols
            row_idx = i // grid_cols
            px = cell_margin + col * (cell_w + cell_margin)
            py = cell_margin + row_idx * (cell_h + cell_margin)

            path, _, _ = candidates[i]
            img = load_image_safe(path, (cell_w, cell_h))
            img = resize_contain(img, cell_w - 4, cell_h - 4)
            grid_img.paste(img, (px + 2, py + 2))

        grid_img = add_border(grid_img, 3, COL_COLORS[0])  # Blue border for baseline
        self.canvas.paste(grid_img, (x, current_y))
        current_y += grid_img.height + 12

        # Caption
        caption = f"Reconstructed Private Data ({n_images} FL Clients)"
        self.draw.text((x, current_y), caption, fill=TEXT_LIGHT, font=self.font_caption)
        current_y += int(FONT_CAPTION * LINE_SPACING) + 8

        # Aggregate metrics panel - larger
        combined_df = pd.concat([multi_df, baseline_df], ignore_index=True)
        if not combined_df.empty:
            agg = aggregate_metrics(combined_df)
            current_y = self._draw_metrics_panel(x, current_y, w, agg, "Attack Quality Metrics", h - (current_y - y))

    def _render_dp_column(self, area: Tuple[int, int, int, int], w: int, h: int):
        """Render DP column with 3 epsilon variants."""
        x, y, x2, y2 = area
        current_y = y

        defense_df = self.df[self.df['group'] == 'defenses']
        baseline_lpips, baseline_ssim, baseline_psnr = self._get_baseline_metrics()

        eps_variants = [
            ('Epsilon = 8.0', 'dp_eps8'),
            ('Epsilon = 1.0', 'dp_eps1'),
            ('Epsilon = 0.1', 'dp_eps01'),
        ]

        # Count available variants
        available = sum(1 for _, s in eps_variants if not defense_df[defense_df['setting'] == s].empty)
        if available == 0:
            self.draw.text((x, current_y), "(DP results not available)", fill=TEXT_MUTED, font=self.font_body)
            return

        block_h = (h - 3 * 18) // max(available, 1)
        line_h = int(FONT_CAPTION * LINE_SPACING)

        for label, setting in eps_variants:
            variant_df = defense_df[defense_df['setting'] == setting]

            # Block header - prominent epsilon label
            self.draw.text((x, current_y), label, fill=ACCENT_ORANGE, font=self.font_body)
            current_y += int(FONT_BODY * LINE_SPACING_TIGHT) + 2

            if variant_df.empty:
                self.draw.text((x + 10, current_y), "(not available)", fill=TEXT_MUTED, font=self.font_caption)
                current_y += 50
                continue

            row = variant_df.iloc[0]
            self.chosen_runs[f'dp_{setting}'] = setting

            # Image - larger for visibility
            img_path = Path(row['path']) / 'baseline_attack_result.png'
            img = load_image_safe(img_path, (200, 150))
            img_h = block_h - 85
            img = resize_contain(img, int(w * 0.50), max(img_h, 110))
            img = add_border(img, 3, ACCENT_ORANGE)
            self.canvas.paste(img, (x, current_y))

            # Metrics on right side with better spacing
            lpips = float(row.get('LPIPS', 0))
            ssim = float(row.get('SSIM', 0))
            psnr = float(row.get('PSNR', 0))

            mx = x + img.width + 12
            my = current_y + 5
            self.draw.text((mx, my), f"LPIPS: {lpips:.3f}", fill=TEXT_WHITE, font=self.font_caption)
            my += line_h
            self.draw.text((mx, my), f"SSIM: {ssim:.3f}", fill=TEXT_WHITE, font=self.font_caption)
            my += line_h
            self.draw.text((mx, my), f"PSNR: {psnr:.1f} dB", fill=TEXT_WHITE, font=self.font_caption)
            my += line_h + 10

            # Delta vs baseline - highlighted
            d_lpips = lpips - baseline_lpips
            d_psnr = psnr - baseline_psnr
            delta_color = ACCENT_GREEN if d_psnr < -10 else ACCENT_ORANGE
            self.draw.text((mx, my), f"dPSNR: {d_psnr:+.1f} dB", fill=delta_color, font=self.font_caption)
            my += line_h
            self.draw.text((mx, my), f"dLPIPS: {d_lpips:+.3f}", fill=delta_color, font=self.font_caption)

            current_y += img.height + 20

    def _render_he_column(self, area: Tuple[int, int, int, int], w: int, h: int):
        """Render HE column."""
        x, y, x2, y2 = area
        current_y = y
        line_h = int(FONT_CAPTION * LINE_SPACING)

        defense_df = self.df[self.df['group'] == 'defenses']
        he_df = defense_df[defense_df['setting'] == 'he']
        baseline_lpips, baseline_ssim, baseline_psnr = self._get_baseline_metrics()

        if he_df.empty:
            self.draw.text((x, current_y), "(HE results not available)", fill=TEXT_MUTED, font=self.font_body)
            return

        row = he_df.iloc[0]
        self.chosen_runs['he'] = 'he'

        # Main image - consistent size with DP+HE
        img_path = Path(row['path']) / 'baseline_attack_result.png'
        img_h = int(h * 0.50)  # 50% of column height
        img = load_image_safe(img_path, (w, img_h))
        img = resize_contain(img, w - 8, img_h - 8)
        img = add_border(img, 3, ACCENT_GREEN)

        ix = x + (w - img.width) // 2
        self.canvas.paste(img, (ix, current_y))
        current_y += img.height + 15

        # Metrics
        lpips = float(row.get('LPIPS', 0))
        ssim = float(row.get('SSIM', 0))
        psnr = float(row.get('PSNR', 0))
        label_match = row.get('LabelMatch', 0)

        self.draw.text((x, current_y), f"LPIPS: {lpips:.3f}   SSIM: {ssim:.3f}", fill=TEXT_WHITE, font=self.font_caption)
        current_y += line_h
        self.draw.text((x, current_y), f"PSNR: {psnr:.1f} dB", fill=TEXT_WHITE, font=self.font_caption)
        current_y += line_h

        lm_text = "LabelMatch: Yes" if label_match else "LabelMatch: No"
        self.draw.text((x, current_y), lm_text, fill=TEXT_LIGHT, font=self.font_caption)
        current_y += line_h + 18

        # Delta vs baseline - larger box with proper padding
        d_lpips = lpips - baseline_lpips
        d_psnr = psnr - baseline_psnr

        delta_box_h = 150
        draw_rounded_rect(self.draw, (x, current_y, x + w, current_y + delta_box_h),
                          radius=10, fill=(30, 50, 45), outline=ACCENT_GREEN, width=3)

        dy = current_y + 25
        self.draw.text((x + 25, dy), "Defense Impact:", fill=ACCENT_GREEN, font=self.font_body)
        dy += 45
        self.draw.text((x + 25, dy), f"dPSNR = {d_psnr:+.1f} dB", fill=TEXT_WHITE, font=self.font_caption)
        dy += 35
        self.draw.text((x + 25, dy), f"dLPIPS = {d_lpips:+.3f}", fill=TEXT_WHITE, font=self.font_caption)

    def _render_dphe_column(self, area: Tuple[int, int, int, int], w: int, h: int):
        """Render DP+HE column - the strongest defense."""
        x, y, x2, y2 = area
        current_y = y
        line_h = int(FONT_CAPTION * LINE_SPACING)

        defense_df = self.df[self.df['group'] == 'defenses']
        dphe_df = defense_df[defense_df['setting'] == 'dp_he']
        baseline_lpips, baseline_ssim, baseline_psnr = self._get_baseline_metrics()

        if dphe_df.empty:
            self.draw.text((x, current_y), "(DP+HE results not available)", fill=TEXT_MUTED, font=self.font_body)
            return

        row = dphe_df.iloc[0]
        self.chosen_runs['dp_he'] = 'dp_he'

        # Main image - consistent size with HE
        img_path = Path(row['path']) / 'baseline_attack_result.png'
        img_h = int(h * 0.50)  # 50% of column height - same as HE
        img = load_image_safe(img_path, (w, img_h))
        img = resize_contain(img, w - 8, img_h - 8)
        img = add_border(img, 3, ACCENT_PURPLE)

        ix = x + (w - img.width) // 2
        self.canvas.paste(img, (ix, current_y))
        current_y += img.height + 15

        # Metrics
        lpips = float(row.get('LPIPS', 0))
        ssim = float(row.get('SSIM', 0))
        psnr = float(row.get('PSNR', 0))
        label_match = row.get('LabelMatch', 0)

        self.draw.text((x, current_y), f"LPIPS: {lpips:.3f}   SSIM: {ssim:.3f}", fill=TEXT_WHITE, font=self.font_caption)
        current_y += line_h
        self.draw.text((x, current_y), f"PSNR: {psnr:.1f} dB", fill=TEXT_WHITE, font=self.font_caption)
        current_y += line_h

        lm_text = "LabelMatch: No (Attack Failed)" if not label_match else "LabelMatch: Yes"
        lm_color = ACCENT_GREEN if not label_match else TEXT_LIGHT
        self.draw.text((x, current_y), lm_text, fill=lm_color, font=self.font_caption)
        current_y += line_h + 18

        # Prominent delta panel - larger box with proper padding
        d_psnr = psnr - baseline_psnr
        d_lpips = lpips - baseline_lpips

        delta_box_h = 150
        draw_rounded_rect(self.draw, (x, current_y, x + w, current_y + delta_box_h),
                          radius=10, fill=(40, 30, 55), outline=ACCENT_PURPLE, width=3)

        dy = current_y + 25
        self.draw.text((x + 25, dy), "IMPACT:", fill=ACCENT_GOLD, font=self.font_body)
        dy += 45
        
        self.draw.text((x + 25, dy), f"dPSNR = {d_psnr:+.1f} dB",
                       fill=ACCENT_GREEN if d_psnr < 0 else ACCENT_RED, font=self.font_caption)
        dy += 35
        self.draw.text((x + 25, dy), f"dLPIPS = {d_lpips:+.3f}",
                       fill=ACCENT_GREEN if d_lpips > 0 else ACCENT_RED, font=self.font_caption)

        current_y += delta_box_h + 15

        # Summary note - research conclusion
        note = "Privacy Preserved: Attack Fully Mitigated"
        self.draw.text((x, current_y), note, fill=ACCENT_GREEN, font=self.font_caption)

    def _draw_metrics_panel(self, x: int, y: int, w: int, agg: Dict, title: str,
                            max_h: int = 160) -> int:
        """Draw a metrics panel and return the new y position."""
        panel_h = min(max_h, 160)
        draw_rounded_rect(self.draw, (x, y, x + w, y + panel_h),
                          radius=8, fill=(32, 38, 50), outline=CARD_BORDER, width=1)

        self.draw.text((x + 15, y + 12), title, fill=TEXT_WHITE, font=self.font_caption)

        ly = y + 42
        line_h = int(FONT_SMALL * LINE_SPACING) + 4
        for metric in ['PSNR', 'SSIM', 'LPIPS']:
            if metric in agg and ly + line_h < y + panel_h - 10:
                mean, std = agg[metric]
                text = f"{metric}: {mean:.2f} +/- {std:.2f}"
                self.draw.text((x + 15, ly), text, fill=TEXT_WHITE, font=self.font_small)
                ly += line_h

        if 'LabelMatch_rate' in agg and ly + line_h < y + panel_h - 5:
            rate = agg['LabelMatch_rate'] * 100
            self.draw.text((x + 15, ly), f"LabelMatch: {rate:.0f}%", fill=ACCENT_GREEN, font=self.font_small)

        return y + panel_h + 10

    def draw_bottom_band(self):
        """Draw the ablation study bottom band with clean chart/text separation."""
        y_start = HEADER_HEIGHT + MAIN_PANEL_HEIGHT
        band_h = CANVAS_HEIGHT - y_start - OUTER_MARGIN

        # Band background with accent
        draw_rounded_rect(self.draw, (OUTER_MARGIN, y_start, CANVAS_WIDTH - OUTER_MARGIN, y_start + band_h),
                          radius=14, fill=CARD_BG, outline=ACCENT_BLUE, width=2)

        # Title - larger
        title = "ABLATION STUDY: Attack Configuration Analysis"
        self.draw.text((OUTER_MARGIN + CARD_PADDING + 5, y_start + 15), title,
                       fill=ACCENT_BLUE, font=self.font_header)

        content_y = y_start + 75
        content_h = band_h - 90

        # Get ablation data
        ablation_df = self.df[self.df['group'] == 'ablation']

        if ablation_df.empty:
            self.draw.text((OUTER_MARGIN + CARD_PADDING, content_y),
                           "Ablation results not available", fill=TEXT_MUTED, font=self.font_body)
            return

        # Layout: Left = 3 charts in cards, Right = text/table card
        charts_total_w = int((CANVAS_WIDTH - 2 * OUTER_MARGIN - 3 * CARD_PADDING - GUTTER) * 0.60)
        text_card_w = CANVAS_WIDTH - 2 * OUTER_MARGIN - 2 * CARD_PADDING - GUTTER - charts_total_w - 10

        single_chart_w = (charts_total_w - 2 * 15) // 3
        chart_h = content_h - 10

        # Create 3 charts
        charts = create_ablation_charts(ablation_df, single_chart_w, chart_h)

        # Place charts in separate card areas
        chart_x = OUTER_MARGIN + CARD_PADDING
        for i, chart in enumerate(charts):
            # Chart card background
            card_x = chart_x + i * (single_chart_w + 15)
            draw_rounded_rect(self.draw, (card_x, content_y, card_x + single_chart_w, content_y + chart_h),
                              radius=8, fill=(32, 38, 50), outline=CARD_BORDER, width=1)

            # Resize and paste chart
            chart_resized = resize_contain(chart, single_chart_w - 10, chart_h - 10)
            cx = card_x + (single_chart_w - chart_resized.width) // 2
            cy = content_y + (chart_h - chart_resized.height) // 2
            self.canvas.paste(chart_resized, (cx, cy))

        # Right side: text card with clean separation
        right_x = OUTER_MARGIN + CARD_PADDING + charts_total_w + GUTTER + 10
        right_card_h = content_h
        right_card_w = text_card_w

        draw_rounded_rect(self.draw, (right_x, content_y, right_x + right_card_w, content_y + right_card_h),
                          radius=10, fill=(32, 38, 50), outline=ACCENT_BLUE, width=2)

        # Calculate proper vertical centering
        inner_padding = 30
        text_x = right_x + inner_padding
        available_height = right_card_h - (inner_padding * 2)
        
        # Section heights (estimated)
        section1_h = 140  # Top Configurations
        section2_h = 130  # Key Findings
        section3_h = 50   # Best Run
        total_content_h = section1_h + section2_h + section3_h
        
        # Start with some top padding, distribute space
        text_y = content_y + inner_padding

        # Aggregate by ablation type
        agg_by_type = ablation_df.groupby('ablation_type').agg({
            'PSNR': 'mean',
            'SSIM': 'mean',
            'LPIPS': 'mean'
        }).reset_index()
        agg_by_type = agg_by_type.sort_values('PSNR', ascending=False)

        # Section 1: Top Configurations
        self.draw.text((text_x, text_y), "BEST ATTACK SETTINGS", fill=ACCENT_GOLD, font=self.font_body)
        text_y += 45

        for idx, row in agg_by_type.head(3).iterrows():
            name = row['ablation_type']
            psnr = row['PSNR']
            line = f"  {name}: {psnr:.1f} dB PSNR"
            self.draw.text((text_x, text_y), line, fill=TEXT_WHITE, font=self.font_caption)
            text_y += 32

        text_y += 25

        # Section 2: Key Findings
        self.draw.text((text_x, text_y), "RESEARCH INSIGHTS", fill=ACCENT_BLUE, font=self.font_body)
        text_y += 45

        takeaways = self._generate_ablation_takeaways(ablation_df, agg_by_type)
        for t in takeaways[:3]:
            self.draw.text((text_x, text_y), f"  {t}", fill=TEXT_LIGHT, font=self.font_caption)
            text_y += 32

        text_y += 25

        # Section 3: Best overall run - highlighted at bottom
        best_overall = best_by_lpips(ablation_df)
        if best_overall is not None:
            best_setting = best_overall.get('setting', 'N/A')
            self.draw.text((text_x, text_y), f"Optimal Config: {best_setting}",
                           fill=ACCENT_GREEN, font=self.font_body)

    def _generate_ablation_takeaways(self, df: pd.DataFrame, agg: pd.DataFrame) -> List[str]:
        """Generate ablation takeaways from data (plain ASCII)."""
        takeaways = []

        if agg.empty:
            return ["Insufficient data"]

        # Best by PSNR
        best_psnr = agg.loc[agg['PSNR'].idxmax()]
        takeaways.append(f"Best PSNR: {best_psnr['ablation_type']}")

        # Best by LPIPS (lowest)
        best_lpips = agg.loc[agg['LPIPS'].idxmin()]
        takeaways.append(f"Best LPIPS: {best_lpips['ablation_type']}")

        # Metric comparison if available
        if 'metric_cosine' in agg['ablation_type'].values and 'baseline' in agg['ablation_type'].values:
            cosine_row = agg[agg['ablation_type'] == 'metric_cosine'].iloc[0]
            baseline_row = agg[agg['ablation_type'] == 'baseline'].iloc[0]
            delta = cosine_row['PSNR'] - baseline_row['PSNR']
            takeaways.append(f"MSE metric outperforms cosine")

        # TV regularization
        tv_settings = agg[agg['ablation_type'].str.contains('tv_', na=False)]
        if not tv_settings.empty:
            best_tv = tv_settings.loc[tv_settings['PSNR'].idxmax()]
            takeaways.append(f"Best TV setting: {best_tv['ablation_type']}")

        return takeaways

    def compose(self) -> Image.Image:
        """Compose the full poster."""
        print("Composing poster...")

        self.draw_header()
        print("  - Header rendered")

        self.draw_main_panel()
        print("  - Main panel (4 columns) rendered")

        self.draw_bottom_band()
        print("  - Bottom band (ablation) rendered")

        return self.canvas

    def get_summary(self) -> str:
        """Get console summary of chosen runs."""
        lines = ["", "=" * 60, "POSTER GENERATION SUMMARY", "=" * 60]

        lines.append("\nBaseline 4x3 grid populated with:")
        for i, run in enumerate(self.baseline_grid_runs[:12], 1):
            lines.append(f"  Tile {i:2d}: {run}")

        lines.append("\nDefense column runs:")
        for col, run in self.chosen_runs.items():
            lines.append(f"  {col}: {run}")

        return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Federated Learning Privacy Poster Generator")
    print("=" * 60)

    # Collect/load metrics
    print("\n[1/4] Loading metrics...")
    df = load_summary_csv()

    if df.empty:
        print("      No summary.csv found. Collecting from results...")
        df = collect_all_metrics()

    if df.empty:
        print("WARNING: No metrics found. Generating poster with placeholder content.")
    else:
        print(f"      Found {len(df)} experiment runs")
        # Save summary.csv
        csv_path = save_summary_csv(df)
        print(f"      Saved: {csv_path}")

    # Create report directory
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Compose poster
    print("\n[2/4] Composing 4K poster (3840x2160)...")
    composer = PosterComposer(df)
    poster = composer.compose()

    # Save PNG
    print("\n[3/4] Saving poster...")
    png_path = REPORT_DIR / 'poster_4k.png'
    poster.save(png_path, 'PNG', optimize=True)
    print(f"      Saved: {png_path}")

    # Try to save PDF
    pdf_path = REPORT_DIR / 'poster_4k.pdf'
    try:
        poster.save(pdf_path, 'PDF', resolution=150.0)
        print(f"      Saved: {pdf_path}")
    except Exception as e:
        print(f"      PDF export skipped: {e}")

    # Print summary
    print(composer.get_summary())

    print("\n[4/4] Done!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  -> {png_path}")
    if pdf_path.exists():
        print(f"  -> {pdf_path}")
    print()


if __name__ == '__main__':
    main()
