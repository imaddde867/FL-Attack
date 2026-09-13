#!/usr/bin/env python3
"""
make_dashboard.py
=================

Generates an interactive dashboard for the Federated Learning Gradient Inversion
project, providing drill-down access to every run, reconstruction, and metric
recorded in results/report/summary.csv.

Outputs (all under results/report/dashboard/):
  - index.html: interactive single-page dashboard (vanilla HTML/CSS/JS)
  - data.json: precomputed runs + aggregates + metadata
  - assets/images/: cropped Original/Recon/Diff strips (or placeholder)
  - assets/metrics/: copied metrics.txt files when available
  - assets/charts/: pre-rendered matplotlib charts
  - assets/montages/: montage thumbnails (copied or auto-generated)

Usage:
    python scripts/make_dashboard.py
"""

from __future__ import annotations

import json
import math
import re
import shutil
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: pillow is required. Install with: pip install pillow")
    sys.exit(1)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Helpers and data containers
# -----------------------------------------------------------------------------

METRIC_FIELDS = ["MSE", "PSNR", "SSIM", "LPIPS", "LabelMatch"]
VALID_GROUPS = ["showcase", "multi_client", "ablation", "defenses"]
PLACEHOLDER_COLOR = (250, 249, 246)  # matches PALETTE["paper"] = "#faf9f6"


def safe_float(value: object) -> Optional[float]:
    """Convert to float if possible."""
    if value is None:
        return None
    try:
        if isinstance(value, str) and not value.strip():
            return None
        num = float(value)
        if math.isnan(num):
            return None
        return num
    except Exception:
        return None


def slugify(text: str, default: str = "run") -> str:
    """Create filesystem-friendly slug."""
    if not text:
        text = default
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or default


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ranking_tuple(metrics: Dict[str, Optional[float]]) -> Tuple[float, float, float]:
    """Return key for best-sort (LPIPS asc, SSIM desc, PSNR desc)."""
    lpips = metrics.get("LPIPS")
    ssim = metrics.get("SSIM")
    psnr = metrics.get("PSNR")
    lpips_val = lpips if lpips is not None else float("inf")
    ssim_val = -(ssim if ssim is not None else 0.0)
    psnr_val = -(psnr if psnr is not None else 0.0)
    return (lpips_val, ssim_val, psnr_val)


def safe_relative(path: Path, root: Path) -> str:
    """Return POSIX path relative to root when possible."""
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


def infer_client(text_parts: List[str]) -> str:
    """Infer client label from strings."""
    for part in text_parts:
        if not part:
            continue
        match = re.search(r"(c\d+)", str(part).lower())
        if match:
            return match.group(1)
        match = re.search(r"client[_\-]?(\d+)", str(part).lower())
        if match:
            return f"c{match.group(1)}"
    return "global"


def load_metrics_file(path: Path) -> Dict[str, float]:
    """Parse key=value pairs in metrics.txt (best-effort)."""
    metrics: Dict[str, float] = {}
    try:
        for line in path.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
            elif "=" in line:
                key, value = line.split("=", 1)
            else:
                continue
            key = key.strip()
            value = value.strip()
            num = safe_float(value)
            if num is None:
                continue
            metrics[key] = num
    except Exception:
        pass
    return metrics


# Single source of truth for chart/CSS color — mirrored by hand in
# scripts/templates/dashboard.html's :root block. Update both when changing a color.
PALETTE = {
    "paper": "#faf9f6",
    "panel": "#ffffff",
    "ink": "#1a1a1a",
    "muted": "#5c5a52",
    "border": "#ddd9d0",
    "navy": "#1d3557",
    "teal": "#457b9d",
    "rust": "#bc4b2c",
    "olive": "#6a7f3f",
    "gray": "#8d8a80",
}


@dataclass
class RunEntry:
    run_id: str
    group: str
    setting: str
    method: str
    client: str
    source_dir: str
    metrics: Dict[str, Optional[float]]
    image_path: str
    metrics_path: Optional[str]
    search_text: str


# -----------------------------------------------------------------------------
# Dashboard builder
# -----------------------------------------------------------------------------


class DashboardBuilder:
    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.results_dir = self.root / "results"
        self.report_dir = self.results_dir / "report"
        self.summary_csv = self.report_dir / "summary.csv"

        self.output_dir = self.report_dir / "dashboard"
        self.assets_dir = self.output_dir / "assets"
        self.images_dir = self.assets_dir / "images"
        self.metrics_dir = self.assets_dir / "metrics"
        self.charts_dir = self.assets_dir / "charts"
        self.montage_dir = self.assets_dir / "montages"

        self.placeholder_path = self.images_dir / "placeholder.png"

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.prepare_output_dir()
        df = self.load_summary()
        runs, df_augmented = self.process_runs(df)
        aggregates = self.compute_aggregates(df_augmented)
        montages = self.prepare_montages(runs)
        charts = self.generate_charts(df_augmented)
        figures = self.copy_static_figures()

        baseline_index = self.compute_baselines(runs)
        baselines_map = baseline_index.get("by_client", {})
        best_overall_id = self.get_best_run_id(runs)
        best_per_setting = self.get_best_by_method(runs)
        best_baseline_id = baseline_index.get("global")

        key_finding = self.derive_key_finding(df_augmented, baselines_map)
        meta = {
            "build_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "total_runs": len(runs),
            "groups": {
                group: int((df_augmented["group"] == group).sum())
                for group in sorted(df_augmented["group"].dropna().unique())
            },
            "key_finding": key_finding,
            "build_source": safe_relative(self.summary_csv, self.root),
        }

        filter_values = {
            "groups": sorted(df_augmented["group"].dropna().unique().tolist()),
            "methods": sorted(df_augmented["method"].dropna().unique().tolist()),
            "clients": sorted(df_augmented["client"].dropna().unique().tolist()),
        }

        data_blob = {
            "meta": meta,
            "runs": [run.__dict__ for run in runs],
            "aggregates": aggregates,
            "baselines_by_client": baselines_map,
            "baseline_index": baseline_index,
            "best_overall_id": best_overall_id,
            "best_baseline_id": best_baseline_id,
            "best_per_setting": best_per_setting,
            "filter_values": filter_values,
            "charts": charts,
            "montages": montages,
            "figures": figures,
            "placeholder_image": safe_relative(self.placeholder_path, self.output_dir),
        }

        self.write_json(self.output_dir / "data.json", data_blob)
        self.write_index_html()

        self.print_summary(meta, charts)
        self.print_run_commands()

    # ------------------------------------------------------------------
    def prepare_output_dir(self) -> None:
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        ensure_directory(self.images_dir)
        ensure_directory(self.metrics_dir)
        ensure_directory(self.charts_dir)
        ensure_directory(self.montage_dir)
        self.create_placeholder_image()

    # ------------------------------------------------------------------
    def load_summary(self) -> pd.DataFrame:
        if not self.summary_csv.exists():
            print(f"ERROR: Summary file not found at {self.summary_csv}")
            sys.exit(1)

        df = pd.read_csv(self.summary_csv)
        expected_cols = set(["group", "setting", "path"])
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(f"summary.csv missing columns: {missing}")
        return df

    # ------------------------------------------------------------------
    def process_runs(self, df: pd.DataFrame) -> Tuple[List[RunEntry], pd.DataFrame]:
        runs: List[RunEntry] = []
        augmented_rows: List[Dict[str, object]] = []
        slug_counts: Dict[str, int] = defaultdict(int)
        stats = {"copied": 0, "missing_images": 0}

        for idx, row in df.iterrows():
            group = str(row.get("group", "")).strip().lower() or "unspecified"
            setting_raw = str(row.get("setting", "")).strip()
            method = setting_raw.split("/")[0] if setting_raw else "unknown"
            method = method.strip() or "unknown"
            path_raw = str(row.get("path", "")).strip()
            source_path = self.root / path_raw if path_raw else None
            client = infer_client(
                [
                    row.get("client"),
                    setting_raw,
                    path_raw,
                ]
            )

            metrics = {field: safe_float(row.get(field)) for field in METRIC_FIELDS}
            slug_seed = f"{group}-{setting_raw or method}-{client}"
            slug = slugify(slug_seed)
            slug_counts[slug] += 1
            if slug_counts[slug] > 1:
                slug = f"{slug}-{slug_counts[slug]}"

            image_rel = self.copy_image_for_run(source_path, slug)
            if not image_rel:
                stats["missing_images"] += 1
                image_rel = safe_relative(self.placeholder_path, self.output_dir)
            else:
                stats["copied"] += 1

            metrics_path = self.copy_metrics_for_run(source_path, slug)

            run_id = f"run_{idx:04d}_{slug}"
            if path_raw:
                source_dir = safe_relative(self.root / path_raw, self.root)
            else:
                source_dir = "Not available"

            search_text = " ".join(
                filter(
                    None,
                    [
                        group,
                        method,
                        client,
                        setting_raw,
                        source_dir,
                        str(run_id),
                    ],
                )
            ).lower()

            run = RunEntry(
                run_id=run_id,
                group=group,
                setting=setting_raw or method,
                method=method,
                client=client,
                source_dir=source_dir,
                metrics=metrics,
                image_path=image_rel,
                metrics_path=metrics_path,
                search_text=search_text,
            )
            runs.append(run)

            row_dict: Dict[str, object] = {
                "run_id": run_id,
                "group": group,
                "setting": run.setting,
                "method": method,
                "client": client,
            }
            for field in METRIC_FIELDS:
                row_dict[field] = metrics.get(field)
            augmented_rows.append(row_dict)

        if not runs:
            print("WARNING: No runs found in summary.csv")

        df_augmented = pd.DataFrame(augmented_rows)
        print(
            f"[INFO] Processed {len(runs)} runs "
            f"({stats['copied']} images, {stats['missing_images']} placeholders)."
        )
        return runs, df_augmented

    # ------------------------------------------------------------------
    def _find_composite(self, source_dir: Optional[Path]) -> Optional[Path]:
        """Locate the raw baseline_attack_result.png composite for a run's
        source directory (newest match if more than one). Used both to build
        the per-run cropped asset and, separately, by the montage builder,
        which needs the untouched composite rather than the already-cropped
        per-run strip."""
        if not source_dir or not source_dir.exists():
            return None
        direct = source_dir / "baseline_attack_result.png"
        if direct.exists():
            return direct
        files = list(source_dir.rglob("baseline_attack_result.png"))
        files.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
        return files[0] if files else None

    def copy_image_for_run(self, source_dir: Optional[Path], slug: str) -> Optional[str]:
        src = self._find_composite(source_dir)
        if src is None:
            return None
        dest = self.images_dir / f"{slug}.png"
        try:
            bands = self._detect_bands(src)
        except Exception:
            bands = []
        if len(bands) >= 2:
            self._save_band_strip(bands[:3], dest)
        else:
            shutil.copy2(src, dest)
        return safe_relative(dest, self.output_dir)

    # ------------------------------------------------------------------
    def copy_metrics_for_run(
        self, source_dir: Optional[Path], slug: str
    ) -> Optional[str]:
        if not source_dir or not source_dir.exists():
            return None
        direct = source_dir / "metrics.txt"
        if direct.exists():
            candidates = [direct]
        else:
            files = list(source_dir.rglob("metrics.txt"))
            files.sort(
                key=lambda path: path.stat().st_mtime if path.exists() else 0,
                reverse=True,
            )
            candidates = files
        if not candidates:
            return None
        src = candidates[0]
        dest = self.metrics_dir / f"{slug}_metrics.txt"
        shutil.copy2(src, dest)
        return safe_relative(dest, self.output_dir)

    # ------------------------------------------------------------------
    def compute_baselines(self, runs: List[RunEntry]) -> Dict[str, object]:
        def is_baseline(run: RunEntry) -> bool:
            label = run.method.lower()
            setting = run.setting.lower()
            if "baseline" in label or "baseline" in setting:
                return True
            if run.group == "multi_client":
                return label.startswith("bmk") or setting.startswith("bmk")
            return False

        by_group_clients: Dict[str, Dict[str, RunEntry]] = defaultdict(dict)
        by_group_best: Dict[str, RunEntry] = {}
        by_client: Dict[str, RunEntry] = {}
        overall_best: Optional[RunEntry] = None

        for run in runs:
            if not is_baseline(run):
                continue
            group = run.group
            client = run.client or "global"

            current = by_group_clients[group].get(client)
            if not current or ranking_tuple(run.metrics) < ranking_tuple(current.metrics):
                by_group_clients[group][client] = run

            group_best = by_group_best.get(group)
            if not group_best or ranking_tuple(run.metrics) < ranking_tuple(group_best.metrics):
                by_group_best[group] = run

            client_best = by_client.get(client)
            if not client_best or ranking_tuple(run.metrics) < ranking_tuple(client_best.metrics):
                by_client[client] = run

            if not overall_best or ranking_tuple(run.metrics) < ranking_tuple(overall_best.metrics):
                overall_best = run

        baseline_index = {
            "global": overall_best.run_id if overall_best else None,
            "by_client": {client: run.run_id for client, run in by_client.items()},
            "by_group": {},
        }

        if overall_best:
            baseline_index["by_client"].setdefault("global", overall_best.run_id)

        for group, clients in by_group_clients.items():
            baseline_index["by_group"][group] = {
                "global": by_group_best[group].run_id if group in by_group_best else None,
                "clients": {client: run.run_id for client, run in clients.items()},
            }

        return baseline_index

    # ------------------------------------------------------------------
    def get_best_run_id(self, runs: List[RunEntry]) -> Optional[str]:
        if not runs:
            return None
        best = min(runs, key=lambda run: ranking_tuple(run.metrics))
        return best.run_id

    # ------------------------------------------------------------------
    def get_best_by_method(self, runs: List[RunEntry]) -> Dict[str, str]:
        best: Dict[str, RunEntry] = {}
        for run in runs:
            method = run.method
            current = best.get(method)
            if not current or ranking_tuple(run.metrics) < ranking_tuple(
                current.metrics
            ):
                best[method] = run
        return {method: run.run_id for method, run in best.items()}

    # ------------------------------------------------------------------
    def compute_aggregates(self, df: pd.DataFrame) -> Dict[str, object]:
        aggregates: Dict[str, object] = {}
        if df.empty:
            return {"by_group_setting": [], "by_setting": [], "ablations": []}

        def summarize(grouped: pd.DataFrame, group_cols: List[str]) -> List[Dict]:
            rows: List[Dict] = []
            for keys, sub in grouped:
                record: Dict[str, object] = {}
                if isinstance(keys, tuple):
                    for name, value in zip(group_cols, keys):
                        record[name] = value
                else:
                    record[group_cols[0]] = keys
                record["count"] = int(len(sub))
                for field in METRIC_FIELDS:
                    values = sub[field].dropna().astype(float) if field in sub else []
                    if len(values) == 0:
                        continue
                    record[field] = {
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)),
                    }
                rows.append(record)
            return rows

        if {"group", "method"}.issubset(df.columns):
            aggregates["by_group_setting"] = summarize(
                df.groupby(["group", "method"]), ["group", "method"]
            )
        else:
            aggregates["by_group_setting"] = []

        if "method" in df.columns:
            aggregates["by_setting"] = summarize(df.groupby(["method"]), ["method"])
        else:
            aggregates["by_setting"] = []

        ablations_rows: List[Dict[str, object]] = []
        if "method" in df.columns:
            ablation_df = df[df["group"].str.contains("ablation", na=False)]
            if not ablation_df.empty:
                for method, sub in ablation_df.groupby("method"):
                    best_row = sub.sort_values(
                        by=["LPIPS", "SSIM", "PSNR"],
                        ascending=[True, False, False],
                        na_position="last",
                    ).head(1)
                    if best_row.empty:
                        continue
                    row = best_row.iloc[0]
                    ablations_rows.append(
                        {
                            "method": method,
                            "client": row.get("client", ""),
                            "PSNR": safe_float(row.get("PSNR")),
                            "SSIM": safe_float(row.get("SSIM")),
                            "LPIPS": safe_float(row.get("LPIPS")),
                        }
                    )
        aggregates["ablations"] = ablations_rows
        return aggregates

    # ------------------------------------------------------------------
    def derive_key_finding(
        self, df: pd.DataFrame, baselines: Dict[str, str]
    ) -> str:
        if df.empty:
            return "No runs parsed."
        defenses = df[df["group"] == "defenses"]
        if defenses.empty:
            return "No defense runs available."

        baseline = defenses[defenses["method"].str.contains("baseline", na=False)]
        dp_he = defenses[
            defenses["method"].str.contains("dp_he", na=False)
            | defenses["method"].str.contains("dphe", na=False)
        ]
        if not baseline.empty and not dp_he.empty:
            base_psnr = baseline["PSNR"].mean()
            dp_psnr = dp_he["PSNR"].mean()
            delta = base_psnr - dp_psnr
            if not math.isnan(delta):
                return (
                    f"DP+HE reduces PSNR by {delta:.1f} dB relative to baseline "
                    f"({base_psnr:.1f} -> {dp_psnr:.1f})."
                )
        if not baseline.empty:
            base_lpips = baseline["LPIPS"].mean()
            best_def = defenses.sort_values(
                by=["LPIPS", "SSIM"], ascending=[True, False]
            ).head(1)
            if not best_def.empty:
                row = best_def.iloc[0]
                delta = safe_float(row.get("LPIPS"))
                if base_lpips is not None and delta is not None:
                    diff = delta - base_lpips
                    return (
                        f"Best defense ({row['method']}) shifts LPIPS by {diff:+.3f} "
                        f"vs baseline ({base_lpips:.3f})."
                    )
        return "Baseline comparisons unavailable."

    # ------------------------------------------------------------------
    def prepare_montages(self, runs: List[RunEntry]) -> List[Dict[str, str]]:
        montages: List[Dict[str, str]] = []
        source_dir = self.report_dir / "montages"
        if source_dir.exists():
            for png in sorted(source_dir.glob("*.png")):
                dest = self.montage_dir / png.name
                shutil.copy2(png, dest)
                montages.append(
                    {"name": png.stem.replace("_", " ").title(), "path": safe_relative(dest, self.output_dir)}
                )
        if not montages:
            # Auto-generate montage from top runs
            best_runs = sorted(runs, key=lambda r: ranking_tuple(r.metrics))[:6]
            image_paths = [
                self._find_composite(self.root / run.source_dir) for run in best_runs if run.source_dir
            ]
            image_paths = [path for path in image_paths if path is not None]
            if image_paths:
                auto_path = self.montage_dir / "top_ranked.png"
                self.build_montage(image_paths, auto_path)
                montages.append(
                    {"name": "Top reconstructions", "path": safe_relative(auto_path, self.output_dir)}
                )
        return montages

    # ------------------------------------------------------------------
    def copy_static_figures(self) -> Dict[str, str]:
        """Copy pre-generated, non-per-run figures (e.g. the DP noise-scaling proof)
        into the dashboard's asset tree so the template can reference a stable path."""
        figures_src = self.report_dir / "figures"
        figures_dest = self.assets_dir / "figures"
        ensure_directory(figures_dest)
        figures: Dict[str, str] = {}
        for name, filename in [("dp_noise_scaling", "dp_noise_scaling.png")]:
            src = figures_src / filename
            if src.exists():
                dest = figures_dest / filename
                shutil.copy2(src, dest)
                figures[name] = safe_relative(dest, self.output_dir)
        return figures

    # ------------------------------------------------------------------
    def _detect_bands(
        self, composite_path: Path, bg_thresh: int = 235, min_band: int = 30
    ) -> List[Image.Image]:
        """Crop each vertically-stacked panel out of a baseline_attack_result.png
        composite (Original/Recon/[|Diff|], each preceded by a matplotlib title
        on a white band). Detects band boundaries per-image via row/column
        background-brightness thresholding rather than assuming a fixed layout.
        Returns the cropped panels top-to-bottom (empty list if none detected)."""
        with Image.open(composite_path) as img:
            img = img.convert("RGB")
            arr = np.array(img)
            row_mean = arr.mean(axis=(1, 2))
            is_content = row_mean < bg_thresh
            bands: List[Tuple[int, int]] = []
            start: Optional[int] = None
            for i, content in enumerate(is_content):
                if content and start is None:
                    start = i
                elif not content and start is not None:
                    if i - start >= min_band:
                        bands.append((start, i))
                    start = None
            if start is not None and len(is_content) - start >= min_band:
                bands.append((start, len(is_content)))

            def crop_band(row_range: Tuple[int, int]) -> Image.Image:
                r0, r1 = row_range
                col_mean = arr[r0:r1].mean(axis=(0, 2))
                content_cols = np.where(col_mean < bg_thresh)[0]
                if len(content_cols) == 0:
                    c0, c1 = 0, arr.shape[1]
                else:
                    c0, c1 = int(content_cols[0]), int(content_cols[-1]) + 1
                return img.crop((c0, r0, c1, r1)).copy()

            return [crop_band(b) for b in bands]

    def _extract_original_recon(
        self, composite_path: Path, bg_thresh: int = 235, min_band: int = 30
    ) -> Optional[Tuple[Image.Image, Image.Image]]:
        """Original+Recon pair only (for the montage grid). Returns None if
        fewer than two bands are found. Callers must not substitute a Recon
        crop where an Original is expected."""
        bands = self._detect_bands(composite_path, bg_thresh, min_band)
        if len(bands) < 2:
            return None
        return bands[0], bands[1]

    def _save_band_strip(self, bands: List[Image.Image], dest: Path, thumb: int = 360, gap: int = 12) -> None:
        """Lay out 2-3 panel crops (Original/Recon/[Diff]) side by side as one
        landscape strip. Used for the per-run detail image so the run viewer
        gets a wide-and-short asset instead of the tall stacked composite."""
        labels = ["Original", "Recon", "Diff"][: len(bands)]
        thumbs = []
        for band in bands:
            t = band.copy()
            t.thumbnail((thumb, thumb))
            thumbs.append(t)
        label_h = 26
        cell_w = max(t.width for t in thumbs)
        row_h = max(t.height for t in thumbs)
        width = cell_w * len(thumbs) + gap * (len(thumbs) - 1)
        height = label_h + row_h
        canvas = Image.new("RGB", (width, height), color=(250, 249, 246))  # PALETTE["paper"]
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        ink = (26, 26, 26)  # PALETTE["ink"]
        x = 0
        for t, label in zip(thumbs, labels):
            cx = x + (cell_w - t.width) // 2
            cy = label_h + (row_h - t.height) // 2
            canvas.paste(t, (cx, cy))
            draw.text((x + cell_w // 2, 4), label, fill=ink, font=font, anchor="mt")
            x += cell_w + gap
        ensure_directory(dest.parent)
        canvas.save(dest)

    def build_montage(self, image_paths: List[Path], dest: Path) -> None:
        pairs: List[Tuple[Image.Image, Image.Image]] = []
        for img_path in image_paths:
            try:
                extracted = self._extract_original_recon(img_path)
            except Exception:
                continue
            if extracted is not None:
                pairs.append(extracted)
        if not pairs:
            return

        cols = 3
        thumb = 160
        gap = 8
        label_h = 22
        cell_w = thumb * 2 + gap
        cell_h = thumb + label_h
        rows = math.ceil(len(pairs) / cols)
        canvas = Image.new(
            "RGB",
            (cols * cell_w, rows * cell_h),
            color=(250, 249, 246),  # PALETTE["paper"] — light background
        )
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("Arial.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
        ink = (26, 26, 26)  # PALETTE["ink"]

        for idx, (original, recon) in enumerate(pairs):
            col, row = idx % cols, idx // cols
            cell_x, cell_y = col * cell_w, row * cell_h

            original_thumb = original.copy()
            original_thumb.thumbnail((thumb, thumb))
            recon_thumb = recon.copy()
            recon_thumb.thumbnail((thumb, thumb))

            ox = cell_x + (thumb - original_thumb.width) // 2
            oy = cell_y + label_h + (thumb - original_thumb.height) // 2
            rx = cell_x + thumb + gap + (thumb - recon_thumb.width) // 2
            ry = cell_y + label_h + (thumb - recon_thumb.height) // 2
            canvas.paste(original_thumb, (ox, oy))
            canvas.paste(recon_thumb, (rx, ry))

            draw.text((cell_x + thumb // 2, cell_y + 4), "Original", fill=ink, font=font, anchor="mt")
            draw.text(
                (cell_x + thumb + gap + thumb // 2, cell_y + 4), "Recon", fill=ink, font=font, anchor="mt"
            )

        ensure_directory(dest.parent)
        canvas.save(dest)

    # ------------------------------------------------------------------
    def _setup_chart_style(self):
        """Configure matplotlib for the light academic PALETTE."""
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': PALETTE["paper"],
            'axes.facecolor': PALETTE["panel"],
            'axes.edgecolor': PALETTE["border"],
            'axes.labelcolor': PALETTE["ink"],
            'text.color': PALETTE["ink"],
            'xtick.color': PALETTE["muted"],
            'ytick.color': PALETTE["muted"],
            'grid.color': PALETTE["border"],
            'legend.facecolor': PALETTE["panel"],
            'legend.edgecolor': PALETTE["border"],
            'figure.edgecolor': PALETTE["border"],
            'savefig.facecolor': PALETTE["paper"],
        })

    # ------------------------------------------------------------------
    def generate_charts(self, df: pd.DataFrame) -> Dict[str, str]:
        self._setup_chart_style()
        chart_map: Dict[str, str] = {}
        chart_funcs = {
            "defenses_grouped_bars": self.chart_defenses_grouped_bars,
            "defenses_scatter_psnr_vs_lpips": self.chart_defenses_scatter,
            "multiclient_boxplots": self.chart_multiclient_boxplots,
            "ablation_psnr": self.chart_ablation_psnr,
            "ablation_ssim": self.chart_ablation_ssim,
            "ablation_lpips": self.chart_ablation_lpips,
        }

        generated = []
        for name, func in chart_funcs.items():
            dest = self.charts_dir / f"{name}.png"
            func(df, dest)
            chart_map[name] = safe_relative(dest, self.output_dir)
            generated.append(name)
        print(f"[INFO] Generated charts: {', '.join(generated)}")
        return chart_map

    # ------------------------------------------------------------------
    def chart_defenses_grouped_bars(self, df: pd.DataFrame, dest: Path) -> None:
        subset = df[df["group"] == "defenses"]
        if subset.empty:
            self.save_empty_chart(dest, "Defenses comparison", "No defense runs.")
            return
        metrics = subset.groupby("method")[["PSNR", "SSIM", "LPIPS", "LabelMatch"]].agg(
            ["mean", "std"]
        )
        methods = metrics.index.tolist()
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(methods))
        width = 0.2
        psnr_means = metrics[("PSNR", "mean")].tolist()
        psnr_std = metrics[("PSNR", "std")].fillna(0).tolist()
        ssim_means = metrics[("SSIM", "mean")].tolist()
        ssim_std = metrics[("SSIM", "std")].fillna(0).tolist()
        lpips_means = metrics[("LPIPS", "mean")].tolist()
        lpips_std = metrics[("LPIPS", "std")].fillna(0).tolist()
        ax.bar([i - width for i in x], psnr_means, width, yerr=psnr_std, label="PSNR", color=PALETTE["navy"], capsize=3)
        ax.bar(x, ssim_means, width, yerr=ssim_std, label="SSIM", color=PALETTE["teal"], capsize=3)
        ax.bar(
            [i + width for i in x],
            lpips_means,
            width,
            yerr=lpips_std,
            label="LPIPS",
            color=PALETTE["rust"],
            capsize=3,
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Defense Mechanisms: Mean Metrics Comparison", fontweight="bold")
        ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["border"])
        ax.grid(alpha=0.3, axis="y")

        ax2 = ax.twinx()
        label_match = metrics[("LabelMatch", "mean")].tolist()
        ax2.plot(x, label_match, color=PALETTE["olive"], marker="o", label="LabelMatch", linewidth=2)
        ax2.set_ylabel("Label Match Rate")
        fig.tight_layout()
        fig.savefig(dest, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    def chart_defenses_scatter(self, df: pd.DataFrame, dest: Path) -> None:
        subset = df[df["group"] == "defenses"]
        if subset.empty:
            self.save_empty_chart(dest, "Defense scatter", "No defense runs.")
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(
            subset["LPIPS"], subset["PSNR"], c=subset["SSIM"], cmap="viridis", s=100,
            edgecolors=PALETTE["border"], linewidths=1
        )
        ax.set_xlabel("LPIPS (lower = better)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("Defense Analysis: PSNR vs LPIPS", fontweight="bold")
        ax.grid(alpha=0.3)
        cbar = fig.colorbar(scatter, label="SSIM")
        cbar.ax.yaxis.set_tick_params(color=PALETTE["muted"])
        fig.tight_layout()
        fig.savefig(dest, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    def chart_multiclient_boxplots(self, df: pd.DataFrame, dest: Path) -> None:
        subset = df[df["group"] == "multi_client"]
        if subset.empty:
            self.save_empty_chart(dest, "Multi-client distributions", "No multi-client runs.")
            return
        metrics = ["PSNR", "SSIM", "LPIPS"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        clients = sorted(subset["client"].unique())
        for idx, metric in enumerate(metrics):
            axes[idx].boxplot(
                [
                    subset[subset["client"] == client][metric].dropna()
                    for client in clients
                ]
            )
            axes[idx].set_title(metric)
            axes[idx].set_xticklabels(clients, rotation=45)
        fig.suptitle("Multi-client metric distributions")
        fig.tight_layout()
        fig.savefig(dest, dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    def _chart_ablation_single(
        self, df: pd.DataFrame, dest: Path, metric: str, title: str, color: str
    ) -> None:
        subset = df[df["group"] == "ablation"]
        if subset.empty:
            self.save_empty_chart(dest, title, "No ablation runs.")
            return
        summary = subset.groupby("method")[[metric]].agg(["mean", "std"])
        summary = summary.sort_values((metric, "mean"), ascending=(metric == "LPIPS"))
        methods = summary.index.tolist()
        means = summary[(metric, "mean")].tolist()
        std = summary[(metric, "std")].fillna(0).tolist()
        fig, ax = plt.subplots(figsize=(6, 4))
        x = range(len(methods))
        ax.bar(x, means, yerr=std, capsize=4, color=color)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(methods, rotation=40, ha="right")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(dest, dpi=150)
        plt.close(fig)

    def chart_ablation_psnr(self, df: pd.DataFrame, dest: Path) -> None:
        self._chart_ablation_single(df, dest, "PSNR", "Ablation: PSNR (higher = better)", PALETTE["navy"])

    def chart_ablation_ssim(self, df: pd.DataFrame, dest: Path) -> None:
        self._chart_ablation_single(df, dest, "SSIM", "Ablation: SSIM (higher = better)", PALETTE["teal"])

    def chart_ablation_lpips(self, df: pd.DataFrame, dest: Path) -> None:
        self._chart_ablation_single(df, dest, "LPIPS", "Ablation: LPIPS (lower = better)", PALETTE["rust"])

    # ------------------------------------------------------------------
    def save_empty_chart(self, dest: Path, title: str, message: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.6,
            title,
            ha="center",
            va="center",
            color=PALETTE["ink"],  # #1a1a1a, dark text
            fontsize=14,
        )
        ax.text(
            0.5,
            0.4,
            message,
            ha="center",
            va="center",
            color=PALETTE["muted"],  # #5c5a52, muted text
            fontsize=10,
        )
        fig.savefig(dest, dpi=120)
        plt.close(fig)

    # ------------------------------------------------------------------
    def create_placeholder_image(self) -> None:
        ensure_directory(self.placeholder_path.parent)
        width, height = 960, 600
        image = Image.new("RGB", (width, height), color=PLACEHOLDER_COLOR)
        draw = ImageDraw.Draw(image)
        text = "No reconstruction available"
        try:
            font = ImageFont.truetype("Arial.ttf", 32)
        except Exception:
            font = ImageFont.load_default()
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w, text_h = draw.textsize(text, font=font)
        draw.rectangle(
            [(20, 20), (width - 20, height - 20)],
            outline=(80, 90, 120),
            width=3,
        )
        draw.text(
            ((width - text_w) / 2, (height - text_h) / 2),
            text,
            fill=(220, 230, 245),
            font=font,
        )
        image.save(self.placeholder_path)

    # ------------------------------------------------------------------
    def write_json(self, path: Path, payload: Dict[str, object]) -> None:
        ensure_directory(path.parent)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        print(f"[INFO] Wrote data to {safe_relative(path, self.root)}")

    # ------------------------------------------------------------------
    def write_index_html(self) -> None:
        template_path = self.root / "scripts" / "templates" / "dashboard.html"
        html_content = template_path.read_text(encoding="utf-8").strip()
        html_path = self.output_dir / "index.html"
        html_path.write_text(html_content, encoding="utf-8")
        print(f"[INFO] Wrote dashboard HTML to {safe_relative(html_path, self.root)}")

    # ------------------------------------------------------------------
    def print_summary(self, meta: Dict[str, object], charts: Dict[str, str]) -> None:
        print("\nDashboard build summary:")
        print(f"  Build date: {meta['build_time']}")
        print(f"  Total runs parsed: {meta['total_runs']}")
        if meta["groups"]:
            print("  Runs by group:")
            for group, count in meta["groups"].items():
                print(f"    - {group}: {count}")
        print("  Charts generated:")
        for name, path in charts.items():
            print(f"    - {name}: {path}")
        print(f"  Dashboard saved to: {safe_relative(self.output_dir, self.root)}")

    # ------------------------------------------------------------------
    def print_run_commands(self) -> None:
        print("\nCommands to view the dashboard locally:")
        print("  python scripts/make_dashboard.py")
        print("  python -m http.server -d results/report/dashboard 8000")
        print("  open http://localhost:8000")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    builder = DashboardBuilder()
    builder.run()


if __name__ == "__main__":
    main()
