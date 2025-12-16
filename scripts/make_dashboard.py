#!/usr/bin/env python3
"""
make_dashboard.py
=================

Generates an interactive dashboard for the Federated Learning Gradient Inversion
project. The dashboard complements the 4K poster by providing drill-down access
to every run, reconstruction, and metric recorded in results/report/summary.csv.

Outputs (all under results/report/dashboard/):
  - index.html: interactive single-page dashboard (vanilla HTML/CSS/JS)
  - data.json: precomputed runs + aggregates + metadata
  - assets/images/: copied recon images (baseline_attack_result.png or placeholder)
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
PLACEHOLDER_COLOR = (18, 22, 32)


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


def apply_dark_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams["figure.facecolor"] = "#121621"
    plt.rcParams["axes.facecolor"] = "#1c2331"
    plt.rcParams["savefig.facecolor"] = "#121621"
    plt.rcParams["axes.edgecolor"] = "#3b4252"
    plt.rcParams["axes.labelcolor"] = "#f5f6fa"
    plt.rcParams["xtick.color"] = "#d8dee9"
    plt.rcParams["ytick.color"] = "#d8dee9"
    plt.rcParams["grid.color"] = "#2e3440"
    plt.rcParams["font.size"] = 11


apply_dark_style()


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

        baselines_map = self.compute_baselines(runs)
        best_overall_id = self.get_best_run_id(runs)
        best_per_setting = self.get_best_by_method(runs)
        best_baseline_id = baselines_map.get("global")

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
            "best_overall_id": best_overall_id,
            "best_baseline_id": best_baseline_id,
            "best_per_setting": best_per_setting,
            "filter_values": filter_values,
            "charts": charts,
            "montages": montages,
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
    def copy_image_for_run(self, source_dir: Optional[Path], slug: str) -> Optional[str]:
        if not source_dir or not source_dir.exists():
            return None
        candidates = sorted(source_dir.rglob("baseline_attack_result.png"))
        if not candidates:
            return None
        src = candidates[0]
        dest = self.images_dir / f"{slug}.png"
        shutil.copy2(src, dest)
        return safe_relative(dest, self.output_dir)

    # ------------------------------------------------------------------
    def copy_metrics_for_run(
        self, source_dir: Optional[Path], slug: str
    ) -> Optional[str]:
        if not source_dir or not source_dir.exists():
            return None
        candidates = sorted(source_dir.rglob("metrics.txt"))
        if not candidates:
            return None
        src = candidates[0]
        dest = self.metrics_dir / f"{slug}_metrics.txt"
        shutil.copy2(src, dest)
        return safe_relative(dest, self.output_dir)

    # ------------------------------------------------------------------
    def compute_baselines(self, runs: List[RunEntry]) -> Dict[str, str]:
        baselines: Dict[str, RunEntry] = {}
        for run in runs:
            label = run.setting.lower()
            if "baseline" not in label:
                continue
            client = run.client or "global"
            current = baselines.get(client)
            if not current or ranking_tuple(run.metrics) < ranking_tuple(
                current.metrics
            ):
                baselines[client] = run
        if baselines:
            # Determine best global baseline
            all_baseline_runs = list(baselines.values())
            all_baseline_runs.sort(key=lambda x: ranking_tuple(x.metrics))
            baselines["global"] = all_baseline_runs[0]
        return {client: run.run_id for client, run in baselines.items()}

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
            image_paths = [self.output_dir / run.image_path for run in best_runs if run.image_path]
            image_paths = [path for path in image_paths if path.exists()]
            if image_paths:
                auto_path = self.montage_dir / "top_ranked.png"
                self.build_montage(image_paths, auto_path)
                montages.append(
                    {"name": "Top reconstructions", "path": safe_relative(auto_path, self.output_dir)}
                )
        return montages

    # ------------------------------------------------------------------
    def build_montage(self, image_paths: List[Path], dest: Path) -> None:
        cols = 3
        thumb_w, thumb_h = 420, 280
        rows = math.ceil(len(image_paths) / cols)
        canvas = Image.new(
            "RGB",
            (cols * thumb_w, rows * thumb_h),
            color=(12, 16, 24),
        )
        for idx, img_path in enumerate(image_paths):
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img.thumbnail((thumb_w - 10, thumb_h - 10))
                    x = (idx % cols) * thumb_w + (thumb_w - img.width) // 2
                    y = (idx // cols) * thumb_h + (thumb_h - img.height) // 2
                    canvas.paste(img, (x, y))
            except Exception:
                continue
        ensure_directory(dest.parent)
        canvas.save(dest)

    # ------------------------------------------------------------------
    def generate_charts(self, df: pd.DataFrame) -> Dict[str, str]:
        chart_map: Dict[str, str] = {}
        chart_funcs = {
            "defenses_grouped_bars": self.chart_defenses_grouped_bars,
            "defenses_scatter_psnr_vs_lpips": self.chart_defenses_scatter,
            "multiclient_boxplots": self.chart_multiclient_boxplots,
            "ablation_bars": self.chart_ablation_bars,
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
        ax.bar([i - width for i in x], psnr_means, width, yerr=psnr_std, label="PSNR")
        ax.bar(x, ssim_means, width, yerr=ssim_std, label="SSIM")
        ax.bar(
            [i + width for i in x],
            lpips_means,
            width,
            yerr=lpips_std,
            label="LPIPS",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Defenses: Mean metrics with std bars")
        ax.legend()

        ax2 = ax.twinx()
        label_match = metrics[("LabelMatch", "mean")].tolist()
        ax2.plot(x, label_match, color="#f5d742", marker="o", label="LabelMatch")
        ax2.set_ylabel("LabelMatch rate")
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
            subset["LPIPS"], subset["PSNR"], c=subset["SSIM"], cmap="viridis", s=80
        )
        ax.set_xlabel("LPIPS (lower better)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("Defenses: PSNR vs LPIPS (color=SSIM)")
        fig.colorbar(scatter, label="SSIM")
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
    def chart_ablation_bars(self, df: pd.DataFrame, dest: Path) -> None:
        subset = df[df["group"] == "ablation"]
        if subset.empty:
            self.save_empty_chart(dest, "Ablations", "No ablation runs.")
            return
        means = subset.groupby("method")[["PSNR", "SSIM", "LPIPS"]].mean().sort_values(
            by="LPIPS"
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        idx = range(len(means))
        ax.barh(idx, means["PSNR"], label="PSNR")
        ax.barh(idx, means["SSIM"], left=means["PSNR"], label="SSIM")
        ax.set_yticks(list(idx))
        ax.set_yticklabels(means.index.tolist())
        ax.set_xlabel("PSNR + SSIM (stacked)")
        ax.set_title("Ablation families (stacked PSNR/SSIM)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(dest, dpi=150)
        plt.close(fig)

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
            color="#fefefe",
            fontsize=14,
        )
        ax.text(
            0.5,
            0.4,
            message,
            ha="center",
            va="center",
            color="#d1d5db",
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
        html_path = self.output_dir / "index.html"
        html_content = DASHBOARD_HTML.strip()
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
# Static HTML skeleton (vanilla CSS/JS)
# -----------------------------------------------------------------------------

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Federated Learning Gradient Inversion: Interactive Explorer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f131c;
      --bg-card: #171d2a;
      --bg-panel: #1f2636;
      --border: #2e3548;
      --text: #f5f6fa;
      --muted: #c7cddf;
      --accent: #69a9ff;
      --accent-strong: #f5d742;
      --danger: #ff7676;
      font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: inherit;
      line-height: 1.4;
    }
    .page {
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }
    header {
      background: #0b0e15;
      padding: 18px 32px;
      display: flex;
      flex-wrap: wrap;
      align-items: flex-end;
      justify-content: space-between;
      border-bottom: 1px solid var(--border);
    }
    header h1 {
      margin: 0;
      font-size: 26px;
      letter-spacing: 0.02em;
    }
    header .meta {
      text-align: right;
      font-size: 14px;
      color: var(--muted);
    }
    header .key-finding {
      margin-top: 6px;
      font-size: 15px;
      color: var(--accent-strong);
    }
    main {
      display: flex;
      flex: 1;
      overflow: hidden;
    }
    .sidebar {
      width: 320px;
      background: var(--bg-card);
      border-right: 1px solid var(--border);
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .filter-block h3 {
      margin: 0 0 10px;
      font-size: 15px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .filter-options {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .chip {
      padding: 6px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font-size: 13px;
      transition: all 0.2s ease;
    }
    .chip.active {
      background: rgba(105, 169, 255, 0.15);
      border-color: var(--accent);
      color: var(--text);
    }
    .sidebar input[type="search"],
    .sidebar select {
      width: 100%;
      padding: 8px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--bg-panel);
      color: var(--text);
    }
    .quick-actions {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 10px;
    }
    .quick-actions button {
      padding: 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--bg-panel);
      color: var(--text);
      cursor: pointer;
      font-weight: 500;
      font-size: 13px;
    }
    .quick-actions button.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 1px rgba(105, 169, 255, 0.4);
    }
    .viewer {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .viewer-panels {
      display: flex;
      flex-direction: row;
      gap: 20px;
      padding: 20px 28px;
      border-bottom: 1px solid var(--border);
    }
    .panel {
      flex: 1;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .panel h2 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0.04em;
    }
    .selected-image {
      width: 100%;
      max-height: 420px;
      object-fit: contain;
      background: #090d13;
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 10px;
    }
    .metric-card {
      background: var(--bg-panel);
      border-radius: 8px;
      padding: 8px 10px;
      border: 1px solid var(--border);
      text-align: center;
    }
    .metric-card .label {
      font-size: 11px;
      text-transform: uppercase;
      color: var(--muted);
    }
    .metric-card .value {
      font-size: 20px;
      font-weight: 600;
    }
    .run-path {
      font-family: "SFMono-Regular", "Consolas", monospace;
      font-size: 12px;
      color: var(--muted);
      background: #0b0f18;
      padding: 8px 10px;
      border-radius: 6px;
      overflow-x: auto;
      border: 1px solid var(--border);
    }
    .metrics-text {
      background: #0f131d;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      max-height: 140px;
      overflow-y: auto;
      font-family: "SFMono-Regular", "Consolas", monospace;
      font-size: 12px;
    }
    .nav-buttons {
      display: flex;
      gap: 10px;
      justify-content: flex-end;
    }
    .nav-buttons button {
      padding: 8px 14px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--text);
      cursor: pointer;
    }
    .comparison-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
    }
    .comparison-grid img {
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #090d13;
      max-height: 220px;
      object-fit: contain;
    }
    .comparison-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .comparison-actions button {
      margin-left: 6px;
      padding: 6px 12px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: transparent;
      color: var(--text);
      cursor: pointer;
      font-size: 12px;
    }
    .delta-card {
      padding: 8px;
      background: var(--bg-panel);
      border-radius: 6px;
      border: 1px solid var(--border);
      text-align: center;
    }
    .delta-value {
      font-size: 16px;
      font-weight: bold;
    }
    .tabs {
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 0 28px 28px;
    }
    .tab-buttons {
      display: flex;
      gap: 16px;
      margin-top: 16px;
      border-bottom: 1px solid var(--border);
    }
    .tab-buttons button {
      border: none;
      background: none;
      color: var(--muted);
      padding: 12px 4px;
      cursor: pointer;
      border-bottom: 3px solid transparent;
      font-size: 15px;
    }
    .tab-buttons button.active {
      color: var(--text);
      border-color: var(--accent);
    }
    .tab-panel {
      display: none;
      padding: 18px 0;
      flex: 1;
      overflow-y: auto;
    }
    .tab-panel.active {
      display: block;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    thead {
      position: sticky;
      top: 0;
      background: var(--bg-card);
    }
    th, td {
      border-bottom: 1px solid var(--border);
      padding: 8px;
      text-align: left;
    }
    tbody tr {
      cursor: pointer;
    }
    tbody tr.selected {
      background: rgba(105, 169, 255, 0.12);
    }
    .charts-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }
    .charts-grid img,
    .montage-grid img {
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #090d13;
    }
    .montage-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }
    .empty-state {
      padding: 18px;
      border-radius: 10px;
      background: var(--bg-card);
      border: 1px dashed var(--border);
      color: var(--muted);
      text-align: center;
    }
    .leaderboard-controls {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .lightbox {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.8);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 999;
      padding: 20px;
    }
    .lightbox.active {
      display: flex;
    }
    .lightbox img {
      max-width: 90vw;
      max-height: 85vh;
      border-radius: 12px;
      border: 2px solid var(--border);
    }
    @media (max-width: 1100px) {
      main {
        flex-direction: column;
      }
      .sidebar {
        width: 100%;
        border-right: none;
        border-bottom: 1px solid var(--border);
        flex-direction: row;
        flex-wrap: wrap;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <header>
      <div>
        <h1>Federated Learning Gradient Inversion: Interactive Explorer</h1>
        <div class="key-finding" id="keyFinding"></div>
      </div>
      <div class="meta">
        <div id="buildDate"></div>
        <div id="runCounts"></div>
      </div>
    </header>
    <main>
      <aside class="sidebar">
        <div class="filter-block">
          <h3>Search</h3>
          <input type="search" id="searchInput" placeholder="Find path, client, or method" />
        </div>
        <div class="filter-block">
          <h3>Group</h3>
          <div id="groupFilters" class="filter-options"></div>
        </div>
        <div class="filter-block">
          <h3>Method / Setting</h3>
          <div id="methodFilters" class="filter-options"></div>
        </div>
        <div class="filter-block">
          <h3>Client</h3>
          <div id="clientFilters" class="filter-options"></div>
        </div>
        <div class="filter-block">
          <h3>Sort</h3>
          <select id="sortSelect">
            <option value="best">Best (LPIPS asc, SSIM desc, PSNR desc)</option>
            <option value="psnr">PSNR high to low</option>
            <option value="ssim">SSIM high to low</option>
            <option value="lpips">LPIPS low to high</option>
            <option value="labelmatch">LabelMatch high to low</option>
            <option value="path">Run path</option>
          </select>
        </div>
        <div class="filter-block">
          <h3>Quick actions</h3>
          <div class="quick-actions">
            <button id="btnBestOverall">Best overall</button>
            <button id="btnBaselineBest">Baseline best</button>
            <button id="btnBaselineWorst">Worst baseline</button>
            <button id="btnDpSweep">DP sweep</button>
            <button id="btnCompareToggle">Compare vs baseline</button>
          </div>
        </div>
      </aside>
      <section class="viewer">
        <div class="viewer-panels">
          <div class="panel" id="selectedPanel">
            <div class="comparison-header">
              <h2>Selected run</h2>
              <div class="nav-buttons">
                <button id="prevBtn" title="Previous run (Arrow Left)">Prev</button>
                <button id="nextBtn" title="Next run (Arrow Right)">Next</button>
              </div>
            </div>
            <img id="selectedImage" class="selected-image" alt="Selected run image" />
            <div class="metrics-grid" id="selectedMetrics"></div>
            <div class="run-path" id="runPath"></div>
            <div class="metrics-text" id="metricsText">Metrics file not available.</div>
          </div>
          <div class="panel" id="comparisonPanel">
            <div class="comparison-header">
              <h2>Comparison</h2>
              <div class="comparison-actions">
                <span id="comparisonLabel">Comparison disabled</span>
                <button id="lockBaselineBtn">Lock baseline</button>
              </div>
            </div>
            <div class="comparison-grid">
              <div>
                <div style="font-size:14px; color:var(--muted); margin-bottom:6px;">Baseline</div>
                <img id="comparisonBaseImage" alt="Baseline image" />
              </div>
              <div>
                <div style="font-size:14px; color:var(--muted); margin-bottom:6px;">Selected</div>
                <img id="comparisonTargetImage" alt="Selected image" />
              </div>
            </div>
            <div class="metrics-grid" id="deltaMetrics"></div>
          </div>
        </div>
        <section class="tabs">
          <div class="tab-buttons">
            <button data-tab="leaderboards" class="active">Leaderboards</button>
            <button data-tab="distributions">Distributions</button>
            <button data-tab="ablations">Ablations</button>
            <button data-tab="montages">Montages</button>
          </div>
          <div id="tab-leaderboards" class="tab-panel active">
            <div class="leaderboard-controls">
              <div id="filteredCount"></div>
              <div>Click any row to load that run.</div>
            </div>
            <div class="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th data-sort="group">Group</th>
                    <th data-sort="method">Method</th>
                    <th data-sort="client">Client</th>
                    <th data-sort="psnr">PSNR</th>
                    <th data-sort="ssim">SSIM</th>
                    <th data-sort="lpips">LPIPS</th>
                    <th data-sort="labelmatch">LabelMatch</th>
                    <th>Image</th>
                  </tr>
                </thead>
                <tbody id="leaderboardBody"></tbody>
              </table>
            </div>
          </div>
          <div id="tab-distributions" class="tab-panel">
            <div class="charts-grid" id="distributionCharts"></div>
          </div>
          <div id="tab-ablations" class="tab-panel">
            <div class="charts-grid" id="ablationCharts"></div>
            <div id="ablationTable"></div>
          </div>
          <div id="tab-montages" class="tab-panel">
            <div class="montage-grid" id="montageGrid"></div>
            <div class="empty-state" id="montageEmpty" style="display:none;">
              No montages available. Provide montage PNGs in results/report/montages/.
            </div>
          </div>
        </section>
      </section>
    </main>
  </div>
  <div class="lightbox" id="lightbox">
    <img id="lightboxImg" alt="Full view" />
  </div>
  <script>
    const state = {
      data: null,
      runsById: {},
      filters: {
        groups: new Set(),
        methods: new Set(),
        clients: new Set(),
      },
      search: "",
      sort: "best",
      compareEnabled: false,
      lockBaselineId: null,
      selectedRunId: null,
      filteredIds: [],
      tableSort: { field: "psnr", dir: "desc" },
    };

    const METRIC_FIELDS = ["PSNR", "SSIM", "LPIPS", "MSE", "LabelMatch"];

    fetch("data.json")
      .then((res) => res.json())
      .then(initDashboard)
      .catch((err) => {
        console.error(err);
        document.body.innerHTML = "<pre style='padding:20px;color:#fff;'>Failed to load dashboard data.</pre>";
      });

    function initDashboard(data) {
      state.data = data;
      data.runs.forEach((run) => {
        state.runsById[run.run_id] = run;
      });
      populateFilters(data.filter_values);
      state.selectedRunId = data.best_overall_id || (data.runs[0] && data.runs[0].run_id);
      updateMeta(data.meta);
      wireEvents();
      populateCharts(data.charts);
      populateAblations(data.aggregates.ablations || []);
      populateMontages(data.montages || []);
      applyFilters();
    }

    function populateFilters(values) {
      ["groups", "methods", "clients"].forEach((key) => {
        state.filters[key] = new Set(values[key]);
        const container = document.getElementById(`${key.slice(0, -1)}Filters`);
        container.innerHTML = "";
        values[key].forEach((value) => {
          const btn = document.createElement("button");
          btn.className = "chip active";
          btn.textContent = value;
          btn.dataset.value = value;
          btn.addEventListener("click", () => toggleFilter(key, value, btn));
          container.appendChild(btn);
        });
      });
    }

    function toggleFilter(key, value, btn) {
      const set = state.filters[key];
      if (set.has(value)) {
        set.delete(value);
        btn.classList.remove("active");
      } else {
        set.add(value);
        btn.classList.add("active");
      }
      if (set.size === 0) {
        setFilterSet(key, state.data.filter_values[key]);
        return;
      }
      applyFilters();
    }

    function updateMeta(meta) {
      document.getElementById("buildDate").textContent = `Build date: ${meta.build_time}`;
      const counts = Object.entries(meta.groups || {})
        .map(([group, count]) => `${group}: ${count}`)
        .join("  ·  ");
      document.getElementById("runCounts").textContent = `Total runs: ${meta.total_runs} (${counts})`;
      document.getElementById("keyFinding").textContent = meta.key_finding || "";
    }

    function wireEvents() {
      document.getElementById("searchInput").addEventListener("input", (ev) => {
        state.search = ev.target.value.toLowerCase();
        applyFilters();
      });
      document.getElementById("sortSelect").addEventListener("change", (ev) => {
        state.sort = ev.target.value;
        applyFilters();
      });
      document.getElementById("btnBestOverall").addEventListener("click", () => {
        state.sort = "best";
        document.getElementById("sortSelect").value = "best";
        state.selectedRunId = state.data.best_overall_id;
        clearFilterSelections();
        applyFilters();
      });
      document.getElementById("btnBaselineBest").addEventListener("click", () => {
        focusOnMethod("baseline");
        const bestBaseline = state.data.best_baseline_id;
        if (bestBaseline) {
          state.selectedRunId = bestBaseline;
        }
        applyFilters();
      });
      document.getElementById("btnBaselineWorst").addEventListener("click", () => {
        focusOnMethod("baseline");
        const baselineRuns = getRuns().filter((run) => run.method.toLowerCase().includes("baseline"));
        baselineRuns.sort((a, b) => worstSort(a, b));
        state.selectedRunId = baselineRuns[0] ? baselineRuns[0].run_id : state.selectedRunId;
        applyFilters();
      });
      document.getElementById("btnDpSweep").addEventListener("click", () => {
        const dpMethods = state.data.filter_values.methods.filter((m) => m.toLowerCase().includes("dp"));
        setFilterSet("methods", dpMethods);
        applyFilters();
      });
      document.getElementById("btnCompareToggle").addEventListener("click", (ev) => {
        state.compareEnabled = !state.compareEnabled;
        ev.currentTarget.classList.toggle("active", state.compareEnabled);
        updateComparison();
      });
      document.getElementById("lockBaselineBtn").addEventListener("click", () => {
        if (!state.compareEnabled) return;
        const baseline = getBaselineForSelection();
        if (!baseline) return;
        state.lockBaselineId = state.lockBaselineId === baseline.run_id ? null : baseline.run_id;
        updateComparison();
      });
      document.getElementById("prevBtn").addEventListener("click", () => jumpSelection(-1));
      document.getElementById("nextBtn").addEventListener("click", () => jumpSelection(1));
      document.addEventListener("keydown", (ev) => {
        if (ev.key === "ArrowLeft") jumpSelection(-1);
        if (ev.key === "ArrowRight") jumpSelection(1);
      });
      document.querySelectorAll(".tab-buttons button").forEach((btn) => {
        btn.addEventListener("click", () => activateTab(btn.dataset.tab));
      });
      document.getElementById("lightbox").addEventListener("click", () => {
        document.getElementById("lightbox").classList.remove("active");
      });
      document.querySelectorAll("th[data-sort]").forEach((th) => {
        th.addEventListener("click", () => {
          const field = th.dataset.sort;
          const dir = state.tableSort.field === field && state.tableSort.dir === "desc" ? "asc" : "desc";
          state.tableSort = { field, dir };
          renderLeaderboards();
        });
      });
    }

    function clearFilterSelections() {
      ["groups", "methods", "clients"].forEach((key) => {
        setFilterSet(key, state.data.filter_values[key]);
      });
    }

    function focusOnMethod(name) {
      const target = state.data.filter_values.methods.filter((m) => m.toLowerCase().includes(name));
      if (target.length === 0) return;
      setFilterSet("methods", target);
    }

    function setFilterSet(key, values) {
      const arr = Array.isArray(values) ? values : Array.from(values);
      state.filters[key] = new Set(arr);
      const container = document.getElementById(`${key.slice(0, -1)}Filters`);
      Array.from(container.children).forEach((chip) => {
        chip.classList.toggle("active", arr.includes(chip.dataset.value));
      });
    }

    function getRuns() {
      return state.data.runs;
    }

    function applyFilters() {
      let runs = getRuns().slice();
      runs = runs.filter((run) => state.filters.groups.has(run.group));
      runs = runs.filter((run) => state.filters.methods.has(run.method));
      runs = runs.filter((run) => state.filters.clients.has(run.client));
      if (state.search) {
        runs = runs.filter((run) =>
          run.search_text && run.search_text.indexOf(state.search) !== -1
        );
      }
      runs.sort(sorterFor(state.sort));
      state.filteredIds = runs.map((run) => run.run_id);
      if (!state.filteredIds.includes(state.selectedRunId) && state.filteredIds.length > 0) {
        state.selectedRunId = state.filteredIds[0];
      }
      renderLeaderboards();
      renderSelectedRun();
      updateComparison();
    }

    function sorterFor(key) {
      switch (key) {
        case "psnr":
          return (a, b) => numeric(b.metrics.PSNR) - numeric(a.metrics.PSNR);
        case "ssim":
          return (a, b) => numeric(b.metrics.SSIM) - numeric(a.metrics.SSIM);
        case "lpips":
          return (a, b) => numeric(a.metrics.LPIPS) - numeric(b.metrics.LPIPS);
        case "labelmatch":
          return (a, b) => numeric(b.metrics.LabelMatch) - numeric(a.metrics.LabelMatch);
        case "path":
          return (a, b) => a.source_dir.localeCompare(b.source_dir);
        default:
          return (a, b) => bestSort(a, b);
      }
    }

    function bestSort(a, b) {
      const lp = numeric(a.metrics.LPIPS) - numeric(b.metrics.LPIPS);
      if (lp !== 0) return lp;
      const ssim = numeric(b.metrics.SSIM) - numeric(a.metrics.SSIM);
      if (ssim !== 0) return ssim;
      return numeric(b.metrics.PSNR) - numeric(a.metrics.PSNR);
    }

    function worstSort(a, b) {
      const lp = numeric(b.metrics.LPIPS) - numeric(a.metrics.LPIPS);
      if (lp !== 0) return lp;
      return numeric(a.metrics.SSIM) - numeric(b.metrics.SSIM);
    }

    function numeric(value) {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return 0;
      }
      return Number(value);
    }

    function renderLeaderboards() {
      const tbody = document.getElementById("leaderboardBody");
      tbody.innerHTML = "";
      document.getElementById("filteredCount").textContent = `${state.filteredIds.length} runs match filters`;
      const runs = state.filteredIds.map((id) => state.runsById[id]);
      runs.sort((a, b) => {
        const dir = state.tableSort.dir === "asc" ? 1 : -1;
        switch (state.tableSort.field) {
          case "group":
            return a.group.localeCompare(b.group) * dir;
          case "method":
            return a.method.localeCompare(b.method) * dir;
          case "client":
            return a.client.localeCompare(b.client) * dir;
          case "psnr":
            return (numeric(a.metrics.PSNR) - numeric(b.metrics.PSNR)) * dir;
          case "ssim":
            return (numeric(a.metrics.SSIM) - numeric(b.metrics.SSIM)) * dir;
          case "lpips":
            return (numeric(a.metrics.LPIPS) - numeric(b.metrics.LPIPS)) * dir;
          case "labelmatch":
            return (numeric(a.metrics.LabelMatch) - numeric(b.metrics.LabelMatch)) * dir;
          default:
            return 0;
        }
      });
      runs.forEach((run) => {
        const tr = document.createElement("tr");
        if (run.run_id === state.selectedRunId) {
          tr.classList.add("selected");
        }
        tr.innerHTML = `
          <td>${run.group}</td>
          <td>${run.method}</td>
          <td>${run.client}</td>
          <td>${formatMetric(run.metrics.PSNR)}</td>
          <td>${formatMetric(run.metrics.SSIM)}</td>
          <td>${formatMetric(run.metrics.LPIPS)}</td>
          <td>${formatMetric(run.metrics.LabelMatch)}</td>
          <td>${run.image_path ? "image" : "n/a"}</td>
        `;
        tr.addEventListener("click", () => {
          state.selectedRunId = run.run_id;
          renderSelectedRun();
          updateComparison();
          renderLeaderboards();
        });
        tbody.appendChild(tr);
      });
    }

    function renderSelectedRun() {
      const run = state.runsById[state.selectedRunId];
      if (!run) return;
      document.getElementById("selectedImage").src = run.image_path;
      document.getElementById("comparisonTargetImage").src = run.image_path;
      document.getElementById("runPath").textContent = run.source_dir || "Path unavailable";
      const metricsContainer = document.getElementById("selectedMetrics");
      metricsContainer.innerHTML = "";
      ["PSNR", "SSIM", "LPIPS", "MSE", "LabelMatch"].forEach((metric) => {
        const div = document.createElement("div");
        div.className = "metric-card";
        div.innerHTML = `<div class="label">${metric}</div><div class="value">${formatMetric(run.metrics[metric])}</div>`;
        metricsContainer.appendChild(div);
      });
      loadMetricsText(run);
    }

    function formatMetric(value) {
      if (value === undefined || value === null || Number.isNaN(value)) {
        return "—";
      }
      if (Math.abs(value) >= 100) {
        return value.toFixed(1);
      }
      if (Math.abs(value) >= 10) {
        return value.toFixed(2);
      }
      return value.toFixed(3);
    }

    const metricsCache = {};
    function loadMetricsText(run) {
      const box = document.getElementById("metricsText");
      if (!run.metrics_path) {
        box.textContent = "metrics.txt not available for this run.";
        return;
      }
      if (metricsCache[run.metrics_path]) {
        box.textContent = metricsCache[run.metrics_path];
        return;
      }
      fetch(run.metrics_path)
        .then((res) => res.text())
        .then((text) => {
          metricsCache[run.metrics_path] = text;
          if (state.selectedRunId === run.run_id) {
            box.textContent = text;
          }
        })
        .catch(() => {
          box.textContent = "Failed to load metrics.txt.";
        });
    }

    function updateComparison() {
      const label = document.getElementById("comparisonLabel");
      const deltaContainer = document.getElementById("deltaMetrics");
      const baseImg = document.getElementById("comparisonBaseImage");
      if (!state.compareEnabled) {
        label.textContent = "Comparison disabled";
        deltaContainer.innerHTML = "";
        baseImg.src = "";
        return;
      }
      const baseline = getBaselineForSelection();
      const selected = state.runsById[state.selectedRunId];
      if (!baseline || !selected) {
        label.textContent = "No baseline available";
        deltaContainer.innerHTML = "";
        baseImg.src = "";
        return;
      }
      baseImg.src = baseline.image_path;
      const lockedText = state.lockBaselineId ? " (locked)" : "";
      label.textContent = `Comparing to ${baseline.method} / ${baseline.client}${lockedText}`;
      deltaContainer.innerHTML = "";
      [["PSNR", "dB"], ["SSIM", ""], ["LPIPS", ""], ["LabelMatch", ""]].forEach(([metric, suffix]) => {
        const baseVal = baseline.metrics[metric];
        const selVal = selected.metrics[metric];
        const delta = selVal !== undefined && baseVal !== undefined ? selVal - baseVal : null;
        const div = document.createElement("div");
        div.className = "delta-card";
        const deltaText = delta === null || Number.isNaN(delta) ? "—" : formatDelta(delta);
        div.innerHTML = `<div class="label">${metric}</div><div class="delta-value">${deltaText}${suffix}</div>`;
        deltaContainer.appendChild(div);
      });
    }

    function formatDelta(value) {
      const val = value;
      if (val === null || Number.isNaN(val)) return "—";
      const sign = val > 0 ? "+" : "";
      if (Math.abs(val) >= 10) return `${sign}${val.toFixed(1)}`;
      return `${sign}${val.toFixed(3)}`;
    }

    function getBaselineForSelection() {
      if (!state.compareEnabled) return null;
      const selected = state.runsById[state.selectedRunId];
      if (!selected) return null;
      if (state.lockBaselineId) {
        return state.runsById[state.lockBaselineId];
      }
      const client = selected.client || "global";
      const baselineId = state.data.baselines_by_client[client] || state.data.baselines_by_client.global;
      return state.runsById[baselineId];
    }

    function jumpSelection(delta) {
      const idx = state.filteredIds.indexOf(state.selectedRunId);
      if (idx === -1) return;
      const nextIdx = Math.min(Math.max(idx + delta, 0), state.filteredIds.length - 1);
      state.selectedRunId = state.filteredIds[nextIdx];
      renderSelectedRun();
      updateComparison();
      renderLeaderboards();
    }

    function activateTab(tabName) {
      document.querySelectorAll(".tab-buttons button").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.tab === tabName);
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === `tab-${tabName}`);
      });
    }

    function populateCharts(charts) {
      const distContainer = document.getElementById("distributionCharts");
      distContainer.innerHTML = "";
      ["defenses_grouped_bars", "defenses_scatter_psnr_vs_lpips", "multiclient_boxplots"].forEach((key) => {
        const path = charts[key];
        const div = document.createElement("div");
        if (path) {
          div.innerHTML = `<div style="margin-bottom:8px;font-size:14px;color:var(--muted);">${key.replace(/_/g, " ")}</div><img src="${path}" alt="${key}" />`;
        } else {
          div.innerHTML = `<div class="empty-state">Chart ${key} unavailable.</div>`;
        }
        distContainer.appendChild(div);
      });
      const ablationContainer = document.getElementById("ablationCharts");
      ablationContainer.innerHTML = "";
      const ablationPath = charts["ablation_bars"];
      if (ablationPath) {
        const div = document.createElement("div");
        div.innerHTML = `<div style="margin-bottom:8px;font-size:14px;color:var(--muted);">Ablation summary</div><img src="${ablationPath}" alt="Ablation chart" />`;
        ablationContainer.appendChild(div);
      }
    }

    function populateAblations(rows) {
      const container = document.getElementById("ablationTable");
      if (!rows || rows.length === 0) {
        container.innerHTML = '<div class="empty-state">No ablation data.</div>';
        return;
      }
      const table = document.createElement("table");
      table.innerHTML = `
        <thead>
          <tr>
            <th>Method</th>
            <th>Client</th>
            <th>PSNR</th>
            <th>SSIM</th>
            <th>LPIPS</th>
          </tr>
        </thead>
      `;
      const tbody = document.createElement("tbody");
      rows.forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${row.method}</td>
          <td>${row.client}</td>
          <td>${formatMetric(row.PSNR)}</td>
          <td>${formatMetric(row.SSIM)}</td>
          <td>${formatMetric(row.LPIPS)}</td>
        `;
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      container.innerHTML = "";
      container.appendChild(table);
    }

    function populateMontages(montages) {
      const grid = document.getElementById("montageGrid");
      const empty = document.getElementById("montageEmpty");
      if (!montages || montages.length === 0) {
        grid.innerHTML = "";
        empty.style.display = "block";
        return;
      }
      empty.style.display = "none";
      grid.innerHTML = "";
      montages.forEach((item) => {
        const card = document.createElement("div");
        card.innerHTML = `<div style="margin-bottom:6px;color:var(--muted);">${item.name}</div>`;
        const img = document.createElement("img");
        img.src = item.path;
        img.alt = item.name;
        img.addEventListener("click", () => openLightbox(item.path));
        card.appendChild(img);
        grid.appendChild(card);
      });
    }

    function openLightbox(path) {
      const modal = document.getElementById("lightbox");
      document.getElementById("lightboxImg").src = path;
      modal.classList.add("active");
    }
  </script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    builder = DashboardBuilder()
    builder.run()


if __name__ == "__main__":
    main()
