#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# ----------------------------
# Parsing
# ----------------------------
def parse_metrics(metrics_path: Path) -> dict:
    d = {}
    if not metrics_path.exists():
        return d
    for line in metrics_path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            d[k] = float(v)
        except ValueError:
            d[k] = v
    return d

def safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default

# ----------------------------
# Discovery (matches your folders)
# ----------------------------
def collect_runs(results_root: Path):
    runs = []

    # Showcase
    showcase = results_root / "showcase"
    if (showcase / "metrics.txt").exists():
        runs.append({
            "group": "showcase",
            "setting": "showcase",
            "client": "hero",
            "metrics_path": showcase / "metrics.txt",
            "image_path": showcase / "baseline_attack_result.png",
        })

    # Multi-client: results/multi_client/bmk_c*/metrics.txt
    mc_root = results_root / "multi_client"
    if mc_root.exists():
        for mp in sorted(mc_root.glob("bmk_c*/metrics.txt")):
            client = mp.parent.name.replace("bmk_", "")
            runs.append({
                "group": "multi_client",
                "setting": "multi_client",
                "client": client,
                "metrics_path": mp,
                "image_path": mp.parent / "baseline_attack_result.png",
            })

    # Ablation: results/ablation/<setting>/c*/metrics.txt
    ab_root = results_root / "ablation"
    if ab_root.exists():
        for mp in sorted(ab_root.glob("*/*/metrics.txt")):
            setting = mp.parent.parent.name
            client = mp.parent.name
            runs.append({
                "group": "ablation",
                "setting": setting,
                "client": client,
                "metrics_path": mp,
                "image_path": mp.parent / "baseline_attack_result.png",
            })

    return runs

# ----------------------------
# Stats
# ----------------------------
def mean_std(vals):
    vals = [v for v in vals if v == v]  # drop NaNs
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)

# ----------------------------
# Plot helpers
# ----------------------------
def save_bar_with_err(out_png: Path, labels, means, stds, title, ylabel):
    plt.figure()
    x = list(range(len(labels)))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()

def save_boxplot(out_png: Path, label_to_vals, title, ylabel):
    plt.figure()
    labels = list(label_to_vals.keys())
    data = [label_to_vals[k] for k in labels]
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()

# ----------------------------
# Montage helpers
# ----------------------------
def load_font(size=16):
    # Try a few common fonts; fall back to default.
    for name in ["Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def make_grid(images, labels, out_path: Path, cols=4, cell_pad=12, label_h=38, bg=(245,245,245)):
    assert len(images) == len(labels)
    font = load_font(16)

    # Load + normalize sizes
    loaded = []
    for p in images:
        if p and p.exists():
            img = Image.open(p).convert("RGB")
            loaded.append(img)
        else:
            loaded.append(None)

    # Choose target size based on median width
    sizes = [im.size for im in loaded if im is not None]
    if not sizes:
        return
    target_w = sorted([w for (w,h) in sizes])[len(sizes)//2]
    scale_imgs = []
    for im in loaded:
        if im is None:
            scale_imgs.append(None)
            continue
        w,h = im.size
        new_h = int(h * (target_w / w))
        scale_imgs.append(im.resize((target_w, new_h)))

    cell_w = target_w
    cell_h = max(im.size[1] for im in scale_imgs if im is not None) + label_h

    rows = (len(scale_imgs) + cols - 1) // cols
    out_w = cols * cell_w + (cols + 1) * cell_pad
    out_h = rows * cell_h + (rows + 1) * cell_pad

    canvas = Image.new("RGB", (out_w, out_h), bg)
    draw = ImageDraw.Draw(canvas)

    for i, (im, lab) in enumerate(zip(scale_imgs, labels)):
        r = i // cols
        c = i % cols
        x0 = cell_pad + c * (cell_w + cell_pad)
        y0 = cell_pad + r * (cell_h + cell_pad)

        # label area
        draw.rectangle([x0, y0, x0 + cell_w, y0 + label_h], fill=(230,230,230))
        draw.text((x0 + 8, y0 + 8), lab, fill=(10,10,10), font=font)

        # image area
        if im is not None:
            canvas.paste(im, (x0, y0 + label_h))
        else:
            draw.rectangle([x0, y0 + label_h, x0 + cell_w, y0 + cell_h], outline=(120,120,120), width=2)
            draw.text((x0 + 8, y0 + label_h + 8), "missing image", fill=(120,0,0), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results", help="Root results dir (contains ablation/, multi_client/, showcase/)")
    ap.add_argument("--out-dir", default="report", help="Where to write plots/csv/montages")
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(results_root)

    # Enrich with metrics
    rows = []
    for r in runs:
        m = parse_metrics(r["metrics_path"])
        row = {
            "group": r["group"],
            "setting": r["setting"],
            "client": r["client"],
            "metrics_path": str(r["metrics_path"]),
            "image_path": str(r["image_path"]),
            "MSE": safe_float(m.get("MSE")),
            "PSNR": safe_float(m.get("PSNR")),
            "SSIM": safe_float(m.get("SSIM")),
            "LPIPS": safe_float(m.get("LPIPS")),
            "LabelMatch": safe_float(m.get("LabelMatch")),
        }
        rows.append(row)

    # Write all_runs.csv
    all_csv = out_dir / "all_runs.csv"
    with all_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] wrote {all_csv}")

    # ----------------------------
    # Multi-client plots
    # ----------------------------
    mc = [r for r in rows if r["group"] == "multi_client"]
    if mc:
        psnr = [r["PSNR"] for r in mc]
        ssim = [r["SSIM"] for r in mc]
        lpips = [r["LPIPS"] for r in mc]

        save_boxplot(out_dir / "plots/multi_client_psnr_box.png",
                     {"PSNR": psnr}, "Multi-client attack quality (PSNR)", "PSNR (dB) ↑")
        save_boxplot(out_dir / "plots/multi_client_ssim_box.png",
                     {"SSIM": ssim}, "Multi-client attack quality (SSIM)", "SSIM ↑")
        save_boxplot(out_dir / "plots/multi_client_lpips_box.png",
                     {"LPIPS": lpips}, "Multi-client attack quality (LPIPS)", "LPIPS ↓")

        # Montage: all clients
        mc_imgs = [Path(r["image_path"]) for r in mc]
        mc_labels = [
            f'{r["client"]}  PSNR={r["PSNR"]:.2f}  LPIPS={r["LPIPS"]:.3f}'
            for r in mc
        ]
        make_grid(mc_imgs, mc_labels, out_dir / "montages/multi_client_grid.png", cols=3)
        print(f"[OK] wrote {out_dir / 'montages/multi_client_grid.png'}")

    # ----------------------------
    # Ablation plots (mean/std across clients for each setting)
    # ----------------------------
    ab = [r for r in rows if r["group"] == "ablation"]
    if ab:
        by_setting = defaultdict(list)
        for r in ab:
            by_setting[r["setting"]].append(r)

        settings = sorted(by_setting.keys())
        ab_summary = []
        for s in settings:
            items = by_setting[s]
            m_psnr, sd_psnr = mean_std([it["PSNR"] for it in items])
            m_ssim, sd_ssim = mean_std([it["SSIM"] for it in items])
            m_lp, sd_lp = mean_std([it["LPIPS"] for it in items])
            ab_summary.append({
                "setting": s,
                "n": len(items),
                "PSNR_mean": m_psnr, "PSNR_std": sd_psnr,
                "SSIM_mean": m_ssim, "SSIM_std": sd_ssim,
                "LPIPS_mean": m_lp, "LPIPS_std": sd_lp,
            })

        # write ablation_summary.csv
        ab_csv = out_dir / "ablation_summary.csv"
        with ab_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ab_summary[0].keys()))
            w.writeheader()
            w.writerows(ab_summary)
        print(f"[OK] wrote {ab_csv}")

        save_bar_with_err(out_dir / "plots/ablation_psnr_bars.png",
                          settings,
                          [x["PSNR_mean"] for x in ab_summary],
                          [x["PSNR_std"] for x in ab_summary],
                          "Ablation: PSNR by setting (mean±std across clients)", "PSNR (dB) ↑")
        save_bar_with_err(out_dir / "plots/ablation_ssim_bars.png",
                          settings,
                          [x["SSIM_mean"] for x in ab_summary],
                          [x["SSIM_std"] for x in ab_summary],
                          "Ablation: SSIM by setting (mean±std across clients)", "SSIM ↑")
        save_bar_with_err(out_dir / "plots/ablation_lpips_bars.png",
                          settings,
                          [x["LPIPS_mean"] for x in ab_summary],
                          [x["LPIPS_std"] for x in ab_summary],
                          "Ablation: LPIPS by setting (mean±std across clients)", "LPIPS ↓")

        # Montage: one representative client per setting (prefer c0, else first)
        rep_imgs = []
        rep_labels = []
        for s in settings:
            items = by_setting[s]
            pick = None
            for it in items:
                if it["client"] == "c0":
                    pick = it
                    break
            if pick is None:
                pick = items[0]
            rep_imgs.append(Path(pick["image_path"]))
            rep_labels.append(f'{s}  ({pick["client"]})  PSNR={pick["PSNR"]:.2f}  LPIPS={pick["LPIPS"]:.3f}')
        make_grid(rep_imgs, rep_labels, out_dir / "montages/ablation_representatives.png", cols=3)
        print(f"[OK] wrote {out_dir / 'montages/ablation_representatives.png'}")

    # ----------------------------
    # Showcase montage (just copy into report as a single-tile grid)
    # ----------------------------
    sc = [r for r in rows if r["group"] == "showcase"]
    if sc:
        r = sc[0]
        make_grid([Path(r["image_path"])],
                  [f'SHOWCASE  PSNR={r["PSNR"]:.2f}  SSIM={r["SSIM"]:.3f}  LPIPS={r["LPIPS"]:.3f}'],
                  out_dir / "montages/showcase.png",
                  cols=1)
        print(f"[OK] wrote {out_dir / 'montages/showcase.png'}")

    print("\nDone. Open:")
    print(f"  {out_dir}/plots/")
    print(f"  {out_dir}/montages/")
    print(f"  {out_dir}/all_runs.csv")

if __name__ == "__main__":
    main()
