# Dashboard Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the course-judging tabbed dashboard with a single-page, academic-rigorous
narrative report (own visual identity, no poster) that reads well as both a 60-second
hiring-manager skim and a slower reviewer pass, and retire the now-unused poster artifact.

**Architecture:** `scripts/make_dashboard.py`'s `DashboardBuilder` keeps computing
`data.json` (a data/presentation split already exists — the HTML is static and fetches
`data.json` at runtime, so this is a template + chart-style rewrite, not a data-model
rewrite). The embedded `DASHBOARD_HTML` string is extracted to its own file first
(mechanical, testable in isolation) so the redesign happens in a real `.html` file, not
a 1300-line Python string literal. Chart colors move to one `PALETTE` dict so
matplotlib output and the CSS `:root` never drift independently again.

**Tech Stack:** Python 3 (pandas, matplotlib, Pillow — all confirmed installed locally:
pandas 3.0.3), vanilla HTML/CSS/JS (no framework, no build step, no external font/CDN
dependency — matches this repo's zero-dependency static-page pattern), Playwright MCP
for screenshot verification.

## Global Constraints

- No new experiments, no GPU/CelebA/torch runs. Every number displayed must trace to
  `results/report/summary.csv` (already corrected in the prior phase) — this plan only
  touches *presentation* of already-verified numbers, plus one already-built figure
  (`results/report/figures/dp_noise_scaling.png`).
- No external network dependency in the deployed page (no Google Fonts CDN, no JS
  library CDN) — system font stack only, matching the project's existing zero-dependency
  static-page pattern and avoiding a CSP/offline failure mode for a GitHub Pages file.
- Single source of truth for color: `PALETTE` dict in `scripts/make_dashboard.py`,
  mirrored by hand into the template's CSS `:root` block with a pointer comment in both
  places (`# mirrored in scripts/templates/dashboard.html :root` /
  `/* mirrored from scripts/make_dashboard.py PALETTE */`). Do not introduce a build
  step or shared-config-file mechanism to enforce this automatically — two files,
  eyeballed, is right-sized for a project with one maintainer and ~10 colors.
  <!-- kept in sync with the JS-source palette copy inside dashboard.html — update both
  when changing a color -->
- The generator (`scripts/make_dashboard.py`) always writes to `results/report/dashboard/`.
  `docs/` is the deployed copy, kept in sync by `cp -r`. This plan restores that as a
  hard rule (no more hand-editing `docs/index.html` directly — see Task 6).
- Poster is retired, not reduced: `scripts/make_poster.py` and its outputs are deleted
  outright (user's explicit choice), not kept-but-unused.

---

## Design spec (read this before Task 4 — it is the redesign's source of truth)

### Palette (light, academic-rigorous — paper-and-ink, not GitHub-dark)

| Token | Hex | Use |
|---|---|---|
| `--paper` | `#faf9f6` | page background |
| `--panel` | `#ffffff` | card/panel background |
| `--ink` | `#1a1a1a` | primary text |
| `--muted` | `#5c5a52` | secondary text, captions |
| `--border` | `#ddd9d0` | rules, card borders |
| `--navy` | `#1d3557` | primary accent — headings underline, links, PSNR series |
| `--teal` | `#457b9d` | secondary series — SSIM |
| `--rust` | `#bc4b2c` | finding/caveat accent, LPIPS series — used sparingly, marks "pay attention here" |
| `--olive` | `#6a7f3f` | tertiary series — ablation/LabelMatch |
| `--gray` | `#8d8a80` | neutral/baseline series, disabled states |

Exact same six hexes (`navy`, `teal`, `rust`, `olive`, `gray`, plus `paper`/`panel` for
figure background) are used in matplotlib (Task 3) and CSS (Task 4). No dark mode this
pass — the redesign is a single, deliberate light identity (spec explicitly says "own
strong identity," not "supports both themes"); do not add a `prefers-color-scheme` block.

### Type

- Headings: `Georgia, "Iowan Old Style", "Palatino Linotype", serif` — an academic-paper
  serif using only system/pre-installed fonts (no CDN).
- Body/UI: `-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif`.
- Scale: h1 `clamp(1.75rem, 1.4rem + 1.5vw, 2.5rem)`, h2 `1.5rem`, h3 `1.15rem`, body
  `1rem`, caption/small `0.85rem`. Body line-height `1.6`; headings `1.25`.

### Information architecture — narrative sections, in this order, replacing the
current tab bar (`leaderboards` / `ablations` / `distributions` / `montages`) entirely

1. **Header** — h1 "Privacy Leakage in Federated Learning" (as a link to the GitHub
   repo, keep this from the current hand-edit), one-line subtitle stating dataset/model
   ("CelebA · SimpleCNN, 8.76M parameters · gradient inversion attack"), a small
   metadata line (`meta.build_time`, `meta.total_runs`, groups — reuse existing
   `meta` fields, no new data needed).
2. **Overview** — 2–3 sentences: what a gradient inversion attack is, why FL's privacy
   promise is weaker than it looks, what this project measures (attack quality, not
   model utility).
3. **Threat Model & Scope** — new section, does not exist today. Adapt verbatim from
   `README.md`'s already-approved Key Findings prose (README.md:26–53, written and
   reviewed last phase): state the tested release point (single client's raw gradient,
   `attack_source="gradients"`), name local DP as correctly calibrated *for that release
   point*, name the two unevaluated mechanisms (`aggregate_clipped_noisy`,
   `capture_mode='agg_update'`) as a named scope limit, not a gap. This section is what
   makes Results — Defenses legible without an asterisk.
4. **Results — Attack Effectiveness** — the existing leaderboard table + filters +
   selected-run reconstruction viewer (`leaderboardBody`, `selectedPanel`,
   `runSummary`, `metricsText` — keep all of this, it's the strongest interactive
   artifact) plus the multi-client boxplot chart (currently the "distributions" tab) as
   a "robustness across clients" subsection. **Cut**: the autoplay carousel and its
   speed selector, and the shuffle button — both read as course-demo gimmicks, not
   rigor; a plain sortable/filterable table + click-to-view is the correct academic
   register and is less code.
5. **Results — Defenses** — the centerpiece. `defenses_grouped_bars` chart directly
   beside `dp_noise_scaling.png` (new `figures.dp_noise_scaling` data field, Task 3),
   captioned as a pair: left caption "The tested ε range (8, 1, 0.1) produces flat PSNR
   — this is the finding, not noise in the experiment," right caption "Why: injected
   noise scales with √d; at this model's 8.76M parameters even ε=8 already overwhelms
   the clipped signal (see `scripts/dp_noise_scaling_proof.py`)." Below: the "HE" caveat
   paragraph (ported from README.md:38–50) and the `defenses_scatter_psnr_vs_lpips`
   chart.
6. **Ablation Study** — the three split ablation charts (`ablation_psnr`,
   `ablation_ssim`, `ablation_lpips` — Task 3 restores this 3-way split in the
   generator itself, matching what's already deployed but currently undone by the
   generator on disk) with one sentence framing ("which attack hyperparameters matter
   most") plus the montage grid (`prepare_montages` output) as supporting visual
   evidence, no carousel.
7. **Reproduce It** — condensed Quick Start (install, run showcase, run dashboard
   generator), link to the repo. Mirrors README's Quick Start so the two surfaces don't
   drift in wording (copy the same three commands).
8. **Footer** — build metadata (`meta.build_time`), link to GitHub, license.

### JS behavior changes (relative to current `DASHBOARD_HTML`)

- Remove: `autoplayPanel`, `autoplayToggle`, `autoplaySpeed`, `autoplayStatus`,
  `shuffleBtn` and all their event listeners/intervals.
- Remove: `tab-*` switching logic entirely — all sections render at once, page scrolls.
- Keep unchanged: `renderLeaderboards`, `renderSelectedRun`, filter dropdowns backed by
  `filter_values`, `openLightbox`/`lightbox` for chart/image zoom, montage grid
  rendering.
- Add: on load, set the two new figure `<img>` tags' `src` from `data.figures.*`
  (Task 3 adds this field) instead of a hardcoded path, so a future regenerate can't
  silently break the image path.

---

## Task 1: Retire the poster

**Files:**
- Delete: `scripts/make_poster.py`
- Delete: `results/report/poster_4k.png`, `results/report/poster_4k.pdf`,
  `results/report/poster_4k_light.png`, `results/report/poster_4k_light.pdf`
- Modify: `README.md:11` (hero image line), `README.md:64-65` (Quick Start step)

**Interfaces:** None — this task has no consumers in later tasks. It is independent and
can be its own commit/PR checkpoint.

- [ ] **Step 1: Delete the poster script and its outputs**

```bash
git rm scripts/make_poster.py
git rm results/report/poster_4k.png results/report/poster_4k.pdf \
       results/report/poster_4k_light.png results/report/poster_4k_light.pdf
```

- [ ] **Step 2: Remove the poster hero line from README**

Remove line 11 entirely:
```markdown
![Research Poster](results/report/poster_4k.png)
```
(Task 7 adds a replacement hero image once one exists — leaving no hero image is the
correct intermediate state, not a broken `<img>` reference to a deleted file.)

- [ ] **Step 3: Remove the poster step from README Quick Start**

In the Quick Start code block (README.md:57-69), delete these two lines:
```bash
# Generate poster visualization
python scripts/make_poster.py
```

- [ ] **Step 4: Verify no remaining references**

Run: `grep -rn "make_poster\|poster_4k" --include="*.py" --include="*.md" .`
Expected: no output (empty). If `results/report/dashboard/index.html` or
`docs/index.html` reference a poster image, note it — Task 4/6 replace those files
entirely, so a stale reference there is expected to disappear later, not now.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "chore: retire poster artifact and generator"
```

---

## Task 2: Extract the dashboard HTML template from the Python string

**Files:**
- Create: `scripts/templates/dashboard.html`
- Modify: `scripts/make_dashboard.py:858-862` (`write_index_html`), remove the
  `DASHBOARD_HTML` string constant (lines 890-2212)

**Interfaces:**
- Produces: `scripts/templates/dashboard.html` — a static file `write_index_html` reads
  and copies verbatim. Task 4 edits this file's *content*; this task only moves it.

This task is purely mechanical — prove the output is byte-identical before and after
the move. It does **not** reconcile the current three-way divergence between this
string, `results/report/dashboard/index.html`, and the hand-edited `docs/index.html`
(documented in this plan's investigation, not re-litigated here) — Task 4 replaces the
content outright, making that divergence moot.

- [ ] **Step 1: Capture the current generator output as a baseline**

```bash
python3 scripts/make_dashboard.py
cp results/report/dashboard/index.html /tmp/index_before.html
```

- [ ] **Step 2: Extract the string literal to a file**

Open `scripts/make_dashboard.py`, find the block:
```python
DASHBOARD_HTML = """
<!DOCTYPE html>
...
</html>
"""
```
(lines 890-2212). Copy everything between the two `"""` delimiters (i.e. from
`<!DOCTYPE html>` through the closing `</html>`) into a new file
`scripts/templates/dashboard.html`, then delete the entire `DASHBOARD_HTML = """ ... """`
assignment (and its two comment lines at 886-889) from `make_dashboard.py`.

- [ ] **Step 3: Point `write_index_html` at the file**

Replace (`scripts/make_dashboard.py:858-862`):
```python
    def write_index_html(self) -> None:
        html_path = self.output_dir / "index.html"
        html_content = DASHBOARD_HTML.strip()
        html_path.write_text(html_content, encoding="utf-8")
        print(f"[INFO] Wrote dashboard HTML to {safe_relative(html_path, self.root)}")
```
with:
```python
    def write_index_html(self) -> None:
        template_path = self.root / "scripts" / "templates" / "dashboard.html"
        html_content = template_path.read_text(encoding="utf-8").strip()
        html_path = self.output_dir / "index.html"
        html_path.write_text(html_content, encoding="utf-8")
        print(f"[INFO] Wrote dashboard HTML to {safe_relative(html_path, self.root)}")
```

- [ ] **Step 4: Verify byte-identical output**

```bash
python3 scripts/make_dashboard.py
diff /tmp/index_before.html results/report/dashboard/index.html
```
Expected: no output (files identical). If they differ, the extraction introduced a
whitespace or encoding change — fix before proceeding.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_dashboard.py scripts/templates/dashboard.html
git commit -m "refactor: extract dashboard HTML template to its own file"
```

---

## Task 3: Chart restyle, palette consolidation, ablation 3-way split, static figures

**Files:**
- Modify: `scripts/make_dashboard.py` (module-level `apply_dark_style`, class methods
  `_setup_chart_style`, `chart_defenses_grouped_bars`, `chart_defenses_scatter`,
  `chart_multiclient_boxplots`, `chart_ablation_bars` → split into three methods,
  `generate_charts`, `run`)

**Interfaces:**
- Produces: `PALETTE` dict (module-level, near the top of the file) with keys
  `paper`, `panel`, `ink`, `muted`, `border`, `navy`, `teal`, `rust`, `olive`, `gray` —
  same hex values as the Design Spec table above. Task 4's template mirrors these six
  chart-relevant hexes (`navy`/`teal`/`rust`/`olive`/`gray`/`paper`) by hand in its CSS.
- Produces: `data_blob["figures"]` — `{"dp_noise_scaling": "assets/figures/dp_noise_scaling.png"}`,
  a new top-level key in `data.json`, consumed by Task 4's template JS.
- Produces: `chart_map` keys `ablation_psnr`, `ablation_ssim`, `ablation_lpips`
  (replacing the single `ablation_bars` key) — consumed by Task 4's template.

- [ ] **Step 1: Add the `PALETTE` dict and remove the duplicate dark-style setup**

Replace (`scripts/make_dashboard.py:151-164`):
```python
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
```
with:
```python
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
```
(The module-level `apply_dark_style()` call and its `dark_background` style are gone —
`_setup_chart_style`, called once per `generate_charts`, is now the only place rcParams
are set, removing the prior redundant/conflicting double-styling.)

- [ ] **Step 2: Rewrite `_setup_chart_style` to use `PALETTE`**

Replace (`scripts/make_dashboard.py:641-656`):
```python
    def _setup_chart_style(self):
        """Configure matplotlib for GitHub-inspired dark theme."""
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': '#0d1117',
            'axes.facecolor': '#161b22',
            'axes.edgecolor': '#30363d',
            'axes.labelcolor': '#8b949e',
            'text.color': '#e6edf3',
            'xtick.color': '#8b949e',
            'ytick.color': '#8b949e',
            'grid.color': '#30363d',
            'legend.facecolor': '#161b22',
            'legend.edgecolor': '#30363d',
            'figure.edgecolor': '#30363d',
        })
```
with:
```python
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
```

- [ ] **Step 3: Recolor the defenses grouped-bar chart**

Replace (`scripts/make_dashboard.py:697-716`, the three `ax.bar` calls through
`ax.legend`):
```python
        ax.bar([i - width for i in x], psnr_means, width, yerr=psnr_std, label="PSNR", color="#58a6ff", capsize=3)
        ax.bar(x, ssim_means, width, yerr=ssim_std, label="SSIM", color="#3fb950", capsize=3)
        ax.bar(
            [i + width for i in x],
            lpips_means,
            width,
            yerr=lpips_std,
            label="LPIPS",
            color="#f85149",
            capsize=3,
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Defense Mechanisms: Mean Metrics Comparison", fontweight="bold")
        ax.legend(facecolor='#161b22', edgecolor='#30363d')
```
with:
```python
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
```

And a few lines below, replace the `LabelMatch` line color and `savefig` call
(`scripts/make_dashboard.py:718-722`):
```python
        ax2.plot(x, label_match, color="#d29922", marker="o", label="LabelMatch", linewidth=2)
        ax2.set_ylabel("Label Match Rate")
        fig.tight_layout()
        fig.savefig(dest, dpi=150, facecolor='#0d1117', edgecolor='none')
        plt.close(fig)
```
with:
```python
        ax2.plot(x, label_match, color=PALETTE["olive"], marker="o", label="LabelMatch", linewidth=2)
        ax2.set_ylabel("Label Match Rate")
        fig.tight_layout()
        fig.savefig(dest, dpi=150)
        plt.close(fig)
```
(dropping the hardcoded `facecolor`/`edgecolor` savefig args lets the rcParams set in
Step 2 govern consistently — the same fix applies to `chart_defenses_scatter` below.)

- [ ] **Step 4: Recolor the defenses scatter chart**

Replace (`scripts/make_dashboard.py:729-739`):
```python
        scatter = ax.scatter(
            subset["LPIPS"], subset["PSNR"], c=subset["SSIM"], cmap="plasma", s=100, edgecolors='#30363d', linewidths=1
        )
        ax.set_xlabel("LPIPS (lower = better)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("Defense Analysis: PSNR vs LPIPS", fontweight="bold")
        ax.grid(alpha=0.3)
        cbar = fig.colorbar(scatter, label="SSIM")
        cbar.ax.yaxis.set_tick_params(color='#8b949e')
        fig.tight_layout()
        fig.savefig(dest, dpi=150, facecolor='#0d1117', edgecolor='none')
        plt.close(fig)
```
with:
```python
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
```
(`plasma` is a dark-background-optimized colormap; `viridis` reads correctly on a
light background and remains colorblind-safe.)

- [ ] **Step 5: Split `chart_ablation_bars` into three separate chart methods**

Replace the entire method (`scripts/make_dashboard.py:768-793`):
```python
    def chart_ablation_bars(self, df: pd.DataFrame, dest: Path) -> None:
        subset = df[df["group"] == "ablation"]
        if subset.empty:
            self.save_empty_chart(dest, "Ablations", "No ablation runs.")
            return
        summary = subset.groupby("method")[["PSNR", "SSIM", "LPIPS"]].agg(["mean", "std"])
        summary = summary.sort_values(("LPIPS", "mean"))
        methods = summary.index.tolist()
        metrics = ["PSNR", "SSIM", "LPIPS"]
        titles = ["PSNR (higher better)", "SSIM (higher better)", "LPIPS (lower better)"]
        colors = ["#5ac8fa", "#a390f0", "#f7b05b"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
        x = range(len(methods))
        for idx, metric in enumerate(metrics):
            means = summary[(metric, "mean")].tolist()
            std = summary[(metric, "std")].fillna(0).tolist()
            axes[idx].bar(x, means, yerr=std, capsize=4, color=colors[idx])
            axes[idx].set_title(titles[idx])
            axes[idx].set_xticks(list(x))
            axes[idx].set_xticklabels(methods, rotation=40, ha="right")
            axes[idx].grid(alpha=0.3, axis="y")
        axes[0].set_ylabel("Score")
        fig.suptitle("Ablation families: mean ± std per metric")
        fig.tight_layout()
        fig.savefig(dest, dpi=150)
        plt.close(fig)
```
with:
```python
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
```

- [ ] **Step 6: Update `generate_charts`' chart map**

Replace (`scripts/make_dashboard.py:667-671`):
```python
        chart_funcs = {
            "defenses_grouped_bars": self.chart_defenses_grouped_bars,
            "defenses_scatter_psnr_vs_lpips": self.chart_defenses_scatter,
            "multiclient_boxplots": self.chart_multiclient_boxplots,
            "ablation_bars": self.chart_ablation_bars,
        }
```
with:
```python
        chart_funcs = {
            "defenses_grouped_bars": self.chart_defenses_grouped_bars,
            "defenses_scatter_psnr_vs_lpips": self.chart_defenses_scatter,
            "multiclient_boxplots": self.chart_multiclient_boxplots,
            "ablation_psnr": self.chart_ablation_psnr,
            "ablation_ssim": self.chart_ablation_ssim,
            "ablation_lpips": self.chart_ablation_lpips,
        }
```

- [ ] **Step 7: Recolor `chart_multiclient_boxplots`' facecolor call**

Replace (`scripts/make_dashboard.py:764-765`):
```python
        fig.suptitle("Multi-client metric distributions")
        fig.tight_layout()
        fig.savefig(dest, dpi=150)
```
No change needed here — this method already has no hardcoded facecolor override, so it
already inherits the new light rcParams from Step 2. Skip this step (kept in the plan
so a reviewer can confirm it was checked, not missed).

- [ ] **Step 8: Add `copy_static_figures` and wire `figures` into `data_blob`**

Add a new method, placed after `prepare_montages` (`scripts/make_dashboard.py`, after
line 616):
```python
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
```

Then in `run()` (`scripts/make_dashboard.py:203-250`), add the call and the new
`data_blob` key. Replace:
```python
        montages = self.prepare_montages(runs)
        charts = self.generate_charts(df_augmented)
```
with:
```python
        montages = self.prepare_montages(runs)
        charts = self.generate_charts(df_augmented)
        figures = self.copy_static_figures()
```
And replace:
```python
            "charts": charts,
            "montages": montages,
            "placeholder_image": safe_relative(self.placeholder_path, self.output_dir),
        }
```
with:
```python
            "charts": charts,
            "montages": montages,
            "figures": figures,
            "placeholder_image": safe_relative(self.placeholder_path, self.output_dir),
        }
```

- [ ] **Step 9: Run and smoke-test the output**

```bash
python3 scripts/make_dashboard.py
python3 -c "
import json
from PIL import Image
d = json.load(open('results/report/dashboard/data.json'))
assert set(d['charts']) == {
    'defenses_grouped_bars', 'defenses_scatter_psnr_vs_lpips',
    'multiclient_boxplots', 'ablation_psnr', 'ablation_ssim', 'ablation_lpips',
}, d['charts']
assert d['figures'].get('dp_noise_scaling'), d['figures']
img = Image.open('results/report/dashboard/assets/charts/defenses_grouped_bars.png')
corner = img.convert('RGB').getpixel((2, 2))
assert sum(corner) > 600, f'corner pixel {corner} looks dark, expected light background'
print('OK')
"
```
Expected: prints `OK`. The pixel check is a cheap, real assertion that the chart is
actually light-themed (a dark-background PNG would fail the `sum(corner) > 600` check;
white ≈ `(250, 249, 246)` sums to ~745).

- [ ] **Step 10: Commit**

```bash
git add scripts/make_dashboard.py
git commit -m "refactor: light academic chart palette, split ablation charts, wire static figures"
```

---

## Task 4: Redesign the dashboard template

**Files:**
- Modify: `scripts/templates/dashboard.html` (full content rewrite per the Design Spec
  above)

**Interfaces:**
- Consumes: `data.json` fields already in the schema (`meta`, `runs`, `aggregates`,
  `baselines_by_client`, `baseline_index`, `best_overall_id`, `best_baseline_id`,
  `best_per_setting`, `filter_values`, `charts`, `montages`, `placeholder_image`) plus
  the new `figures` field from Task 3.
- No new interfaces produced — this is the leaf of the dependency chain (Task 6 just
  regenerates and deploys it).

This is a design task, not a mechanical one — the **Design spec** section above
(palette table, type scale, IA section list, JS removal list) is this task's complete
specification. Implement it directly in `scripts/templates/dashboard.html`:

- [ ] **Step 1: Replace `:root` CSS variables**

Set the CSS custom properties to the Design Spec palette table's six chart-relevant
tokens plus `--paper`/`--panel`/`--ink`/`--muted`/`--border` (ten total, matching
`PALETTE` in `scripts/make_dashboard.py` exactly). Update the `<meta name="color-scheme">`
tag (if present) to `light` (no dark variant this pass — see Design Spec's Palette
section).

- [ ] **Step 2: Replace the font stack**

Apply the Design Spec's Type section: serif heading stack, sans body stack, the five
listed `font-size`/`line-height` rules. Remove any `@font-face` or Google Fonts
`<link>` if present (Global Constraints: no external font CDN).

- [ ] **Step 3: Remove the tab bar and autoplay/shuffle UI**

Delete the `tab-leaderboards`/`tab-ablations`/`tab-distributions`/`tab-montages`
button bar and its click-handler JS. Delete `autoplayPanel`, `autoplayToggle`,
`autoplaySpeed`, `autoplayStatus`, `shuffleBtn` elements and their JS (the `setInterval`
autoplay loop and the shuffle click handler) per the Design Spec's JS behavior list.

- [ ] **Step 4: Restructure the page into the eight IA sections**

Reorder/wrap existing sections into `<section>` elements matching the Design Spec's
Information Architecture list (Header, Overview, Threat Model & Scope, Results —
Attack Effectiveness, Results — Defenses, Ablation Study, Reproduce It, Footer). Move
existing elements (`leaderboardBody`, `selectedPanel`, `runSummary`, `metricsText`,
`ablationCharts`, `distributionCharts`, `montageGrid`) into their new section homes —
do not rewrite their internal markup or the JS that populates them, only their
container placement, since Task 3 kept their backing data fields unchanged (except the
three-way ablation chart split, whose new keys `ablation_psnr`/`ablation_ssim`/
`ablation_lpips` replace the old single `ablation_bars` reference in the ablation
rendering JS).

- [ ] **Step 5: Write the Overview and Threat Model & Scope section copy**

Overview (verbatim, 3 sentences):
```html
<p>Federated learning promises that raw client data never leaves the device — only
gradients are shared. This project shows that promise is weaker than it looks: an
honest-but-curious server can reconstruct a recognizable face from a single client's
gradient update alone, no aggregation required. The benchmark below measures
reconstruction quality (attack strength) under no defense and under two candidate
defenses; it does not measure model accuracy, so it cannot speak to the
privacy/utility tradeoff.</p>
```

Threat Model & Scope (verbatim, adapted from `README.md:26-53`):
```html
<p><strong>Every result below reads one client's raw gradient before any
aggregation</strong> (<code>attack_source="gradients"</code>). Local differential
privacy — clip that client's release, add calibrated Gaussian noise — is the correct
mechanism for exactly this release point, and it is what's evaluated here. It is
<em>not</em> a bug that ε=8, 1, and 0.1 all land at roughly the same PSNR: per-coordinate
Gaussian noise has L2 norm scaling ~σ√d, and at this model's d≈8.76M parameters, even
ε=8 already injects noise ~1800× the clipped signal bound. See the figure below.</p>
<p>The "HE" row does not test encryption: the implementation quantizes gradients and
adds fixed-scale Laplace noise; for this model size it never executes real Paillier
encryption, and even when it does, the code decrypts the result before scoring it — a
real secure-aggregation deployment would never expose that decrypted intermediate to
this attacker. An honest version of this experiment exists in the codebase
(<code>capture_mode='agg_update'</code> + <code>aggregate_clipped_noisy</code>'s
central-DP mechanism, attacking only the FedAvg-averaged update) but no GPU/CelebA
compute was available to run it this cycle — a named limitation, not a filled-in
result.</p>
```

- [ ] **Step 6: Wire the DP noise-scaling figure into Results — Defenses**

Add, beside the `defenses_grouped_bars` chart image:
```html
<figure class="paired-figure">
  <img id="dpNoiseScalingFig" alt="DP noise L2 norm vs. clipped signal, by model dimensionality" />
  <figcaption>Why the ε sweep is flat: injected noise scales with √d. At this model's
  8.76M parameters, every tested ε lands in the noise-dominated region.</figcaption>
</figure>
```
And in the data-load JS (where `charts` paths are currently assigned to `<img>` tags),
add:
```javascript
document.getElementById("dpNoiseScalingFig").src = data.figures.dp_noise_scaling;
```

- [ ] **Step 7: Write the Reproduce It section**

Mirror README's Quick Start exactly (same three commands, so the two surfaces don't
drift in wording):
```html
<pre><code># Install dependencies
pip install -r requirements.txt

# Run showcase experiment
bash scripts/run_showcase.sh

# Launch local dashboard
python -m http.server --directory docs 8000</code></pre>
<p><a href="https://github.com/imaddde867/FL-Attack">View source on GitHub</a></p>
```

- [ ] **Step 8: Local visual check**

```bash
python3 scripts/make_dashboard.py
cd results/report/dashboard && python3 -m http.server 8000 &
```
Open `http://localhost:8000` in a browser (or proceed to Task 6's Playwright
screenshot step) and confirm: no tab bar, no autoplay controls, sections appear in the
order from the Design Spec, the DP figure loads beside the defenses bar chart, no
console errors (undefined `data.figures` etc.). Stop the server: `kill %1`.

- [ ] **Step 9: Commit**

```bash
git add scripts/templates/dashboard.html
git commit -m "feat: redesign dashboard as a single-page academic-rigorous narrative report"
```

---

## Task 5: Deploy — regenerate, sync to docs/, screenshot-verify

**Files:**
- Regenerate: `results/report/dashboard/` (via script)
- Sync: `docs/` (overwritten from `results/report/dashboard/`)
- Create (verification artifacts, not committed): screenshots in the scratchpad

**Interfaces:** Consumes Tasks 3+4's finished generator and template. Produces the
actual live-deployed dashboard.

- [ ] **Step 1: Regenerate**

```bash
python3 scripts/make_dashboard.py
```
Expected: prints a build summary ending in "Dashboard saved to: results/report/dashboard".
This regeneration reads the already-corrected `results/report/summary.csv` (the
`he_test` row was removed from that CSV in the prior correctness-and-cleanup phase),
so the deployed copy's stale `he_test` row and outdated key-finding text — both present
in the currently-live `docs/data.json` — are fixed as a side effect of this one
regenerate, not a separate step. (Deliberately not done as an earlier standalone
hotfix: the generator on disk before Task 3 only produced a single combined
`ablation_bars` chart, while the currently-deployed `docs/` has three separate
already-split ablation charts — an intermediate rerun before Task 3 would have
*regressed* that split. Task 3 restores the 3-way split in the generator itself, so
this regenerate is the first point where rerunning the generator is safe.)

- [ ] **Step 2: Sync to docs/**

```bash
rm -rf docs/assets docs/data.json docs/index.html
cp -r results/report/dashboard/assets docs/assets
cp results/report/dashboard/data.json docs/data.json
cp results/report/dashboard/index.html docs/index.html
```
(`results/report/dashboard/index_generated.html`, if present, is a stray/duplicate
output file — do not copy it; if it still exists after Task 2's extraction, delete it:
`rm -f results/report/dashboard/index_generated.html`.)

- [ ] **Step 3: Verify no stale data made it through**

```bash
grep -c "he_test" docs/data.json || echo "0 (clean)"
python3 -c "import json; print(json.load(open('docs/data.json'))['meta']['key_finding'])"
```
Expected: `he_test` count is 0 (or the `grep` prints nothing and the `echo` fallback
fires); the key-finding string should read a PSNR delta consistent with
`results/report/summary.csv` (baseline 29.38, DP+HE 6.37 → "~23.0 dB").

- [ ] **Step 4: Screenshot-verify via Playwright**

```bash
cd docs && python3 -m http.server 8000 &
```
Use the Playwright MCP tools: `mcp__playwright__browser_navigate` to
`http://localhost:8000`, then `mcp__playwright__browser_resize` to `1440x900` and
`mcp__playwright__browser_take_screenshot` (full page) saved to the scratchpad as
`dashboard_desktop.png`; then `browser_resize` to `390x844` and screenshot again as
`dashboard_mobile.png`. Also call `mcp__playwright__browser_console_messages` and
confirm no JS errors (specifically: no "Cannot read property of undefined" from the
`data.figures` wiring in Task 4 Step 6).

```bash
kill %1
```

- [ ] **Step 5: Read the screenshots and actually look**

Use the `Read` tool on both PNGs. Confirm against the Design Spec: light palette
throughout (no leftover dark-theme CSS), no tab bar, sections in the specified order,
the DP noise-scaling figure renders beside the defenses bar chart with its caption,
type is legible at both widths, no layout overflow (horizontal scrollbar) on the
390px-wide screenshot.

If something's visibly wrong, fix it in `scripts/templates/dashboard.html` and repeat
from Step 1.

- [ ] **Step 6: Commit**

```bash
git add docs/ results/report/dashboard/
git commit -m "chore: regenerate and deploy redesigned dashboard"
```

---

## Task 6: README — swap hero image, tidy Quick Start

**Files:**
- Modify: `README.md`

**Interfaces:** Consumes the desktop screenshot produced in Task 5 Step 4 (or a
purpose-cropped figure from it) as the new hero image.

The Key Findings table, DP/HE narrative bullets, Project Structure, Experiments table,
Usage, Requirements, and Notes sections (`README.md:13-53`, `71-128`) were already
corrected in the prior phase and need no further rewrite — this task only touches the
hero image and the now-stale poster-free Quick Start block.

- [ ] **Step 1: Produce a hero image**

Crop or reuse the Task 5 desktop screenshot's header + Overview + Threat-Model region
(the top ~800px) as `results/report/figures/dashboard_hero.png`. If cropping isn't
practical inline, take a dedicated Playwright screenshot instead: navigate to the
locally-served dashboard, resize to `1440x900`, screenshot without full-page (viewport
only) and save directly to `results/report/figures/dashboard_hero.png`.

- [ ] **Step 2: Replace the hero line**

Add back, at README.md line 11 (left empty by Task 1 Step 2):
```markdown
![Dashboard preview](results/report/figures/dashboard_hero.png)
```

- [ ] **Step 3: Verify the Quick Start block reads correctly post-poster-removal**

Confirm README.md's Quick Start block (originally lines 55-69, now missing the two
poster lines removed in Task 1) still has three commands (install, run showcase, launch
dashboard) and no dangling comment referring to a step that no longer exists.

- [ ] **Step 4: Commit**

```bash
git add README.md results/report/figures/dashboard_hero.png
git commit -m "docs: add redesigned-dashboard hero image, tidy Quick Start"
```

---

## Not part of this plan (handled in conversation afterward)

- **Site post copy** (imadlab.com's Supabase project entry `full_description`): prose
  text handed to the user directly in chat once Task 6 lands, per the design spec's
  "site post is the last step, after 1–3 land." Not a repo file — the user's entries
  are database rows, not hardcoded pages (established earlier in this project), so
  there is nothing here to commit.

## Self-review notes

- **Spec coverage:** poster retirement (Task 1), own visual identity/palette (Task 3+4
  design spec), narrative restructure problem→method→results→try-it-yourself (Task 4
  IA list = Overview→Threat Model→Results×2→Ablation→Reproduce It), DP/HE story as a
  featured pairing not hidden (Task 4 Step 6), GitHub Pages as sole artifact (poster
  gone, no new artifact type introduced) — all covered.
- **Placeholder scan:** every code step above has literal before/after code or exact
  copy text; the one deliberately-not-fully-specified area (Task 4's overall markup
  reshuffling) is bounded by an explicit list of which elements move where and which
  keep their existing internals unchanged, per this plan's Design Spec section, not a
  bare "restyle it" instruction.
- **Type/name consistency:** `PALETTE` keys (Task 3 Step 1) match the CSS variable
  names referenced in Task 4 Step 1 and the Design Spec table; chart_map keys
  `ablation_psnr`/`ablation_ssim`/`ablation_lpips` (Task 3 Step 6) match the template
  JS reference in Task 4 Step 4; `data_blob["figures"]["dp_noise_scaling"]` (Task 3
  Step 8) matches the JS reference in Task 4 Step 6.
