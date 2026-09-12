# FL-Attack Correctness & Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FL-Attack's published claims accurate (correct threat-model framing for DP/HE, synced numbers), remove dead code, and deduplicate two pairs of near-identical scripts — without rerunning any experiment (no GPU/CelebA/compute available this cycle) and without changing any code path that produced a number already in `results/report/summary.csv`.

**Architecture:** Seven independent, sequentially-ordered tasks. No new services, no new dependencies. Verification is via `python3 -m py_compile` (no torch in this environment), `--dry-run` CLI diffing for the two experiment-generator scripts, grep-based no-stale-reference checks, and one new pytest file with real numeric assertions (pure math, no torch) for the synthetic DP proof.

**Tech Stack:** Python 3 stdlib + numpy + matplotlib (already installed, verified). No torch required for anything in this plan.

## Global Constraints

- Do not rerun `run_experiment.py` or change any value in `results/report/summary.csv` except removing the `he_test` row (Task 3) — every other row must remain byte-identical, because the current DP code is correct for its tested threat model (see spec finding 1) and no other change in this plan touches a numeric path.
- Do not wire `aggregate_clipped_noisy` into `run_experiment.py`. It is documented (Task 1), not called.
- Do not touch `docs/index.html` or `docs/data.json` — both are retired by the (separate, not-yet-planned) presentation redesign; fixing the currently-published dashboard is wasted effort.
- No torch import may be added to any script that previously didn't need torch (the synthetic proof script in Task 4 must stay torch-free).
- Follow existing code style in each file (type hints where the file already uses them, docstrings matching the file's existing docstring style).

Spec reference: `docs/superpowers/specs/2026-09-12-project-rework-design.md`

---

### Task 1: Document the threat models (rename + docstrings, no behavior change)

**Files:**
- Modify: `Differential_privacy.py` → renamed to `differential_privacy.py`
- Modify: `run_experiment.py:236-247`

**Interfaces:**
- No public function signatures change. `aggregate_clipped_noisy`, `gaussian_sigma_for_dp`, `clip_gradients`, `add_gaussian_noise` keep identical signatures and behavior.

- [ ] **Step 1: Rename the file to match its import**

```bash
git mv Differential_privacy.py differential_privacy.py
```

- [ ] **Step 2: Verify nothing else references the old casing**

Run: `grep -rn "Differential_privacy" --include="*.py" --include="*.sh" --include="*.md" .`
Expected: no output (the only prior references were the file itself and the already-lowercase import in `run_experiment.py`).

- [ ] **Step 3: Document `aggregate_clipped_noisy`'s threat model in its docstring**

In `differential_privacy.py`, replace the `aggregate_clipped_noisy` docstring:

```python
def aggregate_clipped_noisy(
    client_gradients: List[List[torch.Tensor]], 
    max_norm: float, 
    sigma: float, 
    device: torch.device = None
) -> List[torch.Tensor]:
    """
    Central DP-FedAvg aggregation with per-client clipping and noise addition.

    This is NOT the mechanism used by run_experiment.py's --dp-epsilon flag.
    That flag implements *local* DP: it clips and noises one client's raw
    gradient before any adversary (including this project's gradient-inversion
    attacker) can read it — the correct mechanism when the adversary's read
    point is a single client's pre-aggregation update, which is what every
    published result in this project tests (attack_source="gradients").

    This function implements *central* DP-FedAvg instead: it clips each
    client's gradient, averages the clipped values, and adds noise scaled
    down by num_clients — a guarantee that covers only the released
    aggregate. It provides no protection to an adversary who reads any
    individual client's update before aggregation, so it must not be used to
    "fix" the local-DP path above; doing so would understate the noise
    actually needed and produce a broken privacy guarantee for that release
    point.

    This function is implemented but not evaluated anywhere in this project.
    It's the mechanism a secure-aggregation / agg_update-style experiment
    would need (see fl_system.py's capture_mode='agg_update'), which this
    project has never run — no GPU/CelebA compute was available to do so.

    Steps:
        1. Clip each client's gradients to max_norm (L2)
        2. Average clipped gradients
        3. Add calibrated Gaussian noise
    
    Args:
        client_gradients: List of gradient lists, shape [num_clients × num_params]
        max_norm: Per-client gradient clipping threshold
        sigma: Base noise scale (adjusted by num_clients)
        device: Target device for computation
    
    Returns:
        Aggregated noisy gradients
    """
```

- [ ] **Step 4: Add a threat-model comment at the DP application site in `run_experiment.py`**

Find (around line 236):

```python
    # ============================================================
    # Apply Privacy Defenses (DP and/or HE)
    # ============================================================
    if args.dp_epsilon is not None:
```

Replace with:

```python
    # ============================================================
    # Apply Privacy Defenses (DP and/or HE)
    #
    # Threat model: the attacker below reads ONE client's raw gradient
    # before any aggregation (attack_source="gradients" in every published
    # result). Local DP — clip this client's own release to dp_max_norm,
    # add Gaussian noise calibrated to that same sensitivity — is the
    # correct mechanism for that read point, and that's what this block
    # does. It is deliberately NOT the central-DP-FedAvg mechanism in
    # differential_privacy.aggregate_clipped_noisy(), which only protects a
    # released aggregate and would understate the noise needed here. See
    # that function's docstring for the unevaluated secure-aggregation
    # threat model.
    # ============================================================
    if args.dp_epsilon is not None:
```

- [ ] **Step 5: Syntax-check both edited files**

Run: `python3 -m py_compile differential_privacy.py run_experiment.py`
Expected: no output, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add differential_privacy.py Differential_privacy.py run_experiment.py
git commit -m "docs: fix filename casing, document DP threat models

Rename Differential_privacy.py -> differential_privacy.py to match its
import (was working only on case-insensitive filesystems). Document
that run_experiment.py's DP block is local DP (correct for the single-
client-gradient-leak threat model every published result tests) and
that aggregate_clipped_noisy is a different, unevaluated central-DP
mechanism for a secure-aggregation threat model this project never ran.
No behavior change."
```

---

### Task 2: Remove dead code (unused attack method, never-run layer-weight modes)

**Files:**
- Modify: `gradient_attack.py:162-208` (delete `reconstruct_image`), `gradient_attack.py:639-696` (trim `_prepare_layer_weights`)
- Modify: `run_experiment.py:121-140` (`_parse_layer_weights` valid-modes set)
- Modify: `scripts/exp_phase1.py:142-213` (`get_layer_weighting_configs`, `get_combined_best_configs`, `get_all_configs`)

**Interfaces:**
- `_prepare_layer_weights(indices, layer_weights, target_grads, param_names=None) -> List[float]` — signature unchanged; the `'early'`, `'early_linear'`, `'early_strong'`, `'early_conv'`, `'spatial'` string branches are removed, `None`/`'uniform'`/`'none'`, `'auto'`/`'auto_norm'`/`'inv_norm'`, and explicit list-of-floats branches are unchanged.
- `get_layer_weighting_configs() -> List[ExperimentConfig]` now returns 2 configs (uniform, auto) instead of 6.
- `get_combined_best_configs` is deleted; `get_all_configs()` no longer calls it.

- [ ] **Step 1: Confirm zero callers before deleting (belt and suspenders)**

Run: `grep -rn "reconstruct_image\b" --include="*.py" .`
Expected: exactly one match — the `def reconstruct_image(` line itself in `gradient_attack.py`.

- [ ] **Step 2: Delete `reconstruct_image` from `gradient_attack.py`**

Remove the entire method (lines 162-208, from `def reconstruct_image(self, captured_gradients, num_iterations=5000, lr=0.1):` through the `return dummy_data.detach(), history` line and the blank line before `def reconstruct_with_label_inference`). The class's next method, `reconstruct_with_label_inference`, becomes the first method after `__init__`.

- [ ] **Step 3: Trim `_prepare_layer_weights` to drop unevidenced modes**

In `gradient_attack.py`, find the `_prepare_layer_weights` function. Replace the four blocks for `'early'`, `'early_linear'`, `'early_strong'`, and `'early_conv'/'spatial'` (currently between the `'auto'/'auto_norm'/'inv_norm'` block and the final `return [1.0] * n`) — i.e. delete this whole span:

```python
    if mode == 'early':
        # Exponential decay: w_i = exp(-0.08 * i)
        # Normalized so mean(w) = 1
        ws = []
        for idx in indices:
            w = math.exp(-0.08 * idx)
            ws.append(w)
        # Normalize: divide by mean to get mean = 1
        mean_w = sum(ws) / len(ws) if ws else 1.0
        ws = [w / mean_w for w in ws]
        return ws
    
    if mode == 'early_linear':
        # Linear decay: w_i = 1 - i/(L-1)
        # Normalized so mean(w) = 1
        L = total_layers
        ws = []
        for idx in indices:
            w = max(1.0 - idx / max(L - 1, 1), 0.01)  # Avoid zero weight
            ws.append(w)
        # Normalize: divide by mean to get mean = 1
        mean_w = sum(ws) / len(ws) if ws else 1.0
        ws = [w / mean_w for w in ws]
        return ws
    
    if mode == 'early_strong':
        # Strong exponential decay: w_i = exp(-0.20 * i)
        # Normalized so mean(w) = 1
        ws = []
        for idx in indices:
            w = math.exp(-0.20 * idx)
            ws.append(w)
        # Normalize: divide by mean to get mean = 1
        mean_w = sum(ws) / len(ws) if ws else 1.0
        ws = [w / mean_w for w in ws]
        return ws
    
    if mode in ('early_conv', 'spatial'):
        # Upweight early convolutional layers based on parameter names
        ws = []
        for i, idx in enumerate(indices):
            name = param_names[idx] if param_names and idx < len(param_names) else ''
            is_conv = 'conv' in name.lower() or 'features.0' in name.lower() or 'features.3' in name.lower()
            is_early = idx < total_layers // 2
            if is_conv and is_early:
                w = 3.0
            elif is_conv:
                w = 1.5
            elif is_early:
                w = 1.2
            else:
                w = 0.5
            ws.append(w)
        s = sum(ws) + 1e-8
        ws = [w * (n / s) for w in ws]
        return ws
    
    return [1.0] * n
```

Replace it with just:

```python
    return [1.0] * n
```

Also update the function's docstring to drop the removed modes from its list (keep only `None`/`'uniform'`/`'none'`, `'auto'`/`'auto_norm'`/`'inv_norm'`, and explicit list-of-floats):

```python
    """
    Prepare layer weights for gradient matching.
    
    Supports multiple weighting strategies:
    - None / 'uniform' / 'none': Equal weights for all layers
    - 'auto' / 'auto_norm' / 'inv_norm': Inverse of gradient norm (normalizes contribution)
    - List of floats: Explicit weights per layer

    Modes not listed here ('early', 'early_linear', 'early_strong',
    'early_conv', 'spatial') were removed: no config.json in results/ ever
    used a non-null layer_weights value, so none of these modes have an
    evidenced result behind them.
    """
```

- [ ] **Step 4: Update `run_experiment.py`'s valid-modes allowlist to match**

Find:

```python
    val = str(arg_val).strip().lower()
    valid_modes = {
        'auto', 'auto_norm', 'inv_norm', 'early', 'early_linear',
        'early_strong', 'early_conv', 'spatial', 'uniform', 'none'
    }
```

Replace with:

```python
    val = str(arg_val).strip().lower()
    valid_modes = {
        'auto', 'auto_norm', 'inv_norm', 'uniform', 'none'
    }
```

- [ ] **Step 5: Remove the now-dead layer-weight strategies from `scripts/exp_phase1.py`**

Replace `get_layer_weighting_configs`:

```python
def get_layer_weighting_configs() -> List[ExperimentConfig]:
    """
    Layer Weighting Ablation
    
    Strategies tested:
    - uniform: Equal weight for all layers (baseline)
    - auto: Inverse gradient norm (normalize contribution)

    Goal: Improve spatial coherence and reduce high-frequency noise
    by focusing on early layers that capture low-frequency structure.
    """
    configs = []
    
    strategies = [
        ("uniform", "Uniform weighting (baseline)"),
        ("auto", "Auto inverse-norm weighting"),
    ]
    
    for i, (strategy, desc) in enumerate(strategies):
        lw = None if strategy == "uniform" else strategy
        configs.append(ExperimentConfig(
            name=f"p1_layer_{strategy}",
            description=f"Layer weighting: {desc}",
            category="layer_weighting",
            tv_weight=1e-6,  # Fixed at baseline best
            layer_weights=lw,
            attack_iterations=3000,
            attack_restarts=5,
            priority=20 + i,
        ))
    
    return configs
```

Delete `get_combined_best_configs` entirely (its four configs all used `early`/`early_linear`/`early_strong`/`early_conv`, none of which exist anymore).

In `get_all_configs`, remove the call to it:

```python
def get_all_configs() -> List[ExperimentConfig]:
    """Get all Phase 1 experiment configurations."""
    configs = []
    configs.append(get_baseline_reference())
    configs.extend(get_tv_sweep_configs())
    configs.extend(get_layer_weighting_configs())
    return sorted(configs, key=lambda c: c.priority)
```

In `main()`'s argparse, `--mode combined` becomes dead — remove it from the choices list and the `elif` branch that calls `get_combined_best_configs()`:

```python
    parser.add_argument("--mode", type=str, default="all",
                       choices=["all", "tv-sweep", "layer-ablation", "baseline"],
                       help="Which experiments to run")
```

```python
    if args.mode == "tv-sweep":
        configs = get_tv_sweep_configs()
    elif args.mode == "layer-ablation":
        configs = get_layer_weighting_configs()
    elif args.mode == "baseline":
        configs = [get_baseline_reference()]
    else:  # all
        configs = get_all_configs()
```

- [ ] **Step 6: Syntax-check all four edited files**

Run: `python3 -m py_compile gradient_attack.py run_experiment.py scripts/exp_phase1.py`
Expected: no output, exit code 0.

- [ ] **Step 7: Verify the dry-run path still produces valid commands**

Run: `python3 scripts/exp_phase1.py --mode layer-ablation --dry-run`
Expected: prints exactly 2 experiments (`p1_layer_uniform`, `p1_layer_auto`), no Python traceback.

Run: `python3 scripts/exp_phase1.py --mode all --dry-run`
Expected: no traceback, no experiment named `p1_combined_*`.

- [ ] **Step 8: Commit**

```bash
git add gradient_attack.py run_experiment.py scripts/exp_phase1.py
git commit -m "refactor: remove dead code with no evidenced result

Delete GradientInversionAttack.reconstruct_image (zero callers anywhere)
and the layer_weights modes early/early_linear/early_strong/early_conv/
spatial from _prepare_layer_weights, _parse_layer_weights's allowlist,
and exp_phase1.py's config generators. Verified via every config.json
under results/: layer_weights is null in all of them, so none of these
modes ever produced a published result. Uniform and auto/auto_norm/
inv_norm weighting, and explicit per-layer float weights, are unchanged."
```

---

### Task 3: Remove the `he_test` debug row from the results data

**Files:**
- Modify: `results/report/summary.csv`
- Delete: `results/defenses/he_test/` (directory)

**Interfaces:** None (data-only change).

- [ ] **Step 1: Confirm this row is genuinely debug output, not a designed experiment**

Run: `grep -n "he_test" scripts/analyze_defenses.py`
Expected: no output — `he_test` is not in that script's `conditions` allowlist (`baseline`, `dp_eps8`, `dp_eps1`, `dp_eps01`, `he`, `dp_he`), confirming it was never part of the intended defense comparison.

- [ ] **Step 2: Remove the row from `summary.csv`**

Run: `grep -v ",he_test," results/report/summary.csv > /tmp/summary_filtered.csv && mv /tmp/summary_filtered.csv results/report/summary.csv`

- [ ] **Step 3: Verify exactly one row was removed and the header survived**

Run: `wc -l results/report/summary.csv` before/after should differ by exactly 1 line, and:
Run: `head -1 results/report/summary.csv`
Expected: `MSE,PSNR,SSIM,LPIPS,LabelMatch,group,setting,path,ablation_type` (header intact).
Run: `grep -c "he_test" results/report/summary.csv`
Expected: `0`.

- [ ] **Step 4: Delete the stray results directory**

```bash
rm -rf results/defenses/he_test
```

- [ ] **Step 5: Commit**

```bash
git add -A results/report/summary.csv results/defenses/he_test
git commit -m "chore: remove he_test debug run from published results

he_test was never part of the defense comparison (absent from
analyze_defenses.py's condition list) and has an empty LPIPS value.
It was leaking into results/report/summary.csv and, downstream, into
docs/data.json's charts as an unlabeled partial result. docs/ itself
is being replaced by the presentation redesign, so it isn't touched
here — this just stops the debug row from being a valid input to
whatever regenerates the dashboard next."
```

---

### Task 4: Synthetic proof of the DP noise/dimensionality saturation

**Files:**
- Create: `scripts/dp_noise_scaling_proof.py`
- Create: `scripts/test_dp_noise_scaling_proof.py`

**Interfaces:**
- Produces: `gaussian_sigma(epsilon: float, delta: float, max_norm: float) -> float`
- Produces: `expected_noise_l2_norm(sigma: float, d: int) -> float`
- Produces: `main() -> None` — writes `results/report/figures/dp_noise_scaling.png`

- [ ] **Step 1: Write the failing test**

Create `scripts/test_dp_noise_scaling_proof.py`:

```python
"""Tests for the DP noise-vs-dimensionality synthetic proof (no torch needed)."""
import math

from dp_noise_scaling_proof import gaussian_sigma, expected_noise_l2_norm


def test_gaussian_sigma_matches_analytic_formula():
    # sigma = max_norm * sqrt(2 ln(1.25/delta)) / epsilon
    sigma = gaussian_sigma(epsilon=8.0, delta=1e-5, max_norm=1.0)
    expected = math.sqrt(2 * math.log(1.25 / 1e-5)) / 8.0
    assert abs(sigma - expected) < 1e-9


def test_gaussian_sigma_at_tested_epsilons():
    # The three epsilons actually used in results/report/summary.csv.
    sigma_8 = gaussian_sigma(epsilon=8.0, delta=1e-5, max_norm=1.0)
    sigma_1 = gaussian_sigma(epsilon=1.0, delta=1e-5, max_norm=1.0)
    sigma_01 = gaussian_sigma(epsilon=0.1, delta=1e-5, max_norm=1.0)
    assert 0.60 < sigma_8 < 0.61
    assert 4.8 < sigma_1 < 4.9
    assert 48.0 < sigma_01 < 49.0


def test_noise_norm_scales_with_sqrt_d():
    sigma = 1.0
    norm_at_1 = expected_noise_l2_norm(sigma, d=1)
    norm_at_100 = expected_noise_l2_norm(sigma, d=100)
    assert abs(norm_at_1 - 1.0) < 1e-9
    assert abs(norm_at_100 - 10.0) < 1e-9


def test_model_dimensionality_dominates_signal_at_every_tested_epsilon():
    # d = 8,760,962 is SimpleCNN's actual parameter count (fl_system.py).
    d = 8_760_962
    max_norm = 1.0
    for epsilon in (8.0, 1.0, 0.1):
        sigma = gaussian_sigma(epsilon=epsilon, delta=1e-5, max_norm=max_norm)
        noise_norm = expected_noise_l2_norm(sigma, d)
        # Even the weakest tested epsilon (8.0) buries the clipped signal.
        assert noise_norm > 1000 * max_norm
```

- [ ] **Step 2: Run it to verify it fails on import**

Run: `cd scripts && python3 -m pytest test_dp_noise_scaling_proof.py -v`
Expected: `ModuleNotFoundError: No module named 'dp_noise_scaling_proof'` (or ImportError) — the module doesn't exist yet.

- [ ] **Step 3: Write `scripts/dp_noise_scaling_proof.py`**

```python
#!/usr/bin/env python3
"""
Synthetic proof: why the DP epsilon sweep in results/report/summary.csv is flat.

No CelebA, no GPU, no torch. Reproduces the analytic Gaussian mechanism
formula from differential_privacy.gaussian_sigma_for_dp (duplicated here,
not imported, so this script has zero dependency on torch being installed)
and shows that at this project's actual model dimensionality, injected
noise dwarfs the clipped signal at every epsilon that was tested — this is
an experimental-design finding, not a bug in the DP mechanism (see
differential_privacy.aggregate_clipped_noisy's docstring and
run_experiment.py's DP block comment for the full threat-model discussion).
"""
import math

import matplotlib.pyplot as plt
import numpy as np

# SimpleCNN's actual parameter count (fl_system.py: 3 conv blocks + 2 linear layers).
MODEL_PARAM_COUNT = 8_760_962
DELTA = 1e-5
MAX_NORM = 1.0
TESTED_EPSILONS = (8.0, 1.0, 0.1)


def gaussian_sigma(epsilon: float, delta: float, max_norm: float) -> float:
    """Analytic Gaussian mechanism noise std, same formula as
    differential_privacy.gaussian_sigma_for_dp."""
    return max_norm * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def expected_noise_l2_norm(sigma: float, d: int) -> float:
    """Expected L2 norm of a d-dimensional i.i.d. N(0, sigma^2) noise vector."""
    return sigma * math.sqrt(d)


def main() -> None:
    d_values = np.logspace(1, 8, 200)

    fig, ax = plt.subplots(figsize=(9, 6))

    for epsilon in TESTED_EPSILONS:
        sigma = gaussian_sigma(epsilon, DELTA, MAX_NORM)
        noise_norms = [expected_noise_l2_norm(sigma, d) for d in d_values]
        ax.plot(d_values, noise_norms, label=f"ε={epsilon} (σ={sigma:.3f})", linewidth=2)

    ax.axhline(MAX_NORM, color="black", linestyle="--", linewidth=1.5,
               label=f"clipped signal norm (max_norm={MAX_NORM})")
    ax.axvline(MODEL_PARAM_COUNT, color="gray", linestyle=":", linewidth=1.5,
               label=f"this model's d = {MODEL_PARAM_COUNT:,}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameter count (d)")
    ax.set_ylabel("expected noise L2 norm")
    ax.set_title("Per-coordinate DP noise vs. clipped signal, by model size")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = "results/report/figures/dp_noise_scaling.png"
    fig.savefig(out_path, dpi=150)
    print(f"[INFO] Wrote {out_path}")

    print("\nAt this model's dimensionality (d={:,}):".format(MODEL_PARAM_COUNT))
    for epsilon in TESTED_EPSILONS:
        sigma = gaussian_sigma(epsilon, DELTA, MAX_NORM)
        noise_norm = expected_noise_l2_norm(sigma, MODEL_PARAM_COUNT)
        print(f"  epsilon={epsilon:<5} sigma={sigma:>10.4f}  "
              f"noise_norm={noise_norm:>12.1f}  (signal capped at {MAX_NORM})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd scripts && python3 -m pytest test_dp_noise_scaling_proof.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the script itself and verify the figure is produced**

Run: `mkdir -p results/report/figures && python3 scripts/dp_noise_scaling_proof.py`
Expected: prints the three epsilon/sigma/noise_norm lines (all `noise_norm` values in the thousands or more) and `[INFO] Wrote results/report/figures/dp_noise_scaling.png`.
Run: `test -f results/report/figures/dp_noise_scaling.png && echo OK`
Expected: `OK`.

- [ ] **Step 6: Commit**

```bash
git add scripts/dp_noise_scaling_proof.py scripts/test_dp_noise_scaling_proof.py results/report/figures/dp_noise_scaling.png
git commit -m "feat: add synthetic proof of DP noise/dimensionality saturation

New evidence for spec finding 1: per-coordinate Gaussian noise has L2
norm ~sigma*sqrt(d); at this model's d=8,760,962 parameters, even the
weakest tested epsilon (8.0) produces noise ~1800x the clipped signal
bound. Dependency-light (numpy+matplotlib only, no torch) so it runs
standalone as documentation of why the tested epsilon range couldn't
show graduated protection, independent of any GPU/CelebA availability."
```

---

### Task 5: Merge the two poster scripts behind a `--theme` flag

**Files:**
- Modify: `scripts/make_poster.py`
- Delete: `scripts/make_poster_light.py`

**Interfaces:**
- Produces: `apply_theme(name: str) -> None` — sets all theme-dependent module globals.
- New CLI flag: `--theme {dark,light}` (default `dark`), added to `make_poster.py`'s argparse.
- Output filenames become theme-dependent: `poster_4k.png`/`.pdf` for `dark`, `poster_4k_light.png`/`.pdf` for `light` (matches the filenames already present in `results/report/`, so existing README/site image links keep working unchanged).

- [ ] **Step 1: Replace the static color block with a `THEMES` dict**

In `scripts/make_poster.py`, replace the existing block (currently starting at `# Colors (refined dark theme with high contrast)` and running through the `COL_COLORS = [...]` list) with:

```python
# Colors: two complete palettes, selected at runtime by apply_theme().
THEMES = {
    "dark": dict(
        BG_COLOR=(22, 27, 38), CARD_BG=(38, 44, 58), CARD_BORDER=(60, 75, 100),
        HEADER_BG=(15, 18, 28), TEXT_WHITE=(255, 255, 255), TEXT_DARK=(255, 255, 255),
        TEXT_LIGHT=(220, 225, 235), TEXT_MUTED=(150, 160, 180),
        ACCENT_BLUE=(80, 150, 255), ACCENT_GREEN=(100, 200, 120),
        ACCENT_ORANGE=(255, 170, 50), ACCENT_RED=(255, 90, 90),
        ACCENT_PURPLE=(180, 100, 220), ACCENT_GOLD=(255, 215, 0),
        COL_COLORS=[(70, 140, 255), (255, 160, 40), (80, 200, 120), (180, 100, 220)],
        HEADER_BOX_FILL=(30, 50, 80),
        SUBTITLE_TEXT=(220, 225, 235),
        CAPTION_TEXT_ON_CARD=(255, 255, 255),
        PANEL_TITLE_TEXT=(255, 255, 255),
        DP_DELTA_BOX_FILL=(30, 50, 45),
        DPHE_DELTA_BOX_FILL=(40, 30, 55),
        DPHE_IMPACT_LABEL_COLOR=(255, 215, 0),
        METRICS_PANEL_BG=(32, 38, 50),
        CARD_BG_ALT=(32, 38, 50),
        SECTION_LABEL_COLOR=(255, 215, 0),
        CHART_TEXT_COLOR="white",
        CHART_EDGE_COLOR="white",
        CHART_SPINE_COLOR="#465673",
        POSTER_FILENAME="poster_4k",
    ),
    "light": dict(
        BG_COLOR=(245, 247, 250), CARD_BG=(255, 255, 255), CARD_BORDER=(200, 210, 225),
        HEADER_BG=(30, 60, 110), TEXT_WHITE=(255, 255, 255), TEXT_DARK=(30, 35, 45),
        TEXT_LIGHT=(60, 70, 85), TEXT_MUTED=(120, 130, 150),
        ACCENT_BLUE=(40, 100, 200), ACCENT_GREEN=(30, 150, 80),
        ACCENT_ORANGE=(220, 130, 20), ACCENT_RED=(200, 50, 50),
        ACCENT_PURPLE=(130, 60, 180), ACCENT_GOLD=(180, 140, 0),
        COL_COLORS=[(50, 120, 220), (230, 140, 30), (40, 160, 90), (140, 70, 190)],
        HEADER_BOX_FILL=(240, 230, 200),
        SUBTITLE_TEXT=(255, 255, 255),
        CAPTION_TEXT_ON_CARD=(30, 35, 45),
        PANEL_TITLE_TEXT=(40, 100, 200),
        DP_DELTA_BOX_FILL=(230, 245, 235),
        DPHE_DELTA_BOX_FILL=(240, 235, 250),
        DPHE_IMPACT_LABEL_COLOR=(130, 60, 180),
        METRICS_PANEL_BG=(235, 240, 248),
        CARD_BG_ALT=(250, 252, 255),
        SECTION_LABEL_COLOR=(220, 130, 20),
        CHART_TEXT_COLOR="#1e232d",
        CHART_EDGE_COLOR="#333333",
        CHART_SPINE_COLOR="#c8d2e1",
        POSTER_FILENAME="poster_4k_light",
    ),
}


def apply_theme(name: str) -> None:
    """Set all theme-dependent module-level constants. Must be called
    before any drawing happens (i.e. at the top of main(), after argparse)."""
    if name not in THEMES:
        raise ValueError(f"Unknown theme '{name}', expected one of {list(THEMES)}")
    globals().update(THEMES[name])
```

- [ ] **Step 2: Replace the six theme-inconsistent call sites**

Each pair below is (old snippet in `make_poster.py` → new snippet). Apply all six:

1.
```python
        draw_rounded_rect(self.draw, (result_box_x, 40, CANVAS_WIDTH - OUTER_MARGIN, 100),
                          radius=10, fill=(30, 50, 80), outline=ACCENT_GOLD, width=3)
```
→
```python
        draw_rounded_rect(self.draw, (result_box_x, 40, CANVAS_WIDTH - OUTER_MARGIN, 100),
                          radius=10, fill=HEADER_BOX_FILL, outline=ACCENT_GOLD, width=3)
```

2.
```python
        self.draw.text((OUTER_MARGIN + 25, 120), subtitle, fill=TEXT_LIGHT, font=self.font_subheader)
```
→
```python
        self.draw.text((OUTER_MARGIN + 25, 120), subtitle, fill=SUBTITLE_TEXT, font=self.font_subheader)
```

3. Every `fill=TEXT_WHITE` on the DP-column, DP+HE-column, and metrics-panel caption lines (the LPIPS/SSIM/PSNR lines, the "dPSNR"/"dLPIPS" lines, the metrics-panel per-metric lines, and the "BEST ATTACK SETTINGS" list lines) becomes `fill=CAPTION_TEXT_ON_CARD`. There are 9 such call sites; each looks like:
```python
            self.draw.text((mx, my), f"LPIPS: {lpips:.3f}", fill=TEXT_WHITE, font=self.font_caption)
```
→
```python
            self.draw.text((mx, my), f"LPIPS: {lpips:.3f}", fill=CAPTION_TEXT_ON_CARD, font=self.font_caption)
```
(Same substitution — `fill=TEXT_WHITE` → `fill=CAPTION_TEXT_ON_CARD` — applies to the `SSIM`, `PSNR`, `LPIPS: ... SSIM: ...` (x2, one per column), `dPSNR`, `dLPIPS` (x2 pairs), the per-metric line inside `_draw_metrics_panel`, and the `f"  {name}: {psnr:.1f} dB PSNR"` list line. Do not change the panel *title* text — that's substitution 4 below — only the metric value/caption lines.)

4.
```python
        self.draw.text((x + 15, y + 12), title, fill=TEXT_WHITE, font=self.font_caption)
```
→
```python
        self.draw.text((x + 15, y + 12), title, fill=PANEL_TITLE_TEXT, font=self.font_caption)
```

5. DP delta box:
```python
        draw_rounded_rect(self.draw, (x, current_y, x + w, current_y + delta_box_h),
                          radius=10, fill=(30, 50, 45), outline=ACCENT_GREEN, width=3)
```
→
```python
        draw_rounded_rect(self.draw, (x, current_y, x + w, current_y + delta_box_h),
                          radius=10, fill=DP_DELTA_BOX_FILL, outline=ACCENT_GREEN, width=3)
```

6. DP+HE delta box and its "IMPACT:" label:
```python
        draw_rounded_rect(self.draw, (x, current_y, x + w, current_y + delta_box_h),
                          radius=10, fill=(40, 30, 55), outline=ACCENT_PURPLE, width=3)

        dy = current_y + 25
        self.draw.text((x + 25, dy), "IMPACT:", fill=ACCENT_GOLD, font=self.font_body)
```
→
```python
        draw_rounded_rect(self.draw, (x, current_y, x + w, current_y + delta_box_h),
                          radius=10, fill=DPHE_DELTA_BOX_FILL, outline=ACCENT_PURPLE, width=3)

        dy = current_y + 25
        self.draw.text((x + 25, dy), "IMPACT:", fill=DPHE_IMPACT_LABEL_COLOR, font=self.font_body)
```

7. Metrics panel background:
```python
        draw_rounded_rect(self.draw, (x, y, x + w, y + panel_h),
                          radius=8, fill=(32, 38, 50), outline=CARD_BORDER, width=1)
```
→
```python
        draw_rounded_rect(self.draw, (x, y, x + w, y + panel_h),
                          radius=8, fill=METRICS_PANEL_BG, outline=CARD_BORDER, width=1)
```

8. Chart card and right text-card backgrounds (two call sites, both `fill=(32, 38, 50)` → `fill=CARD_BG_ALT`):
```python
            draw_rounded_rect(self.draw, (card_x, content_y, card_x + single_chart_w, content_y + chart_h),
                              radius=8, fill=(32, 38, 50), outline=CARD_BORDER, width=1)
```
→
```python
            draw_rounded_rect(self.draw, (card_x, content_y, card_x + single_chart_w, content_y + chart_h),
                              radius=8, fill=CARD_BG_ALT, outline=CARD_BORDER, width=1)
```
and
```python
        draw_rounded_rect(self.draw, (right_x, content_y, right_x + right_card_w, content_y + right_card_h),
                          radius=10, fill=(32, 38, 50), outline=ACCENT_BLUE, width=2)
```
→
```python
        draw_rounded_rect(self.draw, (right_x, content_y, right_x + right_card_w, content_y + right_card_h),
                          radius=10, fill=CARD_BG_ALT, outline=ACCENT_BLUE, width=2)
```

9. "BEST ATTACK SETTINGS" section label:
```python
        self.draw.text((text_x, text_y), "BEST ATTACK SETTINGS", fill=ACCENT_GOLD, font=self.font_body)
```
→
```python
        self.draw.text((text_x, text_y), "BEST ATTACK SETTINGS", fill=SECTION_LABEL_COLOR, font=self.font_body)
```

10. Matplotlib chart styling (edgecolor, title/tick color, spine color):
```python
        ax.bar(x, means, yerr=stds, capsize=3, color=color, alpha=0.85,
               edgecolor='white', linewidth=0.5)
        ax.set_title(title, fontsize=14, color='white', fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9, color='white')
        ax.tick_params(colors='white')

        for spine in ax.spines.values():
            spine.set_color('#465673')
```
→
```python
        ax.bar(x, means, yerr=stds, capsize=3, color=color, alpha=0.85,
               edgecolor=CHART_EDGE_COLOR, linewidth=0.5)
        ax.set_title(title, fontsize=14, color=CHART_TEXT_COLOR, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9, color=CHART_TEXT_COLOR)
        ax.tick_params(colors=CHART_TEXT_COLOR)

        for spine in ax.spines.values():
            spine.set_color(CHART_SPINE_COLOR)
```

- [ ] **Step 3: Make the output filenames theme-dependent**

Find:
```python
    png_path = REPORT_DIR / 'poster_4k.png'
    poster.save(png_path, 'PNG', optimize=True)
    print(f"      Saved: {png_path}")

    # Try to save PDF
    pdf_path = REPORT_DIR / 'poster_4k.pdf'
```
Replace with:
```python
    png_path = REPORT_DIR / f'{POSTER_FILENAME}.png'
    poster.save(png_path, 'PNG', optimize=True)
    print(f"      Saved: {png_path}")

    # Try to save PDF
    pdf_path = REPORT_DIR / f'{POSTER_FILENAME}.pdf'
```

- [ ] **Step 4: Add the `--theme` CLI flag and call `apply_theme()` first thing in `main()`**

Find `main()`'s argparse setup and add:
```python
    parser.add_argument('--theme', choices=['dark', 'light'], default='dark',
                         help='Color theme for the generated poster')
```
Immediately after `args = parser.parse_args()`, before any other logic:
```python
    apply_theme(args.theme)
```

- [ ] **Step 5: Delete the now-redundant light script**

```bash
git rm scripts/make_poster_light.py
```

- [ ] **Step 6: Syntax-check and smoke-test both themes**

Run: `python3 -m py_compile scripts/make_poster.py`
Expected: no output.

Run: `python3 -c "
import sys; sys.path.insert(0, 'scripts')
import make_poster
make_poster.apply_theme('dark')
assert make_poster.BG_COLOR == (22, 27, 38)
assert make_poster.POSTER_FILENAME == 'poster_4k'
make_poster.apply_theme('light')
assert make_poster.BG_COLOR == (245, 247, 250)
assert make_poster.POSTER_FILENAME == 'poster_4k_light'
print('OK')
"`
Expected: `OK` (this only checks constant-swapping; it does not run `main()`, which needs `results/report/summary.csv` data and a display-capable matplotlib backend — out of scope for this syntax/wiring check).

- [ ] **Step 7: Commit**

```bash
git add scripts/make_poster.py
git commit -m "refactor: merge make_poster_light.py into make_poster.py via --theme

The two scripts were ~85% identical (diff: 153 of ~1050 lines), differing
only in a hardcoded color palette plus several call sites that had drifted
to reference the wrong named constant or a bare literal instead of a
theme token. Consolidated into one THEMES dict + apply_theme(), selected
by a new --theme {dark,light} flag (default dark, preserving today's
default output filename poster_4k.png; --theme light preserves the
existing poster_4k_light.png naming)."
```

---

### Task 6: Extract shared subprocess-runner boilerplate from exp_base.py / exp_phase1.py

**Files:**
- Create: `scripts/exp_common.py`
- Modify: `scripts/exp_base.py`
- Modify: `scripts/exp_phase1.py`

**Interfaces:**
- Produces: `run_experiment_subprocess(cmd: List[str], dry_run: bool) -> Tuple[str, Optional[str]]` — returns `(status, error)` where `status` is `"skipped"`, `"success"`, `"failed"`, or `"error"`, and `error` is `None` on success/skip.
- Produces: `parse_metrics_file(path: Path) -> Dict[str, Any]` — parses a `metrics.txt` (`key: value` lines), converting each value to `int` if it has no `.`, else `float`, else leaving it as a string. This is `exp_phase1.py`'s existing (more general) behavior; `exp_base.py` currently only tries `float`, which is safe to widen since every metric it has ever recorded (PSNR/SSIM/MSE/LabelMatch) is written with a decimal point.

Only the mechanical subprocess-invocation/error-handling and metrics-file-parsing are extracted. Each script's own `ExperimentConfig` dataclass, `to_flags()`, per-experiment metric enrichment (`category`, `tv_weight`, `layer_weights` fields that `exp_phase1.py` adds and `exp_base.py` doesn't), and report generation stay where they are — they differ enough between the two scripts that forcing a shared abstraction would risk changing behavior neither script can currently be executed end-to-end to re-verify (no compute this cycle; `--dry-run` verification below only checks the parts that don't need training).

- [ ] **Step 1: Create `scripts/exp_common.py`**

```python
"""Shared subprocess-execution helpers for the exp_*.py experiment scripts.

Kept deliberately small: only the parts of run_experiment()/parse_metrics()
that were byte-for-byte identical (or safely unifiable) between exp_base.py
and exp_phase1.py live here. Each script's own config dataclass, flag
building, and report generation stay separate — they differ enough that
forcing a shared abstraction isn't worth it, especially since neither
script can be run end-to-end in an environment without GPU/CelebA to
re-verify the merge didn't change behavior.
"""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def run_experiment_subprocess(cmd: List[str], dry_run: bool) -> Tuple[str, Optional[str]]:
    """Run one experiment's run_experiment.py invocation.

    Returns (status, error) where status is one of
    "skipped" / "success" / "failed" / "error", and error is None unless
    status is "failed" or "error".
    """
    if dry_run:
        print("[DRY RUN] Skipping execution")
        return "skipped", None

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        return "success", None
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Experiment failed: {e}")
        return "failed", str(e)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return "error", str(e)


def parse_metrics_file(path: Path) -> Dict[str, Any]:
    """Parse a metrics.txt file ('key: value' per line) into a dict,
    converting each value to int (no '.') or float where possible."""
    metrics: Dict[str, Any] = {}
    if not path.exists():
        return metrics
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()
            try:
                metrics[key] = int(val) if "." not in val else float(val)
            except ValueError:
                metrics[key] = val
    return metrics


def build_experiment_command(config_name: str, exp_dir: Path,
                              base_flags: List[str], config_flags: List[str]) -> List[str]:
    """Build the run_experiment.py subprocess command shared by both scripts."""
    return (
        [sys.executable, "run_experiment.py", "--out-dir", str(exp_dir), "--save-config"]
        + base_flags
        + config_flags
    )
```

- [ ] **Step 2: Wire `exp_base.py`'s `run_experiment()` to use the shared helpers**

Replace:

```python
def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    base_flags: List[str],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment and return metrics."""
    
    exp_dir = output_dir / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "run_experiment.py",
        "--out-dir", str(exp_dir),
        "--save-config",
    ] + base_flags + config.to_flags()
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'='*70}")
    print(f"Description: {config.description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    if dry_run:
        print("[DRY RUN] Skipping execution")
        return {"name": config.name, "status": "skipped"}
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Let output stream to terminal
        )
        metrics = parse_metrics(exp_dir / "metrics.txt")
        metrics["name"] = config.name
        metrics["status"] = "success"
        metrics["output_dir"] = str(exp_dir)
        
        # Save config alongside results
        with open(exp_dir / "experiment_config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        return metrics
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Experiment failed: {e}")
        return {"name": config.name, "status": "failed", "error": str(e)}
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return {"name": config.name, "status": "error", "error": str(e)}


def parse_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Parse metrics.txt into a dictionary."""
    metrics = {}
    if not metrics_path.exists():
        return metrics
    
    for line in metrics_path.read_text().strip().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    
    return metrics
```

with:

```python
def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    base_flags: List[str],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment and return metrics."""

    exp_dir = output_dir / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_experiment_command(config.name, exp_dir, base_flags, config.to_flags())

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'='*70}")
    print(f"Description: {config.description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    status, error = run_experiment_subprocess(cmd, dry_run)
    if status == "skipped":
        return {"name": config.name, "status": "skipped"}
    if status != "success":
        return {"name": config.name, "status": status, "error": error}

    metrics = parse_metrics_file(exp_dir / "metrics.txt")
    metrics["name"] = config.name
    metrics["status"] = "success"
    metrics["output_dir"] = str(exp_dir)

    with open(exp_dir / "experiment_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    return metrics
```

Add the import near the top of `scripts/exp_base.py` (alongside the existing imports):

```python
from exp_common import build_experiment_command, parse_metrics_file, run_experiment_subprocess
```

- [ ] **Step 3: Wire `exp_phase1.py`'s `run_experiment()` to use the shared helpers**

Replace:

```python
def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    base_flags: List[str],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment and return metrics."""
    
    exp_dir = output_dir / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "run_experiment.py",
        "--out-dir", str(exp_dir),
        "--save-config",
    ] + base_flags + config.to_flags()
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'='*70}")
    print(f"Description: {config.description}")
    print(f"Category: {config.category}")
    print(f"Key settings: TV={config.tv_weight:.0e}, layers={config.layer_weights or 'uniform'}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    if dry_run:
        print("[DRY RUN] Skipping execution")
        return {"name": config.name, "category": config.category, "status": "skipped"}
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
        )
        metrics = parse_metrics(exp_dir / "metrics.txt")
        metrics["name"] = config.name
        metrics["category"] = config.category
        metrics["status"] = "success"
        metrics["output_dir"] = str(exp_dir)
        metrics["tv_weight"] = config.tv_weight
        metrics["layer_weights"] = config.layer_weights or "uniform"
        
        # Save config alongside results
        with open(exp_dir / "experiment_config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        return metrics
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Experiment failed: {e}")
        return {"name": config.name, "category": config.category, "status": "failed", "error": str(e)}
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return {"name": config.name, "category": config.category, "status": "error", "error": str(e)}


def parse_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Parse metrics.txt into a dictionary."""
    metrics = {}
    if not metrics_path.exists():
        return metrics
    with open(metrics_path) as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                try:
                    if "." in val:
                        metrics[key] = float(val)
                    else:
                        metrics[key] = int(val)
                except ValueError:
                    metrics[key] = val
    return metrics
```

with:

```python
def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    base_flags: List[str],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment and return metrics."""

    exp_dir = output_dir / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_experiment_command(config.name, exp_dir, base_flags, config.to_flags())

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'='*70}")
    print(f"Description: {config.description}")
    print(f"Category: {config.category}")
    print(f"Key settings: TV={config.tv_weight:.0e}, layers={config.layer_weights or 'uniform'}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    status, error = run_experiment_subprocess(cmd, dry_run)
    if status == "skipped":
        return {"name": config.name, "category": config.category, "status": "skipped"}
    if status != "success":
        return {"name": config.name, "category": config.category, "status": status, "error": error}

    metrics = parse_metrics_file(exp_dir / "metrics.txt")
    metrics["name"] = config.name
    metrics["category"] = config.category
    metrics["status"] = "success"
    metrics["output_dir"] = str(exp_dir)
    metrics["tv_weight"] = config.tv_weight
    metrics["layer_weights"] = config.layer_weights or "uniform"

    with open(exp_dir / "experiment_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    return metrics
```

Add the import near the top of `scripts/exp_phase1.py`:

```python
from exp_common import build_experiment_command, parse_metrics_file, run_experiment_subprocess
```

- [ ] **Step 4: Syntax-check all three files**

Run: `python3 -m py_compile scripts/exp_common.py scripts/exp_base.py scripts/exp_phase1.py`
Expected: no output.

- [ ] **Step 5: Verify `--dry-run` output is unchanged for both scripts**

Run and save output before this task's changes were made is not possible retroactively, so instead verify structurally:

Run: `python3 scripts/exp_base.py --dry-run --mode full 2>&1 | grep -c "^EXPERIMENT:"`
Expected: `7` (quick validation + showcase + 5 ablations, matching `get_quick_validation_config` + `get_showcase_config` + `get_ablation_configs`).

Run: `python3 scripts/exp_phase1.py --mode all --dry-run 2>&1 | grep -c "^EXPERIMENT:"`
Expected: `6` (1 baseline + 3 TV sweep + 2 layer-weighting, per Task 2's trim — `get_combined_best_configs` no longer contributes).

- [ ] **Step 6: Commit**

```bash
git add scripts/exp_common.py scripts/exp_base.py scripts/exp_phase1.py
git commit -m "refactor: extract shared subprocess-runner boilerplate to exp_common.py

exp_base.py and exp_phase1.py had near-identical subprocess-invocation,
error-handling, and metrics-file-parsing code around genuinely different
experiment configs and reports. Extracted only the mechanical, safely-
unifiable parts (run_experiment_subprocess, parse_metrics_file,
build_experiment_command) into scripts/exp_common.py; each script keeps
its own ExperimentConfig, to_flags(), and report generation, since those
differ enough (category field, tv_weight/layer_weights metric enrichment)
that forcing them together isn't worth the risk without being able to
run either script end-to-end to re-verify (no GPU/CelebA this cycle)."
```

---

### Task 7: Sync README to the verified numbers and the corrected DP/HE story

**Files:**
- Modify: `README.md`

**Interfaces:** None (documentation only).

- [ ] **Step 1: Replace the Key Findings section**

Find (current lines 13-24):

```markdown
## Key Findings

| Configuration | PSNR (dB) | LPIPS ↓ |
|---------------|-----------|---------|
| Baseline (no defense) | 29.38 | 0.117 |
| Differential Privacy (ε=1) | 8.12 | 0.714 |
| Homomorphic Encryption | 12.45 | 0.623 |
| **DP + HE (combined)** | **6.37** | **0.824** |

- Baseline attacks successfully reconstruct recognizable faces
- Privacy defenses significantly degrade reconstruction quality
- Combined DP+HE provides strongest protection
```

Replace with:

```markdown
## Key Findings

All numbers below are read directly from `results/report/summary.csv`.

| Configuration | PSNR (dB) | LPIPS ↓ |
|---|---|---|
| Baseline (no defense) | 29.38 | 0.117 |
| DP, local, ε=8 | 6.71 | 0.807 |
| DP, local, ε=1 | 6.32 | 0.747 |
| DP, local, ε=0.1 | 6.36 | 0.806 |
| "HE" (quantize + fixed noise; see caveat below) | 14.03 | 0.635 |
| DP (ε=1) + "HE" | 6.37 | 0.824 |

- Baseline attacks reconstruct recognizable faces from a single client's
  raw gradient.
- **The DP mechanism above is correctly implemented local DP** — it clips
  and noises one client's own gradient before release, which is the right
  mechanism for the threat model tested here (an adversary reading a
  single pre-aggregation update). It is *not* a bug that ε=8/1/0.1 all land
  at roughly the same PSNR: per-coordinate Gaussian noise has L2 norm
  scaling ~σ√d, and at this model's d≈8.76M parameters, even ε=8 already
  injects noise ~1800× the clipped signal bound. The chosen ε range simply
  couldn't have shown graduated protection at this dimensionality — see
  `results/report/figures/dp_noise_scaling.png` and
  `scripts/dp_noise_scaling_proof.py` for the worked-out proof.
- **The "HE" row does not test encryption.** The implementation quantizes
  gradients and adds a fixed-scale Laplace noise term; for this model size
  it never executes real Paillier encryption, and even when it does, the
  code decrypts the result before scoring it — a real HE/secure-aggregation
  deployment would never expose a decrypted intermediate to the attacker
  this project simulates. The codebase has an implemented-but-unevaluated
  path for the honest version of this experiment (`fl_system.py`'s
  `capture_mode='agg_update'`, paired with `differential_privacy.
  aggregate_clipped_noisy`'s central-DP mechanism) — attacking only the
  FedAvg-averaged update, which is what a curious aggregator actually sees
  under real secure aggregation. No GPU/CelebA compute was available to
  run that experiment this cycle; it's a named limitation, not a filled-in
  result.
- This benchmark measures attack quality only. It does not measure model
  accuracy under each defense, so it cannot speak to the privacy/utility
  tradeoff.
```

- [ ] **Step 2: Update the Notes section**

Find:

```markdown
## Notes

- Results are specific to this experimental setup
- DP/HE implementations are research-grade, not production-ready
- See the interactive dashboard for detailed visualizations
```

Replace with:

```markdown
## Notes

- Results are specific to this experimental setup (single-client gradient
  leak, 8.76M-parameter model, CelebA 64×64).
- DP/HE implementations are research-grade, not production-ready.
- The DP and "HE" findings above share one root cause: neither evaluates
  the secure-aggregation / central-DP release point (the mechanisms for it
  exist in the code but were never run — see Key Findings).
- See the interactive dashboard for detailed visualizations.
```

- [ ] **Step 3: Verify the numbers against the CSV one more time**

Run:
```bash
python3 -c "
import csv
with open('results/report/summary.csv') as f:
    rows = {r['setting']: r for r in csv.DictReader(f) if r['group'] == 'defenses'}
for setting in ('baseline', 'dp_eps8', 'dp_eps1', 'dp_eps01', 'he', 'dp_he'):
    r = rows[setting]
    print(f\"{setting:10s} PSNR={float(r['PSNR']):.2f}  LPIPS={float(r['LPIPS']):.3f}\")
"
```
Expected output (compare each line against the table written in Step 1):
```
baseline   PSNR=29.38  LPIPS=0.117
dp_eps8    PSNR=6.71  LPIPS=0.807
dp_eps1    PSNR=6.32  LPIPS=0.747
dp_eps01   PSNR=6.36  LPIPS=0.806
he         PSNR=14.03  LPIPS=0.635
dp_he      PSNR=6.37  LPIPS=0.824
```
Also confirm `he_test` is absent: `python3 -c "print('he_test' in open('README.md').read())"` → expected `False`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: sync README numbers to summary.csv, correct DP/HE framing

Replace stale hand-copied numbers (DP eps=1 was 8.12, now 6.32; HE was
12.45, now 14.03) with values read directly from summary.csv, and add
the missing eps=8/eps=0.1 rows. Rewrite the DP/HE narrative around the
verified finding: the DP mechanism is correct local DP for the tested
threat model, not a bug; the epsilon sweep saturates because noise
scales with sqrt(d) at this model's 8.76M parameters. The HE row is
relabeled as quantize+noise, not encryption, with the real secure-
aggregation path named as implemented but unevaluated (no compute
available this cycle)."
```

---

## Self-Review Notes

- **Spec coverage:** Finding 1 (DP framing) → Task 1 + Task 4 + Task 7. Finding 2 (HE framing) → Task 1 + Task 7. Finding 3 (stale numbers) → Task 7. Finding 4 (casing) → Task 1. Finding 5 (dead code) → Task 2. Finding 6 (script duplication) → Tasks 5 and 6. Finding 8 (he_test leak) → Task 3. The presentation redesign (Decision §3) and website post (Decision §4) are intentionally **not** in this plan — the spec's own "Open items for planning phase" flags the dashboard IA/wireframe as unresolved, and both are ordered *after* this plan's corrections land. They need their own design pass before a task-level plan can be written without placeholders.
- **Placeholder scan:** no TBD/TODO; every step has literal code, exact commands, and expected output.
- **Type consistency:** `run_experiment_subprocess` and `parse_metrics_file` are used with identical signatures in both Task 6 call sites. `apply_theme`/`THEMES` keys match between definition (Task 5 Step 1) and the smoke test (Task 5 Step 6).
