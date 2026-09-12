# FL-Attack Rework — Design

## Context

FL-Attack is a student-era (course) project studying gradient inversion attacks
and privacy defenses (DP, HE) in federated learning on CelebA. The course and
its judged demo are over. The author is now a research engineer and wants the
project brought up to their current standard, across four axes: research
direction/rigor, code quality, presentation, and how it reads to hiring
managers, admissions committees, and technical judges.

Constraints discovered during investigation:
- No GPU/CelebA/torch available in this environment, and no CSC compute
  access either (Puhti/Mahti retired for this account; only file retrieval,
  no compute nodes). Confirmed via SSH: no FL-Attack artifacts exist on CSC
  scratch/home beyond what's already committed to this repo. **The numbers in
  `results/report/summary.csv` are the only numbers that exist.** No full
  experimental rerun is possible in this project cycle.
- Two published surfaces (README, personal site) disagree with each other on
  headline numbers; `results/report/summary.csv` is the verified source of
  truth (matches the live site, not the README).
- The dashboard/poster were built to be dumped on a single large screen for a
  now-defunct course judging panel. That audience and constraint no longer
  exist. The author independently considers the current visual design ugly
  (poor proportions, colors, unprofessional) and wants it replaced with its
  own strong identity — not inherited from imadlab.com, since that site's own
  visual identity is separately slated for a future rework and copying it now
  would be wasted work.
- Audience: hiring managers and academic/admissions reviewers, roughly
  equally. Positioning goal: broad, adaptable, rigorous, trustworthy —
  not narrowly "security researcher" or narrowly "industrial AI."

## Verified findings (facts, not hypotheses)

1. **DP epsilon sweep carries no signal.** `run_experiment.py:239-246` clips
   and adds Gaussian noise to the *entire* 8.76M-parameter gradient vector as
   a single L2 ball, per client, then hands it straight to the attacker. Noise
   L2 norm scales ~σ√d; for d≈8.76M, even the weakest tested ε=8 (σ≈0.6)
   produces noise norm ~1780 against a signal capped at max_norm=1.0 — signal
   is already destroyed before ε does anything. Confirmed against
   `summary.csv`: PSNR is flat (6.71 / 6.32 / 6.36 dB) across ε=8/1/0.1.
   `Differential_privacy.py`'s `aggregate_clipped_noisy` — which correctly
   scales noise by `num_clients` for DP-FedAvg — exists but is **never
   called** anywhere in the codebase.
2. **The "Homomorphic Encryption" result doesn't test encryption.**
   `HE_SAMPLE_LIMIT = 10_000` gates real Paillier vs. a fallback path; the
   model has 8.76M parameters, so every recorded HE run took the `else`
   branch (`round(g*precision)/precision` + fixed-scale `Laplace(0, 0.01)`
   noise) — no encryption ever executed. Even when the real-Paillier branch
   *would* run, the code encrypts → adds noise → **decrypts** → hands the
   attacker plaintext; Paillier decryption is exact, so the round-trip itself
   contributes zero protection, only the noise term does. The codebase
   already contains the honest version of this experiment:
   `fl_system.py`'s `capture_mode='agg_update'` gives the attacker only the
   FedAvg average, which is what a curious server actually sees under secure
   aggregation / HE-protected FL.
3. **Numbers disagree across surfaces.** README's defense table (baseline
   29.38 / DP ε=1 8.12 / HE 12.45 / DP+HE 6.37 dB PSNR) does not match
   `results/report/summary.csv` (baseline 29.38 / DP ε=1 6.32 / HE 14.03 /
   DP+HE 6.37 dB PSNR, plus DP ε=8 6.71 and DP ε=0.1 6.36 which README omits
   entirely). The live site's `__PRERENDERED_DATA__` agrees with the csv, not
   the README. `summary.csv` is the source of truth.
4. **Filename/import casing mismatch.** File is `Differential_privacy.py`;
   the only import site (`run_experiment.py`) does `from differential_privacy
   import ...`. Works on case-insensitive filesystems (macOS/Windows), breaks
   on Linux (case-sensitive) — a portability bug in something billed as a
   reproducible benchmark.
5. **Dead code.** `GradientInversionAttack.reconstruct_image` (the plain
   pre-iDLG method) has zero callers anywhere. Several `layer_weights` modes
   in `gradient_attack.py` (`early`, `early_linear`, `early_strong`,
   `early_conv`, `spatial`) are implemented but never appear in any
   `results/ablation/*` output — no published result used them.
6. **Script duplication.** `scripts/make_poster.py` and
   `scripts/make_poster_light.py` are ~85% identical (diff is 153 of ~1050
   lines) — differ only in a hardcoded color palette. `scripts/exp_base.py`
   and `scripts/exp_phase1.py` duplicate the same `ExperimentConfig`
   dataclass shape and subprocess-runner scaffolding around genuinely
   different experiment logic.
7. **`scripts/make_dashboard.py`** is a single 2,226-line file built almost
   entirely around one `DashboardBuilder` class (~2,030 lines). It's
   build-time tooling nobody but the author touches; correctness matters more
   than internal elegance here.

## Decisions

### Scope and ordering
Fix truth, then code, then presentation, then the website post. No new
website copy is written until the corrected DP/HE story is settled, so the
post never states something the code doesn't actually do.

### 1. Scientific / correctness fixes
- Wire `aggregate_clipped_noisy` (or an equivalent correctly-scaled path)
  into the real experiment flow, or explicitly document in-code and in the
  README why per-client full-vector clipping was chosen and what its cost is
  in high dimensions. Either way, stop implying the ε sweep shows graduated
  privacy protection when the data shows a flat noise floor.
- Add a small, dependency-light synthetic script (no CelebA, no GPU) that
  demonstrates the σ√d noise-domination effect analytically/numerically as a
  function of parameter count — produces a supporting figure. This turns "the
  epsilon sweep is broken" into a documented, evidenced insight rather than
  a caveat sentence.
- Reframe the HE result: stop labeling quantize+Laplace as "Homomorphic
  Encryption" protection. Present the `agg_update` capture path (attacker
  only sees the FedAvg average) as the honest secure-aggregation / HE-adjacent
  result, since the codebase already implements it.
- Sync README's defense table to `results/report/summary.csv` (verified
  source of truth); update the personal site's project entry once the new
  story is final (last step, see below).
- `git mv Differential_privacy.py differential_privacy.py`.
- Remove `reconstruct_image` and the never-used `layer_weights` modes listed
  above. Keep only options with an evidenced result behind them.

### 2. Code quality
- Core algorithmic files (`fl_system.py`, `gradient_attack.py` minus the dead
  code above, `differential_privacy.py`, `homomorphic_encryptor.py`,
  `device_utils.py`) are already reasonably clean — type-hinted, documented,
  no full rewrite needed.
- Merge `make_poster.py` + `make_poster_light.py` into one script with a
  `--theme dark|light` flag.
- Extract the shared `ExperimentConfig` + subprocess-runner scaffolding
  between `exp_base.py` and `exp_phase1.py` into one small shared module;
  keep each script's actual experiment logic where it is.
- `make_dashboard.py`: light touch only. Split out obviously-separable
  concerns if doing so doesn't cost more than it's worth; do not attempt a
  full architectural rewrite of a script that only the author runs.

### 3. Presentation — full redesign
The existing dashboard and poster are retired as artifacts of a one-time,
single-screen, course-judging use case that no longer applies. Replace with a
new design that:
- Has its own visual identity — not inherited from imadlab.com's current
  (soon-to-be-reworked) design tokens. Judged on its own merits: proportion,
  color, typography, and information hierarchy done well.
- Restructures the narrative rather than reproducing a dense single-screen
  dump: problem → method → honest results (including the corrected DP/HE
  story as a feature, not something hidden) → try-it-yourself. Should read
  well both as a 60-second skim (hiring manager) and a slower, rigor-checking
  pass (academic reviewer/admissions).
- Keeps the artifact set practical: a web dashboard (GitHub Pages, as today)
  is the primary deliverable; whether a static "poster" image remains a
  separate artifact or gets folded into the dashboard is an implementation
  decision, not a design constraint — resolve it during planning based on
  what the new structure actually needs.

### 4. Website post (imadlab.com project entry)
Last step. Once 1–3 land, revise the existing `full_description` (already
well-written) to reflect the corrected DP/HE story. The self-audit itself
("re-audited past work, found a real methodological bug, fixed the framing,
shipped a synthetic proof") is presented as a feature of the narrative, not
buried — it's a stronger signal of engineering/research judgment than clean
numbers would have been.

## Out of scope
- Any new large-scale experiment requiring GPU/CelebA/CSC compute — none is
  available this cycle.
- Rebuilding imadlab.com's own visual identity — explicitly deferred by the
  author to a separate future task.
- Full architectural rewrite of `make_dashboard.py`.

## Open items for planning phase
- Exact IA/wireframe of the redesigned dashboard.
- Whether the poster image survives as a standalone artifact or is retired.
- Exact wording of the revised DP/HE sections in README and the site entry.
