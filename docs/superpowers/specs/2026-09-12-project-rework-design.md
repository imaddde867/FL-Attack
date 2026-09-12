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

1. **The DP epsilon sweep can't show graduated protection, for three
   separable reasons — only one of them is a bug.** (a) The analytic Gaussian
   mechanism itself, as implemented, is correct. (b)
   `run_experiment.py:239-246` applies it to a *single client's* full
   gradient vector and never divides noise by `num_clients` — this is
   per-client DP, not DP-FedAvg, roughly 10x more noise than the aggregate
   warrants. `Differential_privacy.py`'s `aggregate_clipped_noisy`, which
   scales noise correctly for FedAvg, exists but is **never called** anywhere
   — this is the genuine bug. (c) Independent of (b): per-coordinate Gaussian
   noise has L2 norm scaling ~σ√d; for the model's d≈8.76M parameters, even
   the weakest tested ε=8 (σ≈0.6) produces noise norm ~1780 against a signal
   capped at max_norm=1.0 — the sweep's chosen ε range sits entirely inside
   the noise-dominated regime for this dimensionality, so it was incapable of
   showing graduation regardless of (b). Confirmed against `summary.csv`:
   PSNR is flat (6.71 / 6.32 / 6.36 dB) across ε=8/1/0.1. Framing: the
   mechanism was implemented correctly; the FedAvg scaling was missed; the
   experimental design couldn't have answered the question it posed even
   after fixing that.
2. **The "Homomorphic Encryption" result doesn't test encryption.**
   `HE_SAMPLE_LIMIT = 10_000` gates real Paillier vs. a fallback path; the
   model has 8.76M parameters, so every recorded HE run took the `else`
   branch (`round(g*precision)/precision` + fixed-scale `Laplace(0, 0.01)`
   noise) — no encryption ever executed. Even when the real-Paillier branch
   *would* run, the code encrypts → adds noise → **decrypts** → hands the
   attacker plaintext; Paillier decryption is exact, so the round-trip itself
   contributes zero protection, only the noise term does. `fl_system.py`
   already implements the honest version of this threat model —
   `capture_mode='agg_update'` gives the attacker only the FedAvg average,
   which is what a curious server actually sees under secure aggregation /
   HE-protected FL — **but no run in `results/` ever used it** (confirmed:
   zero `config.json` files anywhere have `attack_source: "agg_update"`).
   This is not an evaluated result waiting to be featured; it's an
   implemented-but-unevaluated threat model.
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
8. **A debug row is (likely) leaking into the published dashboard.**
   `summary.csv` has a `defenses/he_test` row (PSNR 19.04, empty LPIPS) that
   appears in neither the README nor the site. `make_dashboard.py` filters
   chart/ranking data by `group == "defenses"` with no name exclusion, so
   this row is included in `chart_defenses_grouped_bars` /
   `chart_defenses_scatter` unless explicitly filtered — rendering a
   debug/incomplete run as a defense result, with a gap where LPIPS should
   be.

## Decisions

### Scope and ordering
Fix truth, then code, then presentation, then the website post. No new
website copy is written until the corrected DP/HE story is settled, so the
post never states something the code doesn't actually do.

### 1. Scientific / correctness fixes
- **DP fix (single decision, not either/or):** wire the correctly-scaled
  `aggregate_clipped_noisy` path into `run_experiment.py` so the code is
  right going forward — but do **not** rerun and replace the published
  `summary.csv` numbers (no compute available; see Reproducibility policy
  below). The existing DP rows stay, explicitly labeled as produced by the
  pre-fix, per-client-noise code path. README/site text states the DP
  finding as: mechanism correct, FedAvg scaling was missing (now fixed in
  code), and the tested ε range was inside the noise-dominated regime for
  this model's dimensionality regardless — never claim the fixed code was
  used to produce the published numbers.
- Add a small, dependency-light synthetic script (no CelebA, no GPU,
  matplotlib+numpy only — confirmed available) that demonstrates the σ√d
  noise-domination effect analytically as a function of parameter count —
  produces a supporting figure. This is the one genuinely new artifact this
  cycle, and it documents finding (b)+(c) above with evidence rather than a
  caveat sentence.
- **HE reframe is text-only, not a headline result.** No run in `results/`
  ever exercised `capture_mode='agg_update'`, so there is no honest HE/secure-
  aggregation number to feature. Instead: describe the correct threat model
  in prose (README + site), state plainly that the published HE row measured
  fixed-scale Laplace noise and never executed encryption, and note that the
  secure-aggregation path is implemented in `fl_system.py` but unevaluated —
  an explicit, named limitation rather than a filled-in result.
- Sync README's defense table to `results/report/summary.csv` (verified
  source of truth, including the ε=8 and ε=0.1 rows README currently omits);
  update the personal site's project entry once the new story is final (last
  step, see below). Drop or clearly label the `he_test` row as debug output
  before it reaches any published table or chart.
- `git mv Differential_privacy.py differential_privacy.py`.
- Remove `reconstruct_image` and the never-used `layer_weights` modes listed
  above (verified: no `config.json` in `results/` has a non-null
  `layer_weights`). Keep only options with an evidenced result behind them.

### Reproducibility policy
Once the DP path is rewired and dead code/duplication is removed, the
current code can no longer regenerate the numbers already committed to
`results/` — those were produced by the pre-fix code. For a project
presented as a reproducible benchmark, silently shipping code that can't
reproduce its own published results is a bigger credibility risk than the
original bug. Before making any of the correctness/cleanup changes: tag the
current commit (e.g. `results-as-published`), and add one explicit line to
the README: numbers in `results/` were produced at that tag; the current
code contains the corrections described above and will not reproduce the
same DP numbers exactly if rerun.

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
  available this cycle. This includes rerunning the DP sweep with the fixed
  scaling and evaluating the `agg_update` HE threat model — both are named
  limitations, not deliverables.
- A model-accuracy / utility axis alongside attack quality (would require FL
  training runs under each defense — no compute available). The benchmark
  measures attack quality only and cannot speak to the privacy/utility
  tradeoff; state this plainly rather than implying otherwise.
- Rebuilding imadlab.com's own visual identity — explicitly deferred by the
  author to a separate future task.
- Full architectural rewrite of `make_dashboard.py`.

## Open items for planning phase
- Exact IA/wireframe of the redesigned dashboard.
- Whether the poster image survives as a standalone artifact or is retired.
- Exact wording of the revised DP/HE sections in README and the site entry.
- Exact commit ordering around the `results-as-published` tag (tag must be
  cut before the DP rewire / dead-code / rename commits land).
