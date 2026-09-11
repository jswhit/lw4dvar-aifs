# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is a **work-in-progress port** of an experimental long-window 4D-Var
data-assimilation solver from NeuralGCM to ECMWF's **AIFS Single v2**
forecast model. The mature, working version of the solver (built around
NeuralGCM) lives in the sibling directory
`/scratch4/BMC/gsienkf/Jeffrey.Whitaker/long-window-4dvar-jswhit2/` — that
repo's `long_window_4dvar.py` / `long_window_4dvar_utils.py` /
`config.yml.template` are the reference implementation and design pattern
this repo is meant to reproduce, but driven by AIFS instead of NeuralGCM.

The ported driver (`long_window_4dvar.py` / `long_window_4dvar_utils.py`)
now exists and is working — see "The 4D-Var solver" section below for its
structure and history. (This paragraph originally described the
not-yet-ported state of the repo in August 2026; left uncorrected for a
while as the port and its "The 4D-Var solver" section below grew up
alongside it — fixed 2026-09-11.)

## Repository layout

- `run_long_window_4d.sh` — SLURM batch script and the only original file
  in this repo currently. It:
  - loads `cuda` and `rdhpcs-conda` modules, activates the conda env at
    `/scratch4/BMC/gsienkf/whitaker/conda/envs/python3.13`
  - symlinks `models` and `input` from the sibling
    `long-window-4dvar-jswhit2` project (shared model checkpoints / IC data)
  - symlinks `psobs` from `/scratch3/NCEPDEV/da/Jeffrey.Whitaker/psobs`
    (surface pressure observation files, organized by date-stamped text
    files, used as the DA obs source)
  - symlinks `config.yml` -> `config.yml.template` (a template does not
    exist here yet either — see jswhit2's `config.yml.template` for the
    expected schema: an `exp:` block of experiment/cycling parameters and
    a `windows:` block of per-window DA settings)
  - runs `python -u long_window_4dvar.py`
- `aifs-single-2.0/` — vendored clone of ECMWF's official
  `ecmwf/aifs-single-2.0` Hugging Face repo (its own git history; treat as
  upstream, don't restructure it). Contains the model checkpoint
  (`aifs-single-mse-2.0.ckpt`), `inference.yaml` (anemoi-inference config:
  pre/post-processing filters, GRIB encoding, ECMWF Open Data input), and
  the reference `run_AIFS_v2.0.ipynb` notebook showing the canonical
  anemoi-inference invocation (`anemoi-inference run inference.yaml`).
- `AIFS-single-2.0-on-all-GPUs/` — vendored clone of a community wrapper
  around anemoi-inference (also its own git history). Its main value for
  this project is `aifs/compat.py`, which monkey-patches `flash_attn` in
  `sys.modules` with a `torch.nn.functional.scaled_dot_product_attention`
  shim *before* Anemoi imports it — this is what lets AIFS run on non-Ampere
  GPUs (or CPU/MPS). Also provides `aifs/device.py` (device detection),
  `aifs/initial_conditions.py` (ECMWF Open Data IC download + local cache),
  `aifs/forecast.py` (thin wrapper around anemoi-inference exposing
  `run_forecast` / `run_forecast_streaming`), and `aifs/plot.py`. Note: the
  SDPA shim is **not** bit-identical to real flash-attn, and it does not
  support AIFS-ENS (only AIFS Single) — see that repo's README FAQ.

## Running

This is HPC batch code, submitted via Slurm — there is no build step,
package manifest, linter, or test suite in this repo.

```bash
sbatch run_long_window_4d.sh
```

The job requests a single H100 GPU (`gpu-ai4wp` account, `u1-h100`
partition, 8h walltime, 96G mem). It expects to be run from this directory
so the relative symlinks (`models`, `input`, `psobs`, `config.yml`) resolve
correctly. `run_long_window_4d.sh` activates the dedicated `aifs2` env
described below and runs `python -u long_window_4dvar.py` (optionally
`python -u long_window_4dvar.py <config path>` to point at a config other
than the `config.yml` symlink, e.g. for a side-by-side test run).

## The `aifs2` conda environment

The AIFS/anemoi/torch stack is installed in its own dedicated conda env,
separate from the `python3.13` (jax/NeuralGCM) env used by the sibling
jswhit2 project, since the two stacks (jax vs. torch) shouldn't be mixed
and flash-attn requires a specific Python/torch/CUDA ABI:

```
/scratch4/BMC/gsienkf/whitaker/conda/envs/aifs2   (Python 3.12)
```

Activate with `conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/aifs2`.

It was built by pip-installing the exact pins from `aifs-single-2.0`'s
`pyproject.toml`/`uv.lock` (the checkpoint's own validated environment),
**not** the looser community pins in `AIFS-single-2.0-on-all-GPUs/requirements.txt`:
`torch==2.7.1+cu128`, `flash-attn==2.8.3` (real flash-attn, prebuilt wheel
for cu12/torch2.7/cp312 from `cathalobrien/get-flash-attn` — H100 is
Ampere-class+ so it doesn't need the SDPA compatibility shim),
`torch-geometric==2.6.1`, `anemoi-models==0.9.3`, `anemoi-graphs==0.6.4`,
`anemoi-transform==0.1.16.post2`, `anemoi-utils==0.4.35.post3`,
`anemoi-datasets==0.5.26`, plus the `inference` extras
(`anemoi-inference[huggingface]==0.8.3`,
`anemoi-plugins-ecmwf-inference[opendata]==0.2.1`,
`earthkit-regrid==0.5.1`, `ecmwf-opendata==0.3.29`). One conflict needed a
manual pin: pip's resolver initially picked `mir-python==1.29.1.26`, which
requires `numpy>=2`, conflicting with `anemoi-graphs`' `numpy<2` — pinned
down to `mir-python==1.28.1.19` (matching `uv.lock`) to resolve it. Full
`pip freeze` output is saved at
`/scratch4/BMC/gsienkf/whitaker/conda/envs/aifs2-requirements.lock.txt`.

Verified against the local checkpoint with
`anemoi-inference validate aifs-single-mse-2.0.ckpt` from inside
`aifs-single-2.0/` — passes with only harmless patch-version differences
(no missing/incompatible modules).

**Built for the no-internet H100 compute nodes**: all packages (including
the flash-attn wheel, which is pulled from a GitHub release, not PyPI)
were downloaded on a login/build node that has internet access. Nothing
in this env needs to reach the network at runtime — the model checkpoint
is already local (`aifs-single-2.0/aifs-single-mse-2.0.ckpt`). If new
packages are ever needed, install them from a node with internet access
(e.g. `ufe04`), not from an H100 job.

## Running AIFS from historical (non-opendata) initial conditions

`aifs-single-2.0/inference.yaml` ships with `input: opendata`, but ECMWF
Open Data only retains the last ~12 forecast cycles (~2-3 days) — it
cannot initialize a run from an arbitrary past date like the ones this
4D-Var project needs (e.g. 2014-2015, matching jswhit2's ERA5-driven
experiments). `test_forecast/` is a working, checked-in example of doing
this instead from ERA5 via CDS, split into an online fetch step and an
offline run step (the H100 nodes have no internet):

1. `test_forecast/fetch_ic_cds.py` — run on a login node (has internet,
   `~/.cdsapirc` is already configured). Builds a real `anemoi.inference`
   runner from `inference_era5_20141230.yaml` (`input: {cds: {dataset:
   reanalysis-era5-complete, class: ea}}`), then calls
   `create_prognostics_input()` / `create_constant_coupled_forcings_input()`
   / `create_dynamic_forcings_input()` and retrieves each via CDS,
   concatenating to one local GRIB file. `class: ea` is required —
   without it the checkpoint's own metadata requests `class: od`
   (ECMWF's live operational archive), which CDS's ERA5 dataset doesn't
   recognize and silently returns nothing for.
2. Two ERA5 data gaps discovered the hard way, both worth knowing before
   re-running this for a different date:
   - `swh` (significant wave height): the checkpoint's baked-in paramId
     (3100) is from a post-2026 operational wave table ERA5 doesn't have.
     Fixed with a `typed_variables` override in the yaml (same trick the
     original `inference.yaml` already uses for `mwd`) that resolves it
     by shortname within the wave stream instead: `mars: {param: swh,
     stream: wave, levtype: sfc}`.
   - `h1012, h1214, h1417, h1721, h2125, h2530` (wave-period-band
     significant heights): genuinely absent from ERA5 at any paramId or
     shortname — these were introduced for AIFS v2's IFS-50r1-esuite
     fine-tuning, after ERA5 was fixed. `test_forecast/patch_missing_wave_fields.py`
     cold-starts them at zero (a standard, defensible wave-model
     cold-start; do not treat IC files built this way as
     scientifically faithful beyond pipeline testing).
   - Separately, `input: grib`/`GribFileInput` expects every purpose
     (including the "constant" orography/bathymetry forcings: lsm, sdor,
     slor, wmb, surface z) to be present at **both** lagged input dates,
     even though they're time-invariant — `create_constant_coupled_forcings_input`
     only fetches them once. `test_forecast/duplicate_constant_fields.py`
     clones those 5 fields' GRIB messages onto the other date (lossless,
     since the values are truly constant).
3. `test_forecast/inference_local_20141230.yaml` — the offline run
   config: same as the ERA5 fetch config but `input: {grib:
   ic_20141230T00_final.grib}` (the patched local file from steps 1-2)
   instead of `input: cds`. Safe to run on the H100 nodes.
   `test_forecast/run_test_forecast.sh` is the matching sbatch launcher
   (`conda activate .../aifs2`, then `anemoi-inference run
   inference_local_20141230.yaml`).
4. The output GRIB encoding (`class: ai, model: aifs-single`, from the
   original `inference.yaml`) throws a wall of `ECCODES ERROR ... model
   (type=string) failed: Key/value not found` per field/step if left at
   the default GRIB edition 1 (the `ai`-class local table isn't defined
   there) — anemoi-inference auto-retries as edition 2 and succeeds
   regardless, but the `.err` log is enormous. Both yamls here set
   `encoding: {edition: 2, ...}` explicitly to skip the failed edition-1
   attempt and produce a clean log.

Sanity-check a completed run's output before trusting it, e.g. global
mean 2t and msl per step should be physically reasonable and vary
smoothly across steps (confirmed working for the 2014-12-30T00 run:
2t ~285K, msl ~1011.5 hPa, stable across the 6/12/18/24h steps).

## The 4D-Var solver: AIFS-single-2.0 / PyTorch (ported from NeuralGCM/JAX)

`long_window_4dvar.py` / `long_window_4dvar_utils.py` are the ported,
working AIFS/PyTorch versions of the solver (edited in place from the
NeuralGCM/JAX originals in the jswhit2 sibling project — same filenames, same
overall control flow: `for k in range(n_init): for window in windows: ...`,
`utils.get_window`/`get_input`/`get_verif`/`get_psobs`/`compute_optimal`/
`save_*`). New modules added alongside them:

- **`aifs_model.py`** — `AIFSState` (a thin wrapper whose `.state` *is* the
  packed `(1, multi_step=2, n_points, n_vars)` tensor — AIFS's two lagged
  time levels *are* its state, there's no NeuralGCM-style encode step) and
  `AIFSModel` (loads the checkpoint via a real `anemoi.inference.Runner`,
  reusing its variable/grid bookkeeping rather than reimplementing it).
  `AIFSModel.advance()` is the differentiable rollout: it bypasses
  `AnemoiModelEncProcDec.predict_step`'s hardcoded `torch.no_grad()` by
  calling `pre_processors -> model.forward -> post_processors` directly
  (verified against real gradient descent — see `test_forecast/smoke_test_optimize.py`),
  wrapped per-step in `torch.utils.checkpoint.checkpoint`, and reuses
  `Runner.copy_prognostic_fields_to_input_tensor`/
  `add_dynamic_forcings_to_input_tensor` for the sliding-window update.
- **`aifs_grid.py`** — `GridInterpolator`: a k-d tree over AIFS's irregular
  N320 octahedral grid, replacing the NeuralGCM version's bilinear
  regular-grid `interp2d` with k-NN inverse-distance weighting (precomputed
  once per window, not per epoch).
- **`aifs_ic.py`** — historical ERA5 IC/verification fetching (factored out
  of the validated `test_forecast/` scripts), used by `get_input`/`get_verif`.
- **`aifs_inference.yaml`** — static anemoi-inference config (pre/post
  processors, `typed_variables` overrides, `patch_metadata`) the runner
  needs for its checkpoint/variable bookkeeping. Its `input:`/`output:`/
  `date:`/`lead_time:` are unused placeholders — the driver never calls
  `runner.execute()`/`.run()`, only lower-level `Runner` methods.
- **`aifs_prefetch_ic.py`** — run from a login node *before* `sbatch
  run_long_window_4d.sh` to warm the ERA5 cache (`ic_cache/`, by default)
  for a planned run's whole date range; the H100 nodes have no internet, so
  `get_input`/`get_verif` must find everything already cached.

**As of 2026-09-11 the control variable is latent-only** (an increment in
AIFS's hidden-mesh encoder-output space, injected as a one-time forcing at
the first rollout step) — see "`control_space: 'latent'`" below for the
architectural reasoning, and "Merge to mainline" further down for how it
became the only path. `dt_verif`/`dt_init` must be multiples of AIFS's
fixed 6h timestep; observations may be sampled finer than that via the
per-window `dt_obs` key (see "Merge to mainline"). The debugging notes
just below predate the latent-only merge and describe an earlier
physical-space control path (`state_scales`/`control_mask`/
`NONNEGATIVE_VARS`, an increment spanning both lagged time levels of the
packed state directly) that has since been **removed from the code** —
kept here for the hard-won lessons (the AdamW-preconditioning insight in
particular generalizes beyond the code path that prompted it), not as a
description of anything still runnable.

### Debugging notes worth knowing before touching this again

- **`model.interface`'s parameters must be frozen** (`requires_grad_(False)`,
  done once in `AIFSModel.__init__`) — `torch.load` doesn't do this on its
  own. Without it, any state built via `model.advance()` *before* the epoch
  loop (e.g. `get_input()`'s initial forecast) silently carries a live,
  un-freed autograd graph back through that forecast's model-weight-dependent
  computation. Reusing that state as a fixed background across epochs then
  fails on the *second* epoch's `loss.backward()` with "Trying to backward
  through the graph a second time" — the first backward() already freed the
  shared upstream graph. This cost significant debugging time (bisected via
  minimal repro scripts down to: passes with 1-5 chained `model.advance()`
  calls in isolation, fails as soon as a background built *before* the loop
  is reused *inside* it) before the actual cause (unfrozen weights, not
  checkpointing) was found. Nested `torch.utils.checkpoint` (mine wrapping
  anemoi's own internal encoder/processor/decoder checkpointing) was a red
  herring — disabling it did **not** fix the bug.
- **`AIFSModel.prepare_initial_state`** must NOT call
  `runner.create_constant_forcings_inputs()` wholesale — for this checkpoint
  it has a "loaded" sub-branch (z/lsm/sdor/slor) that tries to build an
  input source from `aifs_inference.yaml`'s placeholder `input:` and
  crashes, and a safe "computed" sub-branch (cos_latitude etc., pure
  function of lat/lon) that's genuinely needed. The two are split apart
  manually there — see that method's docstring for exactly which checkpoint
  category maps to which.
- **`GribFileInput.create_input_state()` always requests both of the
  checkpoint's lagged dates**, regardless of what's actually in the GRIB
  file — it can't be used to read a genuine single-date snapshot.
  `get_verif()` needs exactly that (one ERA5 state, not a lagged pair), so
  it uses `aifs_ic.read_single_date_fields()` instead, which reads the raw
  fields directly via `earthkit.data` and bypasses that validation.
- **Finite-difference gradient checks are close to useless here** at normal
  scales: flash-attention forces part of the forward pass into bf16, whose
  quantization noise (~1 part in 256) swamps any legitimate small
  finite-difference signal, especially for a global-sum loss (catastrophic
  cancellation) or a single grid-point-to-grid-point sensitivity. The
  decisive check that actually validated the differentiable rollout was
  "does AdamW, using these gradients, reduce a real loss" (smooth,
  monotonic decrease over 20 steps), not raw `(f(x+eps)-f(x-eps))/2eps`
  agreement.
- **`get_surface_pressure`'s per-observation linear interpolation is fully
  vectorized** (batched `torch.searchsorted` + `gather`), not a Python
  `for j in range(n_obs): out[j] = ...` loop — the original per-observation
  loop version was both ~4x slower (15000 obs × 5 calls/epoch) and the
  first suspect during the graph-reuse debugging above (it wasn't the
  actual cause, but is worth keeping vectorized regardless).
- **`get_surface_pressure` needs `pressure_levels`/`geopotential` in
  top-of-atmosphere-first order** — `torch.searchsorted` requires its search
  array (`relative_height = orography - geopotential`) to be ascending, but
  `AIFSModel`'s level ordering is surface-first (1000hPa at index 0), which
  makes geopotential *increase* and `relative_height` *decrease* along that
  axis. Passing it in that order silently gave `searchsorted` a descending
  array (not an error — undefined/garbage bracket indices), which produced
  a **systematic ~230-280 hPa high bias in H(x) at every station**, not
  random noise. Symptom: the background-check QC (`bg_check`) then rejected
  ~97-98% of observations (the tiny fraction of "used" obs were essentially
  coincidental near-matches against a badly-biased H(x), not genuine
  skillful background fits) — confirmed by first ruling out the k-NN obs
  interpolation itself (checked against real station coordinates: nearest
  neighbors were consistently <0.6° away, correctly weighted). Fixed with a
  `torch.flip` on both arrays before the search. After the fix, `n_used`
  jumped from ~50/9600 to ~9000/9600 per verif time, and O-A actually beats
  O-B for the first ~2.5 days of a 4-day-window test (see below) instead of
  diverging catastrophically at long lead times, confirming the earlier
  divergence was substantially the optimizer chasing noise from
  mismatched/near-coincidental "used" observations, not purely a
  learning-rate problem. The (currently dormant — `ps_operator: mslp` is
  not the default) alternate forward-operator path had the same surface-index
  mixup (`decoded['geopotential'][-1, :]` assumed index -1 = surface; fixed
  to index 0) — check that path too if it's ever turned on.
- **`state_scales` was originally a post-hoc `increment.grad.mul_(...)`
  after `.backward()` -- this does essentially nothing under AdamW.** AdamW
  normalizes each parameter's own step by its own accumulated gradient
  statistics (`m_hat/sqrt(v_hat)`), which is scale-invariant to a constant
  rescaling of that parameter's raw gradient in steady state. Confirmed
  empirically: with every control variable nominally at scale 1.0, u/v/t/z/sp
  all received nearly identical *absolute* per-epoch updates (~0.1-0.2)
  despite natural physical ranges spanning ~7 (v) to ~98526 (sp) -- z and sp
  (whose horizontal std is ~500-3000x u/v/t's) were only negligibly perturbed
  in relative terms. This was the actual root cause of a puzzle the user
  spotted: after a run with strong ps-obs innovation improvement (33.8% RMS
  reduction), the driver's `printz500err` before/after check showed
  *literally unchanged* z500 error. It looked like it could be a
  `printz500err` wiring bug (comparing `model.decode_state(input_encoded)`
  vs `model.decode_state(fix_inputs)`, `fix_inputs.state = params_best`) but
  wasn't -- directly diffing `*_control_inputs.nc` vs `*_optimal_inputs.nc`
  confirmed the two really were (barely) different. **Fix**: reparametrize
  `increment` as living in *scaled* units and divide by `scale_vec`
  *inside* the differentiable forward pass
  (`physical_state = input_state.state + increment / scale_vec` in
  `compute_loss_4dvar`), so autograd's chain rule produces a genuinely
  rescaled gradient -- matching how the original NeuralGCM code's
  `_rescale_state`/`_unscale_state` actually worked. `compute_optimal` now
  builds two separate per-column tensors: `control_mask` (hard 0/1 gate,
  still applied post-`.backward()` via `increment.grad.mul_(control_mask)`
  -- correct for an on/off decision, since a genuinely zero gradient stays
  zero regardless of AdamW normalization) and `scale_vec` (the differentiable
  preconditioner, default 1.0, overridden per-name by `state_scales`).
- **A second, compounding bug in the same `state_scales` loop**: its
  variable/family column lookup checked `model.var_to_idx` *before*
  `model._levels_by_base`, backwards from the (correct) precedent already
  used for `NONNEGATIVE_VARS` elsewhere in the file. `model.var_to_idx` is
  the *raw* checkpoint mapping (`checkpoint.variable_to_input_tensor_index`)
  and has its own single-column entry literally named `'z'` -- the constant
  surface orography, before `AIFSModel.__init__` renames it to
  `_single_by_base['geopotential_at_surface']` specifically to avoid
  colliding with the pressure-level `'z'` family. Because of the lookup
  order, `state_scales: {z: ...}` was silently resolving to that single
  orography column instead of the 14-level `z` family -- so even after the
  scale_vec fix above landed, a first test with `state_scales: {z: 0.02,
  sp: 0.005}` still showed `sp` clearly amplified (measured
  `max|diff|`~56-58 Pa between background and analysis, matching the
  expected ~200x) while `z`'s pressure levels stayed at the *default* scale
  1.0, indistinguishable from unscaled u/v/t (`max|diff|`~0.29 at every
  level, same as u/v/t's ~0.28-0.29) — confirming the family lookup was
  being shadowed. **Fix**: swap the lookup order to check
  `_levels_by_base` first, `var_to_idx` only as a fallback for genuine
  single-level names. After this fix, `z`'s `max|diff|` jumped to ~14.3
  (~49x u/v/t's, matching the intended 1/0.02=50x amplification) and its
  RMS perturbation at every level went from ~0.01m (bug) to ~0.5-0.6m (real).
- **Even with both of the above fixed, z500 verification error still barely
  moves** (NH/tropics/SH/global RMS: 8.54/3.71/7.86/6.70 before vs.
  8.56/3.73/7.87/6.71 after -- a hair *worse*, within noise) despite a
  confirmed-real ~0.5m RMS geopotential correction at every level and the
  best-yet ps-obs fit (53.4% RMS reduction, n_used ~9000-9800/window,
  `learn_rate: 2.e-3`, `state_scales: {z: 0.02, sp: 0.005}`, `lam: 0`, 100
  epochs, no divergence). This is evidence, not proof, that the ~0.5m
  correction just isn't well-correlated with the true ERA5 z500 error
  pattern -- a background z500 error of ~6.7-8.5m is large enough that even
  a perfectly-aligned ~0.5m correction would only be expected to reduce RMS
  by roughly 5-10%, so "no visible improvement" is within what a
  ps-obs-only (no other observation types), single-window, `lam=0`
  4D-Var cost function could plausibly produce this early in tuning -- not
  necessarily a further bug. Worth deliberately investigating (larger
  `state_scales['z']`, more epochs, or checking whether the geopotential
  correction's spatial pattern actually correlates with O-B at all) before
  assuming this is expected/acceptable behavior.

### `control_space: 'latent'` -- optimizing the control variable in AIFS's hidden-mesh space

Added September 2026 to test whether optimizing in AIFS's normalized,
lower-dimensional encoder-output space converges better than the (now
removed) physical-space increment, which needed per-variable `state_scales`
preconditioning -- see the debugging notes above. Originally gated by an
`exp['control_space']` config key (`'physical'` default, `'latent'` opt-in)
selecting between two parallel code paths -- physical `compute_optimal`/
`compute_loss_4dvar` vs. latent `compute_optimal_latent`/
`compute_loss_4dvar_latent` -- kept deliberately side by side so physical
mode stayed a known-good fallback while latent mode was unproven. Once
latent mode was confirmed to out-converge physical mode with no per-variable
tuning (see "real-DA-window results" below) and validated through a real
20-cycle run (see "Merge to mainline" further down), the physical path and
the `control_space` key itself were deleted: `compute_optimal`/
`compute_loss_4dvar` in the current `long_window_4dvar_utils.py` ARE what
this section calls `compute_optimal_latent`/`compute_loss_4dvar_latent` --
those old physical/`_latent`-suffixed names below are historical, not
current API. A stray `control_space: 'latent'` key left in an old config
file is harmless (silently unused) now that latent is the only path.

- **Numbers** (from the checkpoint metadata, `AIFSModel.latent_shape`): the
  hidden mesh is an O96 reduced-Gaussian grid, 40320 nodes x 1024 channels
  (`model_config.model.num_channels` in the checkpoint's embedded config) =
  ~41.3M elements, vs. the physical control's 2 (multi_step) x 542080 (N320
  grid) x 106 (variables) = ~114.9M elements -- about 2.8x fewer elements,
  smaller than the naive "40320 << 542080 nodes" comparison suggests because
  the hidden mesh's 1024 channels partly offset its 13x-fewer-nodes
  advantage. It's also a single fused embedding (not two duplicated lagged
  time levels) and already lives in a roughly homogeneous learned feature
  space, sidestepping the physical-units scale mismatch (~7 for v vs. ~98526
  for sp) that `state_scales` exists to correct.
- **Key architectural finding, and why the analysis is date-shifted**:
  `AnemoiModelEncProcDec.forward()` (anemoi-models==0.9.3) is not an
  autoencoder -- its decoder produces the *next* time level as a residual
  added onto the most recent input level (`_assemble_output`'s skip
  connection), not a reconstruction of the current two-level input. The
  encoder also re-embeds the physical state fresh on every single forward
  call; there's no persistent latent state carried across rollout steps the
  way NeuralGCM's encoded spectral state is. Consequently a latent-space
  control variable can only be injected as a one-time forcing at the *first*
  rollout step (see `AIFSModel._predict_step_with_grad_latent`,
  `AIFSModel.advance`'s `latent_increment` param, and
  `compute_loss_4dvar_latent`'s docstring for the exact mechanics) -- it
  cannot produce a corrected t=0 initial condition, only one at
  `t = window_start + dt_verif`. Since 2026-09-08, `compute_optimal_latent`
  materializes that real corrected state (a genuine `model.advance()` call
  using the best increment found, under `torch.no_grad()`) and returns it as
  an `AIFSState` dated at that shifted time, rather than returning the
  background unchanged -- see "real, date-shifted analysis" below for the
  full plumbing this required. Judge a latent-mode run's *convergence
  quality* from its loss-history JSON (does it converge faster/lower than an
  equivalent physical-space run) and, if useful, from the saved
  `*_latent_increment_*.pt` file (`{'increment', 'latent_scale'}`) for
  offline inspection of just the raw increment.
- No `state_scales`-style per-variable masking exists for latent mode
  (latent channels have no physical-variable identity to key on) -- just a
  single optional global `latent_scale` (default 1.0), divided into the
  increment inside the differentiable forward pass exactly like a
  `state_scales` factor. There's also no `NONNEGATIVE_VARS`-style clamp
  needed: a latent perturbation only reaches physical space through the
  decoder's own output, which already passes through the checkpoint's
  built-in per-variable `boundings` (`_assemble_output`) before
  `compute_loss_4dvar_latent` ever sees it.
- **Validated** with `smoke_test_latent_control.py` (repo root -- must run
  from there, like the real driver, so `aifs_inference.yaml`'s relative
  paths such as `aifs-single-2.0/lsm.grib` resolve) /
  `run_smoke_test_latent_control.sh`: confirms `latent_shape == (40320,
  1024)`, that gradient flows to a nonzero/finite `increment.grad` through
  the manually-unrolled encoder/processor/decoder path (both single-step and
  multi-step), and the decisive check from the physical-space smoke test's
  playbook -- AdamW actually reduces a toy loss (2t target, 15 epochs, real
  ~3.5% decrease, comfortably above the measured GPU-nondeterminism noise
  floor below). A real bug-hunting detour worth remembering: at
  `increment=0` (mathematically should be identical to the standard,
  already-validated non-latent path), the two paths' outputs differed by up
  to ~9.6 in absolute terms (all on the largest-magnitude fields -- z-levels,
  msl, sp, ~1e5 in magnitude; max *relative* error across every variable was
  ~2.1%, on `cos_mwd`). Traced this down in three steps rather than assuming
  either "fine" or "bug": (1) confirmed via `torch.isnan` that the model's
  own pre-existing wave/soil-moisture NaNs (`cdww`, `mwp`, `swh`, `swvl1/2`,
  `sd`, ... -- this cached test IC wasn't run through
  `test_forecast/patch_missing_wave_fields.py`) were identical in location
  and count between both paths, ruling those out; (2) sorted the remaining
  diff by *relative* error (not just absolute) across every one of the 106
  variables to rule out a structural bug hiding on a small-magnitude
  variable behind the large-magnitude fields' bigger absolute diffs -- found
  none (top relative errors were smoothly spread across many unrelated
  variable families -- q at 8 different levels, u/v at multiple levels,
  wave-direction sin/cos -- not concentrated in one suspicious column,
  which is the fingerprint of compounding bf16 rounding through many
  layers, not an indexing/broadcasting bug); (3) decisively confirmed by
  calling the **standard, unmodified** `advance()` path twice in a row on
  identical input with no latent code involved at all -- same ~5.5 max abs
  diff, proving this is ordinary call-to-call GPU/bf16 kernel-selection
  nondeterminism (consistent with CLAUDE.md's existing finding that
  finite-difference checks are noisy here), not anything about the new
  encoder/processor/decoder-splitting code.

### `control_space: 'latent'` real-DA-window results (2026-09-08)

Ran on the exact window `config_4day_50it.yml` uses (`sdate: 2015-01-01T00`,
`n_verif: 16`, `dt_verif: 6`, `weight_decay: 1.e-3`, `lam: 0`, `max_epoch:
100`) for a direct loss-curve comparison, config
`config_4day_100it_latent.yml`. `learn_rate` needed its own tuning pass, same
playbook as physical space's own lr history, and showed the identical
failure shape each time it was too high -- clean monotonic improvement for a
while, then a sudden blowup to NaN within 1-2 epochs (consistent with plain
AdamW step-size overshoot, not a slow drift):

| learn_rate | outcome | epochs before divergence | loss trajectory |
|---|---|---|---|
| `5.e-2` (smoke test's toy-loss value) | diverged | 3 good epochs | 283637 -> 205886 (27% reduction) before NaN at epoch 5 |
| `5.e-3` | diverged | 12 good epochs | 283838 -> 162209 (43% reduction) before NaN at epoch 14 |
| `1.e-3` | **stable, full 100 epochs** | -- | 284006 -> **70369** (75.2% reduction), monotonic every single epoch, still trending down at epoch 100 |

**`1.e-3` beats the physical-space baseline** (`config_4day_50it.yml`:
284033 -> 77673, 72.6% reduction, `learn_rate: 2.e-3` -- itself only reached
after extensive tuning, including the `state_scales: {z: 0.02, sp: 0.005}`
preconditioning fix documented above) -- a lower final loss, over the same
100 epochs, with **zero per-variable preconditioning** (no `state_scales`
analog needed at all in latent mode). This is real, decisive confirmation of
the original motivation (better-conditioned, lower-dimensional control
space -- see the architecture section above): latent-space optimization
converges as well as or better than the heavily-tuned physical-space path,
"for free," modulo needing its own single scalar `learn_rate` tuned (no
surprise there -- every `control_space` needs *some* lr tuning).
Diverged-attempt outputs kept for the record:
`output/test_aifs_4day_100it_latent_lr5e-2_diverged/`,
`.../test_aifs_4day_100it_latent_lr5e-3_diverged/`; the stable run is
`output/test_aifs_4day_100it_latent_lr1e-3/`.

At the time of this run, `compute_optimal_latent` was still diagnostic-only
(returned the background unchanged), so `z500err before`/`after` were
identical by construction. That's since been fixed -- see the next section.

### `control_space: 'latent'` real, date-shifted analysis (2026-09-08)

With the optimization side validated (previous section), extended
`compute_optimal_latent` to return a genuinely corrected `AIFSState` --
enabling multi-window/multi-cycle DA with `control_space: 'latent'`, not
just single-window loss-curve comparisons. Since a latent-space analysis is
only ever valid starting at `window_start + dt_verif` (not `window_start`
itself -- see the architectural finding above), `compute_optimal` (both
branches now) returns an `AIFSState` (bundling `.state` + the date it's
*actually* valid at) instead of a bare tensor -- this is the fix, not just a
rename: every downstream consumer that used to do `fix_inputs =
copy.copy(input_encoded); fix_inputs.state = params_best` (silently assuming
the analysis shares the background's date) was auditing a wrong assumption
that only happened to be harmless for the physical path.

- **`compute_optimal_latent`**: after the epoch loop, makes one final
  `torch.no_grad()`, `use_checkpoint=False` call --
  `model.advance(input_encoded, steps=dt_verif // timestep_hours,
  latent_increment=increment_best / latent_scale)` -- to materialize the
  real corrected state (or the same call with `latent_increment=None`, an
  uncorrected forecast-forward, if no epoch ever improved -- keeps the
  contract "this always returns a state dated `+dt_verif`" unconditional).
  Its `.date` comes out correct automatically since `AIFSModel.advance`
  already accumulates `date += timestep` per internal step.
- **Every consumer of the old `params_best` tensor** (long_window_4dvar.py's
  driver, `save_trajectory_diagnostics`, `save_inputs_pkl`, `save_inputs_nc`,
  `make_forecasts`) now takes the `AIFSState` directly (renamed
  `analysis_state` throughout) instead of stapling a bare tensor onto
  `input_encoded`'s date -- a net simplification, not just a bug fix (it
  deletes the `copy.copy`/`.state = ...` dance at every one of those call
  sites).
- **Cycling loop** (`long_window_4dvar.py`, the `dt_init`-cycling branch):
  now advances from `analysis_state` by only the *remaining* hours to reach
  the next cycle's date (`dt_init - (analysis_state.date -
  input_encoded.date)`), not the full `dt_init` -- advancing the full
  `dt_init` from an already-`+dt_verif`-shifted state would double-count the
  shift. Physical mode is unaffected (shift is always 0 there). Raises a
  clear `ValueError` if `dt_init <= dt_verif` (nothing left to advance) --
  a real config constraint for `control_space: 'latent'` cycling that didn't
  exist for physical mode.
- **`compute_ps_observation_hx`** gained an `oind_start` param so
  `save_trajectory_diagnostics` can correctly align a shifted-date analysis
  trajectory against `psobs_traj`'s oind indexing (oind 0 is the
  window-start slot; a latent analysis only starts existing at oind
  `dt_verif // timestep_hours`, i.e. oind 1 for every config used so far
  since `dt_verif == timestep == 6h`). The resulting shorter arrays are
  padded with a leading NaN (0 for the int-typed `used`/`qc_flag` fields)
  block before being stacked into the saved `observation_diagnostics.nc`
  Dataset, so there's no analysis-side O-A number at the padded slots
  (physically correct -- no analysis exists yet there) instead of silently
  duplicating the background or comparing against the wrong time.
- **`printz500err`'s "after" line**: skipped (prints an explicit `N/A ...`
  message) whenever `analysis_state.date != input_encoded.date`, since the
  only ERA5 truth fetched (`verif_ic`) is at the window-start date --
  comparing a `t+dt_verif` analysis against it directly would silently
  conflate real forecast evolution with analysis error, which is worse than
  reporting nothing. Fetching ERA5 truth at an arbitrary shifted date (to
  make this a real number again) is a separate, larger feature, **not**
  implemented -- noted below as a follow-up gap.
- **Validated end-to-end** (2026-09-08) with two real runs, both clean (no
  crashes, correct dates throughout): (1) a single-cycle re-run of
  `config_4day_100it_latent.yml` confirmed via direct inspection of the saved
  `.pkl`/`.nc` outputs -- `analysis_state.date` correctly `input_encoded.date
  + 6h`, a genuinely different (not accidentally-unchanged) corrected state,
  `observation_diagnostics.nc`'s analysis-side fields correctly NaN-padded at
  oind 0 and populated from oind 1 on, matching the training log's own obs
  counts exactly. (2) `config_4day_100it_latent_n_init2.yml`, `n_init: 2`,
  `dt_init: 6` (== `dt_verif`, the `remaining_hours == 0` no-op-advance case)
  -- cycle 2 correctly started at `2015-01-01T06` (`analysis_state.date`
  from cycle 1, no double-shift), reused the ERA5 verification snapshot
  prefetched for that date (`aifs_prefetch_ic.py`, run from login node
  `ufe04` -- H100 nodes have no internet), and completed its own 100-epoch
  optimization cleanly.

### `reset_skt_ocean` -- resetting ocean surface temperature from ERA5 every analysis time (2026-09-09)

Added at the user's request to reset SST/sea-ice at every DA cycle instead
of letting the model carry its own predicted values forward across cycles.
Investigating what "SST/sea-ice" actually means for this checkpoint turned
into a real finding worth recording:

- **This checkpoint has no `sst` or sea-ice variable at all.** Confirmed
  three independent ways: not in `AIFSModel.var_to_idx` (the 106-variable
  runtime input tensor), not in `checkpoint.typed_variables` (134 variables,
  includes output-only diagnostics too), and not anywhere in the full
  checkpoint metadata blob (~249KB of JSON: config, dataset, training
  provenance -- `sst` appears zero times). This is despite the vendored
  `aifs-single-2.0/README.md` (line 196) listing "sea-surface temperature
  (SST), skin temperature (SKT)" together as a "Both (Prognostic))"
  variable -- a real discrepancy between that table and what's actually
  baked into `aifs-single-mse-2.0.ckpt`, not something resolved from this
  side (only one checkpoint file exists in that repo, so it's not a
  wrong-file issue).
- `skt`'s own MARS metadata (`checkpoint.typed_variables['skt'].mars`)
  resolves to `param: skt`, not aliased to `sst` -- and no merge/combine
  filter for skt/sst exists anywhere in `aifs_ic.py`, `aifs_inference.yaml`,
  or the installed `anemoi-inference`/`anemoi-transform`/`anemoi-datasets`
  packages (grepped the whole install). So if `skt` and `sst` are combined
  via the land-sea mask (per the user's hypothesis, and ECMWF's well-known
  operational convention: `skt` over open ocean is defined as the SST
  analysis, blended via lsm), that blending happens upstream of this entire
  pipeline, in ECMWF's own operational/ERA5 archive -- there's no trace of
  it in any code we have, because by the time this pipeline ever requests
  `param=skt`, it's already the merged product.
- Given that, `skt` is the only available proxy, and there's no sea-ice
  variable to handle separately at all -- `lsm` (the mask used to pick
  "ocean" points) is itself static/constant-in-time, with no time-varying
  ice-extent field in this checkpoint's 106-134 variables regardless.

**Implementation**: `long_window_4dvar_utils.reset_skt_over_ocean(model,
aifs_state, verif_ic, lsm_threshold=0.5)` overwrites `skt` at grid points
where `lsm < lsm_threshold` (both lagged time levels) with `verif_ic['skt']`
(the ERA5 truth already fetched by `get_verif` for the window/cycle's date
-- no extra fetch needed), leaving land `skt` untouched. Gated by `exp:`
config keys `reset_skt_ocean` (bool, default `False`) and
`skt_ocean_lsm_threshold` (default `0.5`) -- top-level `exp:`, not
per-window, same reasoning as `control_variables`/`latent_scale`
(`get_window()` only copies a hardcoded key whitelist from `windows:` into
`exp` -- `dt_obs`, by contrast, IS per-window, read directly out of
`windows[window]` in `get_window()`).

Applied in `long_window_4dvar.py` only at genuine **new analysis times** --
the `k==0`/restart branch and the `dt_init`-cycling branch (both set a local
`is_new_analysis_time = True`) -- NOT the window-to-window-within-one-cycle
`else` branch (`is_new_analysis_time = False`; no new ERA5 truth exists yet
there). `get_input()`'s very first background is itself a model-forecasted
state (spun up from `nhrs_back` hours before `sdate`), so the reset is
meaningful there too, not just on later cycles.

Validated with a standalone CPU script (no GPU job needed -- pure tensor
mechanics, deleted after use): confirmed ocean `skt` at both time levels
exactly matches the ERA5 `verif_ic` values post-reset, land `skt` is
bit-identical to the pre-reset background, the original `aifs_state` object
passed in is untouched (the function clones, doesn't mutate in place), and
the reset is a genuine change, not an accidental no-op (n_ocean=384934,
n_land=157146 out of 542080 N320 grid points -- a plausible global
ocean/land split). Not yet run through a real DA cycle end-to-end with
`reset_skt_ocean: True` set (only the tensor-level mechanics are verified so
far) -- worth doing before relying on it for a real experiment.

### Merge to mainline: `dt_obs` / `control_variables`, retiring the physical control path (2026-09-11)

`long_window_4dvar.py` / `long_window_4dvar_utils.py` were developed for a
while (September 2026) in parallel `_dtobs`-suffixed files
(`long_window_4dvar_dtobs.py` / `long_window_4dvar_utils_dtobs.py`) so an
in-flight mainline latent experiment could keep running unaffected. That
prototype was validated (a 20-cycle real run, job 21155155,
`config_ctlvars_20cycle.yml`, COMPLETED 2026-09-11, matching the mainline's
earlier 20-cycle segment except for the new `control_variables` masking)
and confirmed good, so it has now been promoted: **the `_dtobs` files'
content IS `long_window_4dvar.py`/`long_window_4dvar_utils.py`** (copied
over, with only the module docstrings/comments reworded from "variant"
framing to describe the normal, permanent behavior -- no functional
change). `run_long_window_4d.sh` was reverted to invoke the unsuffixed
`long_window_4dvar.py` (it had temporarily been pointed at
`long_window_4dvar_dtobs.py config_ctlvars_20cycle.yml` for that test).

What this merge actually changed, relative to the pre-2026-09-11 mainline:

- **Physical control space is gone.** The `control_space` config key, the
  `physical`/`compute_optimal`/`compute_loss_4dvar` vs.
  `latent`/`compute_optimal_latent`/`compute_loss_4dvar_latent` split, and
  the `state_scales`/`control_mask`/`NONNEGATIVE_VARS` machinery that
  existed only to precondition/gate the physical-space increment are all
  deleted. `compute_optimal`/`compute_loss_4dvar` are now unconditionally
  the latent-space implementation (what older text in this file, including
  just above, calls `compute_optimal_latent`/`compute_loss_4dvar_latent`).
  The analysis these return is unconditionally an `AIFSState` dated
  `window_start + dt_verif` -- there is no more "physical mode" branch
  where it shares the background's date (the driver's z500-error block
  reflects this: it always fetches a date-shifted ERA5 truth and advances
  the background to match, no `if analysis_state.date == input_encoded.date`
  branch any more).
- **`ps_operator: 'mslp'` is gone** (the GraphCast-era alternate forward
  operator that read mslp/700hPa fields instead of AIFS's own `sp` -- AIFS
  has a native `sp` variable, so it was never the default and had gone
  untested for a while; only `'ps'` and `'logpinterp'` remain).
- **New: `dt_obs`** (per-window config key, hours, default `dt_verif`;
  must divide both `dt_verif` and AIFS's fixed 6h timestep) samples ps
  observations finer than the 6h rollout. An obs between two 6h model
  states has its forward-operator input fields (`loss_variables`, default
  = whatever the active `ps_operator` reads) linearly interpolated in time
  between the bracketing decoded states. **Validated finding: interpolated
  (off-step) slots fit measurably worse than on-step slots** -- traced to
  genuine linear-time-interpolation error from the semidiurnal (S2)
  atmospheric pressure tide (6h is exactly half an S2 cycle), not an obs-
  population or indexing artifact; worse in the tropics (~1.17 hPa
  residual) than at the poles (~0.55 hPa). **Left unmitigated for now by
  user decision** -- see "Known gaps" below for the recorded mitigation
  options if this becomes worth revisiting. `dt_obs == dt_verif` (the
  default) reproduces the pre-merge behavior exactly (regression-tested).
- **New: `control_variables`** (top-level `exp:` config key, list of AIFS
  variable/family base names, e.g. `[u, v, t, z, sp]`) is a decode-time
  mask on the latent control: at the injection step, packed-state columns
  NOT named in `control_variables` are overwritten with the uncorrected
  first-step forecast, zeroing the increment's *direct* effect on them
  (they can still respond indirectly, through model dynamics, to the
  columns that *are* controlled). Absent -> unrestricted (every column
  controllable), the default. This plays a role similar to the old
  physical-mode `state_scales`/`control_mask` (deciding which variables the
  4D-Var analysis is allowed to touch) but is a hard include/exclude list,
  not a continuous per-variable gradient rescaling -- latent mode has no
  physical-units scale mismatch to precondition against in the first place
  (see the "`control_space: 'latent'`" section above), so there's no
  `state_scales` analog needed, only this masking. Validated end-to-end on
  GPU (job 21151256): controlled variables show large, real corrections;
  every non-controlled prognostic variable sits at or below the documented
  bf16 call-to-call nondeterminism floor (confirming no leakage through the
  mask); a lookup-order bug (checking `var_to_idx` before
  `_levels_by_base`, the same class of bug `state_scales` had) was caught
  and fixed before that validation, not after.
- Config files written before this merge that still set `control_space:
  'latent'`, `state_scales`, or `ps_operator: mslp` are not broken by it --
  those keys are simply ignored now (not validated/rejected) -- but new
  configs shouldn't set them.
- **Not yet done, left for a deliberate follow-up**: the `_dtobs`-suffixed
  files themselves (now byte-for-byte superseded duplicates) and the
  `_dtobs`/`ctlvars`-specific test configs (`config_dtobs3_test.yml`,
  `config_dtobs6_check.yml`, `config_ctlvars_test.yml`, `run_dtobs_test.sh`)
  have **not** been deleted -- a job (21188374) that references
  `long_window_4dvar_dtobs.py`/`config_ctlvars_20cycle.yml` directly (its
  spooled copy of `run_long_window_4d.sh`, from before this merge) was
  still running at merge time; clean those up once it finishes, to avoid
  future confusion about which files are canonical.

### Known gaps / next steps

- **`reset_skt_ocean` has now been run through a real multi-cycle DA run**
  (`reset_skt_ocean: True` in both the original mainline 20-cycle run and
  the `control_variables` 20-cycle comparison, job 21155155, completed
  2026-09-11) -- ran clean across all 20 cycles. Its specific effect on
  ps-obs innovation stats near coastlines hasn't been isolated/quantified
  (would need an otherwise-identical run with `reset_skt_ocean: False` to
  diff against), but it's no longer an unvalidated code path.
- The `n_init: 2` validation above used `dt_init: 6 == dt_verif`, exercising
  the `remaining_hours == 0` no-op-advance branch in the cycling loop --
  `remaining_hours > 0` (a real, nonzero `model.advance(analysis_state,
  steps=steps_for_hours(remaining_hours))` call, e.g. `dt_init: 12` with
  `dt_verif: 6`) is still logically straightforward but not yet exercised by
  an actual run. The second cycle's `get_verif` ERA5 snapshot for the new
  cycle date needs to be cached first (`aifs_prefetch_ic.py` from a login
  node -- H100 nodes have no internet); check `ic_cache/` for what's already
  there before assuming a prefetch is needed.
- Fetching ERA5 truth at an arbitrary shifted date (to give `printz500err`'s
  "after" line and a real O-A-at-t+dt_verif number back for latent mode,
  instead of the current "N/A") is unimplemented -- would need extending
  `get_verif`/`aifs_ic.py` to accept an arbitrary date rather than always
  the window-start date.
- Now that `control_space: 'latent'` is confirmed to out-converge the
  physical-space baseline at `learn_rate: 1.e-3` (see above), worth pushing
  `learn_rate` above `1.e-3` (below the `5.e-3` that diverged) to see if it
  can go further while staying stable, same escalation approach physical
  space's own tuning used.
- One DA cycle / 16-verif-step (4-day) / 100-epoch test
  (`config_4day_50it.yml`) is the largest run so far, and the best result to
  date: `learn_rate: 2.e-3`, `weight_decay: 1.e-3`, `lam: 0`,
  `state_scales: {z: 0.02, sp: 0.005}` (with both `state_scales` bugs above
  fixed), 100 epochs, no divergence -- loss fell monotonically from 284033 to
  77673 and was still decreasing at epoch 100 (not plateaued). Ps-obs
  innovation stats: n_used ~7600-9800/9100-10200 per verif time, rms(O-B)
  1.1-2.7 hPa growing with lead time as expected, rms(O-A) a *flat*
  ~0.9-1.2 hPa at every lead time (53.4% aggregate RMS reduction, the best
  of any run so far, and notably the O-A curve no longer grows with lead
  time the way O-B does). z500 verification error is unchanged/a hair worse
  despite a confirmed-real correction there (see debugging notes above) --
  this is the main open scientific question, not an implementation bug at
  this point.
- Learning rate has gone `1.e-4` (stable, small effect) -> `1.e-3` (33.8%
  RMS reduction) -> `2.e-3` (53.4% RMS reduction, still improving at epoch
  100, still no divergence) across successive user-directed increases --
  it's plausible further increases keep helping and haven't found a ceiling
  yet.
- `n_init: 100` (config.yml.template's full production setting) hasn't been
  attempted -- `aifs_prefetch_ic.py` needs to warm the cache for the whole
  100-cycle date range first (100 CDS fetches).
- `plot_4dvar_ngcm.py`-style downstream plotting isn't ported — the saved
  netCDF now uses a flat unstructured `values` dimension (with `latitude`/
  `longitude` as 1D coordinate arrays on it) plus per-variable-family level
  dimensions (`level_z`, `level_t`, ...), not NeuralGCM's regular
  `(level, longitude, latitude)` grid.
- **`dt_obs` sub-6h interpolation's tidal representativeness error is left
  unmitigated** (see "Merge to mainline" above) -- recorded mitigation
  options if it's ever worth revisiting: (1) latitude-dependent oberr
  inflation `sqrt(oberr^2 + resid(lat)^2)` for `alpha != 0` slots in
  `get_psobs`; (2) drop interp slots at `|lat| < ~20`; (3) quadratic 3-point
  time interpolation (t-6/t/t+6), which roughly halves the tidal error;
  (4) a local mean+S1+S2 harmonic fit to the 4 nearest on-step states (6h
  is exactly Nyquist for S2), which essentially removes it but couples 4
  rollout steps per obs term.
- **`dt_obs=3`'s `learn_rate` hasn't been bumped past `1.e-3`** -- confirmed
  stable there (just converges somewhat slower, relative to `dt_obs=6`, than
  it does at `dt_obs=6`) but not pushed toward a divergence ceiling the way
  the latent baseline's own `learn_rate` was.
- **Cleanup of the now-superseded `_dtobs`-suffixed files and their
  dedicated test configs** (`long_window_4dvar_dtobs.py`,
  `long_window_4dvar_utils_dtobs.py`, `config_dtobs3_test.yml`,
  `config_dtobs6_check.yml`, `config_ctlvars_test.yml`, `run_dtobs_test.sh`)
  is deferred until job 21188374 (still running at merge time, and built
  from a `run_long_window_4d.sh` spooled copy that references those
  filenames directly) finishes -- see "Merge to mainline" above.
