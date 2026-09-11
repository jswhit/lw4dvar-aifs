"""
Z500 error as a function of lead time WITHIN the 4D-Var window (0, 6, ...,
96h), averaged across every DA cycle of an experiment -- complements
z500err_ts.py, which only has the single lead time (window_start + dt_verif)
long_window_4dvar.py's own printz500err logs.

Recomputes directly from two sources per cycle, instead of the driver's log:
  - the model side: `<date>_control_forecast_*.nc` / `_optimal_forecast_*.nc`
    (saved by save_xr_trajectory -- the full-window decoded trajectory on
    AIFS's native unstructured grid, 'z' variable at level_z=500).
  - the truth side: ERA5 geopotential at 500 hPa, read directly from the
    cached GRIB files in ic_cache/ (aifs_ic.read_single_date_fields) at each
    lead time's valid date -- already regridded to the model's own grid by
    the CDS fetch (same grid `read_single_date_fields` is used for
    everywhere else in this codebase, e.g. get_verif()), so no extra
    regridding step is needed here either.

No AIFSModel/checkpoint/GPU load needed: lat/lon come straight from the
saved netCDF's own 'latitude'/'longitude' coordinates (written from
model.lats/model.lons when the file was saved), and
aifs_ic.read_single_date_fields's `runner` argument is only ever touched on
a cache MISS (see aifs_ic.fetch_era5_grib) -- passing `runner=None` is safe
whenever ic_cache/ is already warm for the dates needed, which is the
common case for an experiment whose driver job already ran to completion.
A cache miss raises a clear error naming the missing date instead of
silently guessing.
"""

import glob
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

# aifs_ic.py lives at the repo root, one level up from this diagnostics/
# script -- add it to sys.path so `import aifs_ic` resolves regardless of
# the working directory this is invoked from (relative paths used below,
# e.g. './ic_cache/', still assume repo root as the CWD -- see CLAUDE.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import aifs_ic

GRAV = 9.80665
_DATE_RE = re.compile(r'^(\d{4}-\d{2}-\d{2}T\d{2})_control_forecast_(.+)\.nc$')


def add_hours(date_str, nhours):
    d = datetime.strptime(date_str, '%Y-%m-%dT%H')
    return (d + timedelta(hours=int(nhours))).strftime('%Y-%m-%dT%H')


def _parse_nc_date(value):
    """`trajectory_start_date` attr ('2015-01-01 00:00:00', from
    str(datetime...) in save_xr_trajectory) -> '%Y-%m-%dT%H' string."""
    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%dT%H')


def getrms(diff, coslat):
    """Region-masked RMS, NaN-safe against a diverged/missing `diff` (not
    just an out-of-region `coslat`): `nansum` of an all-NaN slice silently
    returns 0.0, not NaN, so masking only via `coslat` (which never has NaN
    where `diff` does) would misreport a fully-NaN forecast as a perfect
    0.0 RMS error instead of "no data" -- discovered from a real diverged
    cycle, see CLAUDE.md. Points where `diff` is NaN are excluded from BOTH
    the numerator and the weight-sum denominator, not just zeroed.
    """
    weight = np.where(np.isnan(diff), np.nan, coslat)
    denom = np.nansum(weight)
    if not denom > 0:
        return np.nan
    return np.sqrt(np.nansum(weight * diff ** 2) / denom)


def region_masks(lats):
    coslat = np.cos(np.radians(lats))
    return {
        'NH': np.where(lats > 20., coslat, np.nan),
        'Tropics': np.where((lats >= -20) & (lats <= 20), coslat, np.nan),
        'SH': np.where(lats < -20., coslat, np.nan),
        'Global': coslat,
    }


_verif_cache = {}


def get_z500_truth(date_str, cache_dir):
    """ERA5 z (geopotential) at 500 hPa, raw (n_points,) array already on
    the model's native grid (see module docstring). Cached in-process since
    the same valid date is shared by many cycles/lead-times."""
    if date_str not in _verif_cache:
        date_dt = datetime.strptime(date_str, '%Y-%m-%dT%H')
        try:
            fields = aifs_ic.read_single_date_fields(None, date_dt, cache_dir)
            _verif_cache[date_str] = fields['z_500']
        except Exception as e:
            raise RuntimeError(
                f"no cached ERA5 GRIB for {date_str} in {cache_dir} ({e}) -- "
                f"prefetch it from a login node first (aifs_prefetch_ic.py), "
                f"this script never fetches over the network itself."
            )
    return _verif_cache[date_str]


def z500_error_curve(nc_path, cache_dir):
    """(start_date, lead_hours, {region: rms_err}, n_diverged) for one saved
    *_forecast_*.nc file. `lead_hours` is relative to THIS file's own
    `trajectory_start_date` -- for an `_optimal_forecast_` file that's
    `window_start + dt_verif` (the latent shift), NOT the window start, so
    callers comparing control vs. optimal on one absolute-lead-time axis
    must add that shift themselves (see experiment_mean_curves) rather than
    treating both files' lead_hours as directly comparable.
    """
    ds = xr.open_dataset(nc_path)
    start_date = _parse_nc_date(ds.attrs['trajectory_start_date'])
    lats = ds['latitude'].values
    masks = region_masks(lats)
    z500 = ds['z'].sel(level_z=500).values  # (time, values)
    ntime = z500.shape[0]
    lead_hours = np.arange(ntime) * 6  # AIFS's native fixed timestep
    errs = {r: np.full(ntime, np.nan) for r in masks}
    n_diverged = 0
    for t in range(ntime):
        vdate = add_hours(start_date, lead_hours[t])
        truth = get_z500_truth(vdate, cache_dir)
        diff = (z500[t] - truth) / GRAV
        if np.isnan(diff).all():
            n_diverged += 1
        for r, mask in masks.items():
            errs[r][t] = getrms(diff, mask)
    ds.close()
    return start_date, lead_hours, errs, n_diverged


def experiment_mean_curves(expt_dir, cache_dir):
    """Average control/optimal z500 error curves across every cycle found in
    expt_dir. Returns (lead_hours, {'control': {region: mean_err},
    'optimal': {region: mean_err}}, n_cycles) -- 'optimal' curves are
    shorter (see save_trajectory_diagnostics: the latent analysis trajectory
    starts dt_verif later than the background one) and averaged only where
    at least one cycle has data at that lead time.
    """
    control_files = sorted(glob.glob(os.path.join(expt_dir, '*_control_forecast_*.nc')))
    regions = ['NH', 'Tropics', 'SH', 'Global']
    accum = {'control': {r: [] for r in regions}, 'optimal': {r: [] for r in regions}}
    n_cycles = 0
    diverged_cycles = []
    for cf in control_files:
        m = _DATE_RE.match(os.path.basename(cf))
        if not m:
            continue
        date, suffix = m.groups()
        of = os.path.join(expt_dir, f'{date}_optimal_forecast_{suffix}.nc')
        if not os.path.exists(of):
            print(f'  warning: no matching optimal_forecast for {cf}, skipping cycle')
            continue
        start_c, lead_c, errs_c, ndiv_c = z500_error_curve(cf, cache_dir)
        start_o, lead_o, errs_o, ndiv_o = z500_error_curve(of, cache_dir)
        if ndiv_c or ndiv_o:
            diverged_cycles.append(date)
            print(f'  cycle {date}: DIVERGED (NaN geopotential at {ndiv_c} background / '
                  f'{ndiv_o} analysis lead times) -- excluded from the mean at those lead times')
        else:
            print(f'  cycle {date}...')
        # `lead_o` is relative to the optimal file's own (shifted) start
        # date -- shift it onto the control file's (window-start) absolute
        # lead-time axis before the two are compared/averaged together.
        shift_hours = (
            datetime.strptime(start_o, '%Y-%m-%dT%H') - datetime.strptime(start_c, '%Y-%m-%dT%H')
        ).total_seconds() / 3600.0
        lead_o_abs = lead_o + round(shift_hours)
        for r in regions:
            accum['control'][r].append((lead_c, errs_c[r]))
            accum['optimal'][r].append((lead_o_abs, errs_o[r]))
        n_cycles += 1
    if diverged_cycles:
        print(f'  ** {len(diverged_cycles)}/{n_cycles} cycles diverged to NaN: {diverged_cycles} **')

    def _mean_over_cycles(curves, all_lead_hours):
        # curves: list of (lead_hours, err) pairs, possibly different lengths
        # (optimal trajectories are shorter/shifted) -- average by lead hour
        # value, not by array index.
        out = np.full(all_lead_hours.shape, np.nan)
        for i, h in enumerate(all_lead_hours):
            vals = [v for lead, err in curves if h in lead for v in [err[lead == h][0]] if not np.isnan(v)]
            if vals:
                out[i] = np.mean(vals)
        return out

    max_lead = max((lead.max() for r in regions for lead, _ in accum['control'][r]), default=0)
    all_lead_hours = np.arange(0, max_lead + 1, 6)
    mean_curves = {kind: {r: _mean_over_cycles(accum[kind][r], all_lead_hours) for r in regions}
                   for kind in ('control', 'optimal')}
    return all_lead_hours, mean_curves, n_cycles


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    # python diagnostics/z500err_window.py [label=dir ...]  (run from the
    # repo root -- relative paths like ic_cache/ below assume it).
    # {label -> output dir}, ic_cache assumed at ./ic_cache/ for each.
    # Defaults to the mainline reset_skt_ocean cycling experiment (see
    # CLAUDE.md). Override with `label=dir` args, e.g.:
    #   python diagnostics/z500err_window.py \
    #       'lr=1e-3=output/test_aifs_ctlvars_n_init20_50it' \
    #       'lr=2e-3=output/test_aifs_ctlvars_n_init20_50it_lr2e-3'
    # (that lr=2e-3 run diverged and was killed -- see CLAUDE.md's
    # "learn_rate sweep" section -- so its curve will mostly be gaps.)
    if len(sys.argv) > 1:
        expts = {}
        for arg in sys.argv[1:]:
            # rsplit (not split) on the LAST '=' -- a label like 'lr=1e-3'
            # is a very natural thing to want, and directory paths never
            # contain '='.
            label, path = arg.rsplit('=', 1)
            expts[label] = path
    else:
        expts = {
            'lr=1.e-3': 'output/test_aifs_latent_reset_skt_n_init20_50it',
#            'lr=2.e-3 (current)': 'output/test_aifs_ctlvars_n_init20_50it_lr2e-3',
        }
    cache_dir = './ic_cache/'

    regions = ['NH', 'Tropics', 'SH', 'Global']
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    any_data = False
    for (label, expt_dir), color in zip(expts.items(), colors):
        if not os.path.isdir(expt_dir):
            print(f'warning: {expt_dir} not found, skipping {label!r}')
            continue
        print(f'{label} ({expt_dir}):')
        lead_hours, curves, n_cycles = experiment_mean_curves(expt_dir, cache_dir)
        if n_cycles == 0:
            print(f'  no complete cycles found yet (job may still be running)')
            continue
        any_data = True
        print(f'  averaged over {n_cycles} cycles')
        for ax, region in zip(axes, regions):
            ax.plot(lead_hours, curves['control'][region], '--', color=color,
                     label=f'{label} background (n={n_cycles})')
            ax.plot(lead_hours, curves['optimal'][region], '-', color=color,
                     label=f'{label} analysis (n={n_cycles})')

    for ax, region in zip(axes, regions):
        ax.set_ylabel('%s Z500\nrms err (m)' % region)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('lead time within window (h)')
    fig.suptitle('Z500 error growth within window, averaged across DA cycles (AIFS-single-2.0)')
    fig.tight_layout()
    fig.savefig('z500err_window.png')
    print('wrote z500err_window.png' if any_data else 'wrote z500err_window.png (empty -- no complete cycles yet)')
