"""Summarize innovation statistics (O-B, O-A) and loss history for the
4-day / 50-epoch AIFS 4D-Var test run.
"""
import glob
import json

import numpy as np
import xarray as xr

OUT_DIR = "output/test_aifs_4day_100it_lr2e-3_lam0_precond_fixed/"

diag_files = sorted(glob.glob(OUT_DIR + "*_observation_diagnostics_*.nc"))
assert diag_files, "no observation_diagnostics.nc found"
ds = xr.open_dataset(diag_files[-1])
print("Loaded:", diag_files[-1])
print()

used = ds["surface_pressure_used"].values.astype(bool)
avail = ds["surface_pressure_available"].values.astype(bool)
omb = ds["surface_pressure_omb"].values
oma = ds["surface_pressure_oma"].values
lead = ds["lead_time_hours"].values

print(f"{'lead(h)':>8} {'n_avail':>8} {'n_used':>8} {'mean(O-B)':>10} {'rms(O-B)':>10} {'mean(O-A)':>10} {'rms(O-A)':>10}")
for i, h in enumerate(lead):
    m = used[i]
    n_avail = int(avail[i].sum())
    n_used = int(m.sum())
    if n_used == 0:
        print(f"{h:8d} {n_avail:8d} {n_used:8d} {'--':>10} {'--':>10} {'--':>10} {'--':>10}")
        continue
    mb, rb = np.nanmean(omb[i, m]), np.sqrt(np.nanmean(omb[i, m] ** 2))
    ma, ra = np.nanmean(oma[i, m]), np.sqrt(np.nanmean(oma[i, m] ** 2))
    print(f"{h:8d} {n_avail:8d} {n_used:8d} {mb:10.3f} {rb:10.3f} {ma:10.3f} {ra:10.3f}")

print()
m_all = used
mb_all = np.nanmean(omb[m_all])
rb_all = np.sqrt(np.nanmean(omb[m_all] ** 2))
ma_all = np.nanmean(oma[m_all])
ra_all = np.sqrt(np.nanmean(oma[m_all] ** 2))
print(f"ALL TIMES: n_used={int(m_all.sum())}  mean(O-B)={mb_all:.3f}  rms(O-B)={rb_all:.3f}  "
      f"mean(O-A)={ma_all:.3f}  rms(O-A)={ra_all:.3f}")
print(f"RMS reduction: {100*(1 - ra_all/rb_all):.1f}%")

print()
loss_files = sorted(glob.glob(OUT_DIR + "*_loss_*.json"))
with open(loss_files[-1]) as f:
    history = json.load(f)
losses = [h["loss"] for h in history]
print(f"Loss history: epoch 1 = {losses[0]:.2f}, epoch {len(losses)} = {losses[-1]:.2f}, "
      f"min = {min(losses):.2f} (epoch {losses.index(min(losses))+1})")
