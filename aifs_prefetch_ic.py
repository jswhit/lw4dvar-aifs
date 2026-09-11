"""
Warm the ERA5 IC/verification cache (aifs_ic.py) for a planned
long_window_4dvar.py run, BEFORE submitting the job to the (offline) H100
partition via run_long_window_4d.sh.

get_input()/get_verif() fetch from CDS (needs internet); the H100 compute
nodes have none. Run this from a login node (e.g. ufe04) first:

    conda activate /scratch4/BMC/gsienkf/whitaker/conda/envs/aifs2
    python -u aifs_prefetch_ic.py
    sbatch run_long_window_4d.sh

Fetches, in order:
  - (non-restart only) the initial "nhrs_back"-hours-before-sdate lagged state
    (only needed once, for k=0's get_input() -- subsequent cycles advance the
    previous analysis forward with the model instead of fetching a new IC)
  - one single-date verification state per DA cycle (get_verif() is called
    for every cycle/window, always at that cycle's `date`)
Both are written to `ic_cache/` (or `exp.ic_cache`, if set), matching exactly
what long_window_4dvar.py will look for -- a warm cache means the actual job
never touches the network.

For a restart run (`restart: True`) the driver reads the analysis pkl named
`<sdate>_optimal_inputs_*` and then advances `date` by `dt_init` *before* its
first get_verif() -- so the verification states it needs run from
`sdate + dt_init` through `sdate + n_init*dt_init`, not from `sdate`. This
script mirrors that shift; no lagged IC is fetched (restart doesn't build one).

For the latent space update the driver also scores its z500 before/after
diagnostic at `cycle_date + dt_verif` (where the latent analysis is actually
valid), so this script additionally fetches ERA5 truth there for every cycle
and window `dt_verif`. When `dt_verif == dt_init` these coincide with the next
cycle's date -- only the trailing one past the last cycle is genuinely new.
"""

import sys

import long_window_4dvar_utils as utils
import aifs_ic

logger = utils.get_logger()

exp_config, windows = utils.load_config()
cache_dir = exp_config.get('ic_cache', exp_config['path_output'] + 'ic_cache/')

logger.info('loading AIFS checkpoint (device=cpu; only used for its variable/grid metadata here)...')
exp_config_cpu = dict(exp_config)
exp_config_cpu['device'] = 'cpu'
model = utils.get_model(exp_config_cpu)
nhrs_back = exp_config['nhrs_back']
if len(sys.argv) > 1:
    date = sys.argv[1]
    n_init = int(sys.argv[2])
else:
    date = exp_config['sdate']
    n_init = exp_config['n_init']

if not exp_config['restart']:
    from datetime import datetime
    back_date = utils.add_hours(date, -nhrs_back)
    logger.info(f'prefetching initial condition: {back_date} (lagged)')
    back_dt = datetime.strptime(back_date, '%Y-%m-%dT%H')
    aifs_ic.fetch_era5_grib(model.runner, back_dt, cache_dir, lagged=True)
else:
    # driver's restart branch advances `date` by dt_init before its first
    # get_verif() -- match that so the fetched verif dates line up.
    date = utils.add_hours(date, exp_config['dt_init'])
    logger.info(f'restart: verification states run from {date} '
                f'(sdate + dt_init) through {utils.add_hours(date, (n_init - 1) * exp_config["dt_init"])}')

from datetime import datetime

# latent space update needs ERA5 truth at cycle_date +
# dt_verif for each window (its z500 before/after diagnostic is scored there).
dt_verifs = sorted({w['dt_verif'] for w in windows.values()})

for k in range(n_init):
    logger.info(f'prefetching verification state {k + 1}/{n_init}: {date}')
    date_dt = datetime.strptime(date, '%Y-%m-%dT%H')
    aifs_ic.fetch_era5_grib(model.runner, date_dt, cache_dir, lagged=False)
    for dv in dt_verifs:
        shifted = utils.add_hours(date, dv)
        logger.info(f'  + z500-diagnostic truth: {shifted} (cycle date + dt_verif {dv}h)')
        aifs_ic.fetch_era5_grib(model.runner, datetime.strptime(shifted, '%Y-%m-%dT%H'), cache_dir, lagged=False)
    date = utils.add_hours(date, exp_config['dt_init'])

logger.info('prefetch complete.')
