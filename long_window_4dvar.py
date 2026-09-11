"""
this version is a prototype 4dvar solver.

compute optimal initial conditions for AIFS-single-2.0, optionally using expanding
optimization time windows

Originator: Greg Hakim
            University of Washington
            January 2025
            rev. December 2025
            4dvar test code: April 2026

Jeff Whitaker added mods for real surface pressure observations (June-Aug 2026)

Ported from NeuralGCM/JAX to AIFS-single-2.0/PyTorch, August 2026 -- see
long_window_4dvar_utils.py's module docstring and aifs_model.py for the
structural differences this port is built around.

Control is latent-only (see long_window_4dvar_utils.py's docstring): the
increment lives in AIFS's hidden-mesh space, and the analysis is always
valid at `window_start + dt_verif`, not `window_start`. Observations may be
sampled finer than the 6h model step via a per-window `dt_obs` key (the
decoded model state is linearly interpolated in time to each obs time).

time scheme:
 init_time: starting time for the forecast (lagging real time) (string)
    n_init: number of initializations to cycle through in an experiment, separated by dt_init
    dt_init: spacing between the init times in hours (integer)
 verif_times: list of string times for computing the loss (list of strings)
    n_verif: number of verification times separated by dt_verif
    dt_verif: spacing between verif times in hours (integer) -- must be a
              multiple of AIFS's fixed 6h timestep.
    dt_obs: observation sampling interval in hours (default dt_verif; must
              divide both dt_verif and 6).

 other parameters:
 max_epoch: max number of iterations during gradient descent
 vvars: variables used to define the loss function
 learn_rate: for AdamW
 weight_decay: for AdamW
 ...
"""

import copy
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import torch

import long_window_4dvar_utils as utils

# initialize logger
logger = utils.get_logger()

# load configuration for the experiment. Config path is an optional argv[1]
# (default 'config.yml'), so a test run can point at a different config
# without touching the `config.yml` symlink an in-flight job may depend on.
config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yml'
logger.info('config file: ' + config_path)
exp_config, windows = utils.load_config(config_path)

# create the experiment directory if it doesn't exist; copy the configution file there
exp_dir = exp_config['path_output']
os.makedirs(exp_dir, exist_ok=True)
shutil.copy(config_path, exp_dir + 'config.yml.' + datetime.now().strftime("%Y%m%d_%H%M%S"))

logger.info('reading model checkpoint ' + str(exp_config['model_name']))
model = utils.get_model(exp_config)
logger.info('model time step = ' + str(model.timestep))
grid_interp = utils.get_grid_interpolator(model)

plevs = model.pressure_levels('z')
nlev500 = int(np.argwhere(plevs == 500)[0].item())
nlev700 = int(np.argwhere(plevs == 700)[0].item())  # used in ps forward operator (if ps_operator='ps')
exp_config['nlev500'] = nlev500
exp_config['nlev700'] = nlev700

lats = model.lats  # (n_points,) -- AIFS's grid is unstructured, not a lat/lon rectangle
coslats = np.cos(np.radians(lats))
mask_nh = np.where(lats > 20., coslats, np.nan)
mask_sh = np.where(lats < -20., coslats, np.nan)
mask_tr = np.where(np.logical_and(lats >= -20, lats <= 20), coslats, np.nan)


def printz500err(label, f_decoded, verif_ic, date):
    # print z500 rms err
    z500err = (
        f_decoded['geopotential'][nlev500, :].detach().cpu().numpy()
        - verif_ic['geopotential'][nlev500, :].detach().cpu().numpy()
    ) / utils.GRAV
    z500rmserrnh = np.sqrt(np.nansum(mask_nh * z500err ** 2) / np.nansum(mask_nh))
    z500rmserrsh = np.sqrt(np.nansum(mask_sh * z500err ** 2) / np.nansum(mask_sh))
    z500rmserrtr = np.sqrt(np.nansum(mask_tr * z500err ** 2) / np.nansum(mask_tr))
    z500rmserrgl = np.sqrt(np.sum(coslats * z500err ** 2) / np.sum(coslats))
    print("%s %s %6.2f %6.2f %6.2f %6.2f" % (label, date, z500rmserrnh, z500rmserrtr, z500rmserrsh, z500rmserrgl))


def steps_for_hours(hours):
    return int(round(hours / (model.timestep.total_seconds() / 3600)))


date = copy.copy(exp_config['sdate'])
analysis_state = None  # make the linter happy
for k in range(exp_config['n_init']):
    logger.info(' ------------------------------')
    logger.info(' ------- ' + date + ' --------')
    logger.info(' ------------------------------')
    exp_config['date'] = date
    exp_config['restart_window'] = exp_config['restart']
    for window in windows.keys():
        # update experiment dictionary
        exp_config = utils.get_window(exp_config, window, windows)
        # log an update about the experiment
        utils.log_window(exp_config, logger)
        # generate initial condition (autonomous cycling or era5 disk/cloud)
        if k == 0 and window == list(windows.keys())[0]:
            # first time, first window, read from a file based on config
            input_encoded, exp_config = utils.get_input(exp_config, model, logger)
            # if we just read an optimal IC file (restart), advance the state
            if exp_config['restart']:
                logger.info('making prior from a forecast initialized with the RESTART optimal...')
                # The saved analysis is dated +dt_verif past its filename date
                # (latent analysis is only valid at +dt_verif -- see
                # compute_optimal). Advance only the REMAINING hours to the
                # next cycle's date, not the full dt_init, or the latent shift
                # is double-counted -- the same correction the k != 0 cycling
                # branch below applies.
                shift_hours = (input_encoded.date - datetime.strptime(date, '%Y-%m-%dT%H')).total_seconds() / 3600.0
                remaining_hours = exp_config['dt_init'] - shift_hours
                if remaining_hours < -1e-6:
                    raise ValueError(
                        f"dt_init ({exp_config['dt_init']}h) must be >= the restart analysis's shift past its "
                        f"filename date ({shift_hours}h, the latent +dt_verif shift) to cycle"
                    )
                input_encoded = model.advance(input_encoded, steps=steps_for_hours(max(remaining_hours, 0.0)))
                # advance init time to reflect the forecast
                date = utils.add_hours(date, exp_config['dt_init'])
                exp_config['date'] = date
                exp_config['sdate'] = date
                exp_config['restart'] = False
                logger.info('current date/time is now: ' + date)
            is_new_analysis_time = True
        elif exp_config['cycle'] and k != 0 and window == list(windows.keys())[0]:
            # use existing IC, only after the first DA time and first window.
            # analysis_state.date is ahead of input_encoded.date by dt_verif
            # (latent analysis is only ever valid at +dt_verif, not the
            # window-start date -- see compute_optimal) -- advance only the
            # REMAINING hours needed to reach the next cycle's date, not the
            # full dt_init (which would double-count the shift already baked
            # into analysis_state).
            logger.info('making prior from a forecast initialized with the previous-time optimal...')
            shift_hours = (analysis_state.date - input_encoded.date).total_seconds() / 3600.0
            remaining_hours = exp_config['dt_init'] - shift_hours
            # remaining_hours == 0 is valid (dt_init == the analysis's own
            # shift, e.g. dt_init == dt_verif -- analysis_state IS already
            # the next cycle's background, steps_for_hours(0) == 0, a no-op
            # advance) -- only a genuinely negative remainder (dt_init less
            # than the shift already baked into analysis_state) is an error.
            # Tolerance guards against float round-off landing at e.g. -1e-9
            # instead of exactly 0.
            if remaining_hours < -1e-6:
                raise ValueError(
                    f"dt_init ({exp_config['dt_init']}h) must be >= the analysis's shift past window start "
                    f"({shift_hours}h, the latent +dt_verif shift) to cycle"
                )
            input_encoded = model.advance(analysis_state, steps=steps_for_hours(max(remaining_hours, 0.0)))
            is_new_analysis_time = True
        else:
            # window-to-window within one cycle -- analysis_state is already
            # a fully valid dated state, no forecast-forward step needed, and
            # this isn't a new analysis time (reset_skt_ocean below doesn't
            # apply here -- there's no new ERA5 truth to reset from until the
            # next DA cycle).
            logger.info('using optimal from previous window...')
            input_encoded = analysis_state
            is_new_analysis_time = False
        # get verification at start of window (including ERA5 orography)
        verif_ic = utils.get_verif(exp_config, model, logger)
        logger.info('get verif_ic...')
        if is_new_analysis_time and exp_config.get('reset_skt_ocean', False):
            logger.info('resetting skt over ocean from ERA5 (reset_skt_ocean)...')
            input_encoded = utils.reset_skt_over_ocean(
                model, input_encoded, verif_ic, lsm_threshold=exp_config.get('skt_ocean_lsm_threshold', 0.5)
            )
        # get real ps observations
        logger.info('reading observations...')
        psobs_traj = utils.get_psobs(exp_config, model, logger, grid_interp)

        # call the optimization routine
        analysis_state, lsave = utils.compute_optimal(exp_config, model, input_encoded, verif_ic, psobs_traj, grid_interp, logger)
        # save initial and final trajectory in model and observation space.
        utils.save_trajectory_diagnostics(
            exp_config, model, input_encoded, analysis_state, verif_ic, psobs_traj,
            exp_config['save_levs'], grid_interp, logger,
        )
        # print z500 error before and after optimization. The latent
        # analysis_state is always valid at window_start + dt_verif, not
        # window_start -- score the background *advanced to that same valid
        # time* and the analysis against ERA5 truth fetched there (an exact,
        # same-valid-time before/after comparison). The extra bg(t0) line is
        # the prior's error at window start, kept for across-cycle tracking.
        vdate = analysis_state.date.strftime('%Y-%m-%dT%H')
        shift_hours = (analysis_state.date - input_encoded.date).total_seconds() / 3600.0
        verif_shifted = utils.get_verif(exp_config, model, logger, date_override=vdate)
        with torch.no_grad():
            bg_shifted = model.advance(
                input_encoded, steps=steps_for_hours(shift_hours), use_checkpoint=False
            )
        printz500err('z500err bg(t0)', model.decode_state(input_encoded), verif_ic, date)
        printz500err('z500err before', model.decode_state(bg_shifted), verif_shifted, vdate)
        printz500err('z500err after', model.decode_state(analysis_state), verif_shifted, vdate)
        # save the inputs and loss info
        utils.save_inputs_pkl(exp_config, input_encoded, analysis_state, lsave)
        utils.save_inputs_nc(exp_config, model, input_encoded, analysis_state, logger)
        # longer forecast, if desired.
        if 'dt_forecast' in exp_config and 'n_forecast' in exp_config:
            utils.make_forecasts(exp_config, model, input_encoded, analysis_state, exp_config['save_levs'], logger)

    # advance init time
    date = utils.add_hours(date, exp_config['dt_init'])

logger.info('-------------------')
logger.info('...JOB COMPLETED...')
logger.info('-------------------')
