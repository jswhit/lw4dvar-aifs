"""
Historical ERA5 initial-condition fetching for the AIFS 4D-Var port, factored
out of the validated test_forecast/{fetch_ic_cds,patch_missing_wave_fields,
duplicate_constant_fields}.py scripts and generalized to an arbitrary date.

ECMWF Open Data (the default `input: opendata` in aifs-single-2.0/inference.yaml)
only retains the last ~12 forecast cycles -- it cannot initialize a run from an
arbitrary past date. This fetches from CDS instead (dataset
`reanalysis-era5-complete`, `class: ea`), which needs internet access -- call
this from a login node, never from the (offline) H100 compute nodes.

Two real ERA5 data gaps, discovered building test_forecast/, are handled here
too:
  - `swh`'s checkpoint-declared paramId (3100) is a post-2026 operational wave
    table code ERA5 doesn't have; resolved by shortname within the wave stream
    instead (same trick the checkpoint already uses for `mwd` -- see the
    `typed_variables` block in aifs_inference.yaml).
  - `h1012, h1214, h1417, h1721, h2125, h2530` (wave-period-band significant
    heights) are genuinely absent from ERA5 (introduced for AIFS v2's
    IFS-50r1-esuite fine-tuning, after ERA5 was fixed). Cold-started at zero
    (a standard, defensible wave-model cold-start) -- see
    `patch_missing_wave_fields` below.

The GRIB round-trip test_forecast/ used (fetch -> patch -> write -> re-read via
GribFileInput) is kept as-is here rather than reimplemented in-process: this
runs once per DA cycle, not once per optimizer epoch, so the extra I/O is
negligible next to the optimization cost, and reusing the exact validated path
is worth more than the (here, irrelevant) speed of avoiding it.
"""

import datetime
import logging
import os

import eccodes
import numpy as np
from earthkit.data.utils.dates import to_datetime

from anemoi.inference.inputs import create_input
from anemoi.inference.inputs.gribfile import GribFileInput

LOG = logging.getLogger(__name__)

CDS_DATASET = "reanalysis-era5-complete"

# checkpoint.mars_requests(..., use_grib_paramid=True) paramIds for the wave
# variables genuinely absent from ERA5 (see module docstring).
_MISSING_WAVE_PARAMIDS = {
    "h1012": 140114,
    "h1214": 140115,
    "h1417": 140116,
    "h1721": 140117,
    "h2125": 140118,
    "h2530": 140119,
}
_CONST_FIELDS = {
    ("lsm", "surface"),
    ("sdor", "surface"),
    ("slor", "surface"),
    ("z", "surface"),
    ("wmb", "meanSea"),
}


def _build_input(runner, config, purpose):
    """Mirrors Runner.create_prognostics_input()/create_constant_coupled_forcings_input(),
    but with an explicit `input:`-style config instead of `runner.config.input`,
    so the same runner can be pointed at CDS for fetching and at a local GRIB
    file for reading back, independent of whatever `aifs_inference.yaml` says.
    """
    if purpose == "prognostics":
        variables = runner.variables.retrieved_prognostic_variables()
    elif purpose == "constant_forcings":
        variables = runner.variables.retrieved_constant_forcings_variables()
    else:
        raise ValueError(purpose)
    if not variables:
        config = "empty"
    return create_input(runner, config, variables=variables, purpose=purpose)


def _patch_missing_wave_fields(grib_path):
    """Append zero-valued h1012..h2530 fields (cloned from an existing `mwp`
    field's grid definition) for every lagged date present in the file."""
    templates = {}
    with open(grib_path, "rb") as f:
        while True:
            msg = eccodes.codes_grib_new_from_file(f)
            if msg is None:
                break
            if eccodes.codes_get(msg, "shortName") == "mwp":
                key = (eccodes.codes_get(msg, "dataDate"), eccodes.codes_get(msg, "dataTime"))
                templates[key] = eccodes.codes_clone(msg)
            eccodes.codes_release(msg)

    with open(grib_path, "ab") as out:
        for key, template in templates.items():
            n = eccodes.codes_get_size(template, "values")
            for name, paramid in _MISSING_WAVE_PARAMIDS.items():
                h = eccodes.codes_clone(template)
                eccodes.codes_set(h, "paramId", paramid)
                eccodes.codes_set_values(h, [0.0] * n)
                assert eccodes.codes_get(h, "shortName") == name
                eccodes.codes_write(h, out)
                eccodes.codes_release(h)
            eccodes.codes_release(template)


def _duplicate_constant_fields(grib_path):
    """`input: grib` expects the constant/orography fields present at BOTH
    lagged dates (GribFileInput has no notion of "this field never changes"),
    even though fetch only retrieves them once (for the current date). Clone
    those 5 fields' messages onto the other lagged date -- lossless, since the
    values genuinely don't vary in time."""
    dates = set()
    const_date = None
    with open(grib_path, "rb") as f:
        while True:
            msg = eccodes.codes_grib_new_from_file(f)
            if msg is None:
                break
            key = (eccodes.codes_get(msg, "dataDate"), eccodes.codes_get(msg, "dataTime"))
            dates.add(key)
            if (eccodes.codes_get(msg, "shortName"), eccodes.codes_get(msg, "typeOfLevel")) in _CONST_FIELDS:
                const_date = key
            eccodes.codes_release(msg)

    if len(dates) < 2 or const_date is None:
        return  # single-date fetch (e.g. verification state) -- nothing to duplicate
    (target_date,) = dates - {const_date}

    templates = {}
    with open(grib_path, "rb") as f:
        while True:
            msg = eccodes.codes_grib_new_from_file(f)
            if msg is None:
                break
            key = (eccodes.codes_get(msg, "shortName"), eccodes.codes_get(msg, "typeOfLevel"))
            date = (eccodes.codes_get(msg, "dataDate"), eccodes.codes_get(msg, "dataTime"))
            if key in _CONST_FIELDS and date == const_date and key not in templates:
                templates[key] = eccodes.codes_clone(msg)
            eccodes.codes_release(msg)

    with open(grib_path, "ab") as out:
        for key, template in templates.items():
            h = eccodes.codes_clone(template)
            eccodes.codes_set(h, "dataDate", target_date[0])
            eccodes.codes_set(h, "dataTime", target_date[1])
            eccodes.codes_write(h, out)
            eccodes.codes_release(h)
            eccodes.codes_release(template)


def fetch_era5_grib(runner, date, cache_dir, lagged=True):
    """Fetch (prognostic + constant-forcing) ERA5 fields for `date` from CDS,
    patch in the zero-filled wave-band fields, and write to a cached local
    GRIB file. Needs internet access (call from a login node).

    Parameters
    ----------
    date : datetime.datetime
    lagged : bool
        True to fetch both of the checkpoint's lagged input dates (for
        building a prognostic initial state); False to fetch just `date`
        (for a single-time verification state).

    Returns
    -------
    str : path to the cached GRIB file.
    """
    date = to_datetime(date)
    os.makedirs(cache_dir, exist_ok=True)
    tag = date.strftime("%Y%m%dT%H") + ("_lagged" if lagged else "")
    grib_path = os.path.join(cache_dir, f"era5_{tag}.grib")
    if os.path.exists(grib_path):
        LOG.info("Reusing cached ERA5 GRIB: %s", grib_path)
        return grib_path

    dates = [date + h for h in runner.checkpoint.lagged] if lagged else [date]

    prog = _build_input(runner, {"cds": {"dataset": CDS_DATASET, "class": "ea"}}, "prognostics")
    fields = prog.retrieve(prog.variables, dates)

    const = _build_input(runner, {"cds": {"dataset": CDS_DATASET, "class": "ea"}}, "constant_forcings")
    if hasattr(const, "retrieve"):
        fields = fields + const.retrieve(const.variables, [date])

    tmp_path = grib_path + ".tmp"
    fields.save(tmp_path)
    os.replace(tmp_path, grib_path)

    _patch_missing_wave_fields(grib_path)
    if lagged:
        _duplicate_constant_fields(grib_path)

    LOG.info("Saved ERA5 GRIB: %s", grib_path)
    return grib_path


def build_input_state(runner, date, cache_dir, lagged=True):
    """Fetch (or reuse cached) ERA5 GRIB for `date` and build the combined
    anemoi `State` dict (prognostic + constant forcings) ready for
    `AIFSModel.prepare_initial_state`.
    """
    grib_path = fetch_era5_grib(runner, date, cache_dir, lagged=lagged)

    prog = GribFileInput(runner, path=grib_path, variables=runner.variables.retrieved_prognostic_variables())
    prognostic_state = prog.create_input_state(date=date)

    const_vars = runner.variables.retrieved_constant_forcings_variables()
    if const_vars:
        const = GribFileInput(runner, path=grib_path, variables=const_vars)
        constants_state = const.create_input_state(date=date)
        input_state = runner._combine_states(prognostic_state, constants_state)
    else:
        input_state = prognostic_state

    return input_state


def read_single_date_fields(runner, date, cache_dir):
    """Fetch (or reuse cached) a *single-date* (non-lagged) ERA5 GRIB for
    `date` and return a plain {variable_name: (n_points,) array} dict.

    Unlike build_input_state(), this does NOT go through
    GribFileInput.create_input_state(): that method unconditionally requests
    both of the checkpoint's lagged dates (`[date + h for h in
    checkpoint.lagged]`) regardless of what's actually in the file, so it
    only works for a proper two-time-level prognostic state. get_verif() just
    needs a snapshot of every fetched field at the one date requested.
    """
    import earthkit.data as ekd

    grib_path = fetch_era5_grib(runner, date, cache_dir, lagged=False)
    fieldlist = ekd.from_source("file", grib_path)
    fields = {}
    for f in fieldlist:
        name = f.metadata("shortName")
        is_pressure_level = f.metadata("typeOfLevel") == "isobaricInhPa"
        key = f"{name}_{int(f.metadata('level'))}" if is_pressure_level else name
        fields[key] = f.to_numpy(flatten=True)
    return fields
