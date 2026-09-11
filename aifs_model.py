"""
Differentiable AIFS-single-2.0 model wrapper for the 4D-Var solver.

AIFS's runtime state is a packed tensor `(batch=1, multi_step=2, n_points, n_vars)`
-- the two lagged time levels *are* the model's state, there is no separate
encode step the way NeuralGCM has. `AIFSState.state` *is* that tensor, and
`AIFSState` bundles it with the date it's actually valid at -- `compute_optimal`
returns a full `AIFSState` (not a bare tensor) precisely because a
`control_space: 'latent'` analysis is valid at a *different* date than its
background (see `advance`'s `latent_increment` param below and
long_window_4dvar_utils.compute_optimal_latent's docstring), so bare tensors
can't be passed around and silently re-dated onto whatever background
happens to be at hand the way they could when every analysis shared its
background's date.

`AnemoiModelEncProcDec.predict_step` (anemoi/models/models/encoder_processor_decoder.py)
hardcodes `with torch.no_grad():` around pre_processors -> forward -> post_processors,
so it can never produce gradients. `_predict_step_with_grad` below replicates that same
sequence without the no_grad, which is otherwise a plain (differentiable) sequence of
tensor ops -- verified against real gradient descent in test_forecast/smoke_test_optimize.py
(AdamW loss decreased smoothly and monotonically over 20 steps using these gradients).

Everything else (packing the initial state, rolling the two-time-level window forward,
refreshing dynamic/boundary forcings each step) reuses the real `anemoi.inference.Runner`
machinery directly (`Runner.prepare_input_tensor`, `.copy_prognostic_fields_to_input_tensor`,
`.add_dynamic_forcings_to_input_tensor`, `.add_boundary_forcings_to_input_tensor`) rather
than reimplementing it -- those are plain, already-correct, already-validated torch ops
with no grad-blocking of their own; only `Runner.forecast()`'s outer
`torch.inference_mode()` context and its call to `predict_step` are bypassed.
"""

import copy
import datetime

import numpy as np
import torch

from anemoi.inference.config.run import RunConfiguration
from anemoi.inference.runners import create_runner
from anemoi.models.distributed.shapes import get_shard_shapes


class AIFSState:
    """Thin wrapper whose `.state` attribute *is* the packed
    `(1, multi_step, n_points, n_vars)` tensor. Mirrors NeuralGCM's
    `input_encoded` object closely enough that `copy.copy(input_encoded)` +
    reassigning `.state` (used throughout long_window_4dvar.py/_utils.py)
    keeps working unchanged.
    """

    __slots__ = ("state", "date")

    def __init__(self, state: torch.Tensor, date: datetime.datetime):
        self.state = state
        self.date = date

    def __copy__(self) -> "AIFSState":
        return AIFSState(self.state, self.date)


class AIFSModel:
    """Wraps a loaded AIFS-single-2.0 checkpoint for differentiable rollouts."""

    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda"):
        """
        Parameters
        ----------
        checkpoint_path : str
            Path to the .ckpt file -- overrides whatever `checkpoint:` is set
            to in `config_path`, so config.yml's `path_model`/`model_name`
            are what actually selects the checkpoint.
        config_path : str
            Path to an anemoi-inference run config (yaml) providing
            pre_processors/post_processors/typed_variables/patch_metadata --
            the runner never actually runs inference through it (`input:`/
            `output:`/`date:`/`lead_time:` in that file are unused
            placeholders), only `runner.model`/`runner.checkpoint` and the
            lower-level tensor-prep methods are used.
        """
        # `device` isn't in aifs_inference.yaml (it has no fixed GPU/CPU
        # assumption baked in) -- pass it as an explicit override so callers
        # that only need checkpoint metadata (aifs_ic.py's prefetch, run from
        # a login node with no GPU) can request device='cpu' without editing
        # the yaml, while the H100 driver gets 'cuda'.
        config = RunConfiguration.load(config_path, [f"device={device}", f"checkpoint={checkpoint_path}"])
        self.runner = create_runner(config)
        self.runner.model.eval()
        # We only ever want gradients w.r.t. OUR control-vector increment,
        # never w.r.t. AIFS's pretrained weights -- torch.load doesn't set
        # requires_grad=False on its own. Without this, any state built via
        # model.advance() (e.g. get_input()'s initial forecast, built once
        # *before* the epoch loop) silently carries a live, un-freed autograd
        # graph back through that forecast's model-weight-dependent
        # computation; reusing that state as a fixed background in a second
        # loss.backward() call then fails with "Trying to backward through
        # the graph a second time" (the upstream graph was already freed by
        # the first backward() that touched it). Freezing the weights here
        # means every model.advance() output is automatically leaf-like
        # (requires_grad=False) unless it flows through OUR increment, which
        # is the correct/intended behavior throughout.
        for p in self.runner.model.parameters():
            p.requires_grad_(False)

        self.checkpoint = self.runner.checkpoint
        self.device = self.runner.device
        self.interface = self.runner.model  # AnemoiModelInterface
        self.multi_step = self.interface.multi_step
        self.timestep: datetime.timedelta = self.checkpoint.timestep

        self.var_to_idx = self.checkpoint.variable_to_input_tensor_index
        self.idx_to_var = self.checkpoint.output_tensor_index_to_variable
        self.lats = np.asarray(self.checkpoint.latitudes)
        self.lons = np.asarray(self.checkpoint.longitudes)
        self.n_points = self.lats.size

        self.pmask_in = torch.as_tensor(
            self.checkpoint.prognostic_input_mask, device=self.device, dtype=torch.long
        )
        self.pmask_out = torch.as_tensor(
            self.checkpoint.prognostic_output_mask, device=self.device, dtype=torch.long
        )
        # Which input-tensor columns are constant in time (orography, land-sea
        # mask, bathymetry, ...): these are never refreshed after the initial
        # state is built -- `.roll()` alone keeps them correct since both time
        # levels already carry the same value.
        self._reset = np.zeros(self.checkpoint.number_of_input_features, dtype=bool)
        for name, i in self.var_to_idx.items():
            if self.checkpoint.typed_variables[name].is_constant_in_time:
                self._reset[i] = True

        # Group each family of pressure-level variables (e.g. z_10, z_50, ...,
        # z_1000) by base name, sorted by level, for decode_state(). The
        # single-level constant surface 'z' (geopotential-at-surface) shares
        # its base letter with the pressure-level 'z' family -- renamed here
        # to avoid the two colliding on the same decode_state() output key.
        self._levels_by_base: dict[str, list[tuple[int, int]]] = {}
        self._single_by_base: dict[str, int] = {}
        for name, i in self.var_to_idx.items():
            base, _, lev = name.rpartition("_")
            if base and lev.isdigit():
                self._levels_by_base.setdefault(base, []).append((int(lev), i))
            elif name == "z":
                self._single_by_base["geopotential_at_surface"] = i
            else:
                self._single_by_base[name] = i
        for base in self._levels_by_base:
            self._levels_by_base[base].sort(key=lambda t: t[0], reverse=True)  # high pressure (surface) first

    # ------------------------------------------------------------------
    # State construction / decoding
    # ------------------------------------------------------------------

    def prepare_initial_state(self, input_state: dict, date: datetime.datetime) -> AIFSState:
        """Build the packed `(1, multi_step, n_points, n_vars)` tensor from a
        combined anemoi `State` dict (see aifs_ic.py), using the real
        `Runner.prepare_input_tensor`.

        `runner.create_constant_forcings_inputs()` (plural) has two
        sub-categories: "constant+forcing" (loaded, e.g. z/lsm/sdor/slor for
        this checkpoint) tries to build an input source from
        `aifs_inference.yaml`'s placeholder `input:` (never meant to be
        instantiated -- see that file's docstring) to re-fetch them, which
        crashes -- skipped here, since aifs_ic.py's fetch already put those
        fields straight into `input_state["fields"]`. "computed+constant"
        (cos_latitude, sin_latitude, cos_longitude, sin_longitude for this
        checkpoint) resolves to `ComputedForcings` (pure function of lat/lon,
        no `input:` dependency) and is genuinely needed -- these aren't in
        the fetched GRIB (nothing to fetch, they're analytic) -- so that half
        is called directly rather than through the crashing plural method.
        `create_dynamic_forcings_inputs()` is safe to call for real as-is:
        AIFS's dynamic forcings (cos_julian_day, insolation, ...) are all in
        the checkpoint's "computed" category too.
        """
        runner = self.runner
        computed_vars, computed_mask = self.checkpoint.select_variables_and_masks(include=["computed+constant"])
        runner.constant_forcings_inputs = (
            runner.create_constant_computed_forcings(computed_vars, computed_mask) if len(computed_mask) else []
        )
        runner.dynamic_forcings_inputs = runner.create_dynamic_forcings_inputs(input_state)
        runner.boundary_forcings_inputs = []  # no boundary/cutout region for a global model

        tensor_np = runner.prepare_input_tensor(input_state)  # (multi_step, n_vars, n_points)
        tensor_np = np.swapaxes(tensor_np, -2, -1)[np.newaxis, ...]  # (1, multi_step, n_points, n_vars)
        state = torch.from_numpy(np.ascontiguousarray(tensor_np, dtype=np.float32)).to(self.device)
        return AIFSState(state, date)

    def prime_runner_from_packed_state(self, aifs_state: AIFSState) -> AIFSState:
        """Re-run the anemoi `Runner` bookkeeping that `prepare_initial_state()`
        builds as a side effect (`runner._input_tensor_by_name`,
        `runner._input_kinds`, and the `*_forcings_inputs` lists), starting
        from an already-packed `(1, multi_step, n_points, n_vars)` state
        instead of a freshly-fetched anemoi `State` dict.

        Needed on the restart path: `get_input()` there loads a pickled
        `AIFSState` straight off disk and never calls
        `prepare_initial_state()`, so the first `advance()` meets a runner
        whose `_input_tensor_by_name` is still the empty `[]` from
        `Runner.__init__` and dies with `IndexError` inside
        `copy_prognostic_fields_to_input_tensor`.

        The packed tensor already carries every input feature -- prognostic,
        constant, and forcing columns -- at both lagged time levels, so a
        full anemoi `State` dict is reconstructed from it directly (no GRIB,
        no network). `prepare_input_tensor()` recomputes the analytic forcing
        columns from the state's date + lat/lon (exactly as every
        `advance()` step already does), so the returned `AIFSState` is
        functionally identical to the one passed in -- callers may use
        either; the restart branch uses the returned one for clarity.
        """
        packed = aifs_state.state[0].detach().cpu().numpy()  # (multi_step, n_points, n_vars)
        fields = {name: packed[:, :, idx] for name, idx in self.var_to_idx.items()}
        input_state = {
            "date": aifs_state.date,
            "latitudes": self.lats,
            "longitudes": self.lons,
            "fields": fields,
        }
        return self.prepare_initial_state(input_state, aifs_state.date)

    def decode_state(self, aifs_state: AIFSState, time_index: int = -1) -> dict[str, torch.Tensor]:
        """Slice one time level of the packed tensor into a dict of
        {base_variable_name: tensor}, with pressure-level variables stacked
        into shape (n_levels, n_points) (highest pressure/surface first) and
        single-level variables shape (n_points,). Replaces NeuralGCM's
        `model.decode()` for everything the loss function/diagnostics need
        (all inputs needed downstream -- geopotential, temperature,
        specific_humidity, surface_pressure, geopotential_at_surface -- are
        prognostic/constant inputs, so no separate "model output" decoding is
        needed).
        """
        tensor = aifs_state.state[:, time_index, :, :]  # (1, n_points, n_vars)
        tensor = tensor[0]  # (n_points, n_vars)
        out: dict[str, torch.Tensor] = {}
        for base, levels in self._levels_by_base.items():
            cols = [i for _, i in levels]
            out[base] = tensor[:, cols].transpose(0, 1)  # (n_levels, n_points)
        for name, i in self._single_by_base.items():
            out[name] = tensor[:, i]  # (n_points,)
        # NeuralGCM-style aliases used by the ported loss/QC code.
        if "z" in out:
            out["geopotential"] = out["z"]
        if "t" in out:
            out["temperature"] = out["t"]
        if "q" in out:
            out["specific_humidity"] = out["q"]
        if "sp" in self._single_by_base:
            out["surface_pressure"] = out["sp"]
        return out

    def pressure_levels(self, base: str = "z") -> np.ndarray:
        return np.array([lev for lev, _ in self._levels_by_base[base]], dtype=np.float32)

    @property
    def latent_shape(self) -> tuple[int, int]:
        """(num_hidden_mesh_nodes, num_channels) -- the shape of the
        encoder's hidden-mesh output (`x_latent` in
        `AnemoiModelEncProcDec.forward`), i.e. the shape `control_space:
        latent`'s control variable must have. For this checkpoint: a O96
        reduced-Gaussian hidden mesh (40320 nodes) x 1024 channels, vs. the
        physical control's 2 (multi_step) x 542080 (N320 grid) x 106
        (variables) -- about 2.8x fewer elements, and already a single
        normalized embedding rather than two lagged physical time levels.
        """
        m = self.interface.model
        n_hidden = m.node_attributes.num_nodes[m._graph_name_hidden]
        return (int(n_hidden), int(m.num_channels))

    # ------------------------------------------------------------------
    # Differentiable rollout
    # ------------------------------------------------------------------

    def _predict_step_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Replicates AnemoiModelEncProcDec.predict_step (pre_processors ->
        forward -> post_processors) without its hardcoded `torch.no_grad()`.
        """
        xin = x[:, 0 : self.multi_step, None, ...]  # add ensemble dim
        xin = self.interface.pre_processors(xin, in_place=False)
        with torch.autocast(device_type=self.device.type, dtype=self.runner.autocast):
            y = self.interface.model.forward(xin, model_comm_group=None, grid_shard_shapes=None)
        y = self.interface.post_processors(y.float(), in_place=False)
        return torch.squeeze(y, dim=1)  # (batch, n_points, n_vars)

    def _predict_step_with_grad_latent(self, x: torch.Tensor, latent_increment: torch.Tensor) -> torch.Tensor:
        """Same as `_predict_step_with_grad`, but `AnemoiModelEncProcDec.forward`
        (anemoi-models==0.9.3, anemoi/models/models/encoder_processor_decoder.py)
        is manually unrolled into its encoder -> [+ latent_increment] ->
        processor -> decoder stages, using the model's own submodules/helper
        methods (`_assemble_input`, `node_attributes`, `_run_mapper`,
        `_assemble_output`) rather than reimplementing their logic, so this
        stays correct if anemoi-models' internals change shape/dtype
        handling -- only re-verify if the encoder/processor/decoder wiring in
        `forward()` itself changes.

        AIFS's decoder produces the *next* time level as a residual added
        onto the most recent input level (see `_assemble_output`'s skip
        connection) -- it does not reconstruct the two-level input state
        itself. So this can only inject a perturbation at a forward step,
        not produce a "corrected initial condition" the way the physical
        control variable does; see `compute_loss_4dvar_latent`'s docstring
        in long_window_4dvar_utils.py for how the driver uses this.

        Single-GPU only: `model_comm_group=None` throughout, dropping the
        real `forward()`'s distributed grid/channel-sharding machinery
        (this repo never runs sharded).
        """
        xin = x[:, 0 : self.multi_step, None, ...]  # add ensemble dim
        xin = self.interface.pre_processors(xin, in_place=False)
        m = self.interface.model  # AnemoiModelEncProcDec
        with torch.autocast(device_type=self.device.type, dtype=self.runner.autocast):
            batch_size = xin.shape[0]
            ensemble_size = xin.shape[2]

            x_data_latent, x_skip, shard_shapes_data = m._assemble_input(xin, batch_size)
            x_hidden_latent = m.node_attributes(m._graph_name_hidden, batch_size=batch_size)
            shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, None)

            x_data_latent, x_latent = m._run_mapper(
                m.encoder,
                (x_data_latent, x_hidden_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_data, shard_shapes_hidden),
                model_comm_group=None,
                x_src_is_sharded=False,
                x_dst_is_sharded=False,
                keep_x_dst_sharded=True,
            )

            x_latent = x_latent + latent_increment

            x_latent_proc = m.processor(
                x_latent, batch_size=batch_size, shard_shapes=shard_shapes_hidden, model_comm_group=None
            )
            x_latent_proc = x_latent_proc + x_latent  # residual, matches forward()

            x_out = m._run_mapper(
                m.decoder,
                (x_latent_proc, x_data_latent),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_data),
                model_comm_group=None,
                x_src_is_sharded=True,
                x_dst_is_sharded=False,
                keep_x_dst_sharded=False,
            )

            y = m._assemble_output(x_out, x_skip, batch_size, ensemble_size, xin.dtype)
        y = self.interface.post_processors(y.float(), in_place=False)
        return torch.squeeze(y, dim=1)  # (batch, n_points, n_vars)

    def _advance_one_step(
        self, state: torch.Tensor, date: datetime.datetime, latent_increment: torch.Tensor = None
    ) -> torch.Tensor:
        if latent_increment is None:
            y_pred = self._predict_step_with_grad(state)
        else:
            y_pred = self._predict_step_with_grad_latent(state, latent_increment)

        check = self._reset.copy()
        new_state = self.runner.copy_prognostic_fields_to_input_tensor(state, y_pred, check)
        forcing_state = {"date": date, "latitudes": self.lats, "longitudes": self.lons, "fields": {}}
        new_state = self.runner.add_dynamic_forcings_to_input_tensor(new_state, forcing_state, date, check)
        new_state = self.runner.add_boundary_forcings_to_input_tensor(new_state, forcing_state, date, check)

        if not check.all():
            mapping = {v: k for k, v in self.var_to_idx.items()}
            missing = [mapping[i] for i in range(len(check)) if not check[i]]
            raise ValueError(f"Missing variables in input tensor after step: {sorted(missing)}")
        return new_state

    def advance(
        self,
        aifs_state: AIFSState,
        steps: int = 1,
        use_checkpoint: bool = True,
        latent_increment: torch.Tensor = None,
    ) -> AIFSState:
        """Differentiable multi-step rollout. Each step is individually
        wrapped in torch.utils.checkpoint (the direct analog of the NeuralGCM
        code's `jax.checkpoint(model.advance)` inside `lax.scan`), so
        backprop through a long rollout costs ~O(1) activation memory per
        step instead of O(n_steps).

        latent_increment : torch.Tensor, optional
            A `(n_hidden_nodes, num_channels)` perturbation (see
            `latent_shape`) added to the encoder's hidden-mesh output,
            applied ONLY on the first of `steps` internal single-step
            advances. AIFS re-encodes from the physical state on every
            step (see `_predict_step_with_grad_latent`'s docstring), so
            there is no persistent latent state to keep perturbing across
            steps -- applying it more than once would double-count it.
        """
        state = aifs_state.state
        date = aifs_state.date
        for i in range(steps):
            date = date + self.timestep
            step_latent_increment = latent_increment if i == 0 else None
            if use_checkpoint:
                state = torch.utils.checkpoint.checkpoint(
                    self._advance_one_step, state, date, step_latent_increment, use_reentrant=False
                )
            else:
                state = self._advance_one_step(state, date, step_latent_increment)
        return AIFSState(state, date)
