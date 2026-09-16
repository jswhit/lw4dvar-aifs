"""
Defines the contract the long-window 4D-Var solver core (long_window_4dvar.py
/ long_window_4dvar_utils.py) needs from a forecast-model backend, so a
future backend (a different checkpoint or architecture) can be swapped in
without touching the solver itself. AIFSModel (aifs_model.py) is the only
concrete implementation, and now formally subclasses LatentForecastModel
below -- see CLAUDE.md's "Isolating AIFS-specific code" section for what
that migration did and did not cover (the `model.runner` escape hatch used
by aifs_ic.py's IC-fetching glue is a deliberate, documented exception, not
an oversight -- see InitialConditionProvider's docstring below).

SCOPE: a backend must expose a differentiable hidden-mesh ("latent")
injection point -- the solver's control variable lives there exclusively
(see CLAUDE.md's "control_space: 'latent'" section for why the earlier
physical-space control path was removed). A backend with no accessible
latent representation between its encoder and decoder is out of scope
entirely, not a case this interface degrades gracefully for -- there is
deliberately no physical-space fallback method here.

WHAT THIS DOES NOT COVER: historical IC/verification fetching (ERA5/CDS for
AIFS today, via aifs_ic.py) is deliberately a SEPARATE concern from model
architecture -- see InitialConditionProvider below, sketched but not fully
worked out. aifs_grid.GridInterpolator also needs no change at all: it's
already backend-agnostic (a k-d tree over whatever (lons, lats) it's given),
so it isn't part of this interface -- any backend's `.lats`/`.lons` feed it
directly.
"""

import abc
import datetime
from typing import Optional, Protocol

import numpy as np
import torch


class ModelState(Protocol):
    """Whatever a backend's `advance()`/`decode_state()` pass around as "the
    state" -- opaque to the solver core beyond these two attributes. AIFS's
    `AIFSState.state` happens to be two packed lagged time levels; a
    different backend's `.state` could be shaped completely differently (a
    single time level, a different internal variable ordering, a different
    dtype layout...) as long as that SAME backend's own `advance`/
    `decode_state` can round-trip it -- the solver core never indexes into
    `.state` directly, only through `decode_state()`, so this stays opaque
    by design. `.date` is load-bearing: `compute_optimal`'s analysis is only
    ever valid at `window_start + dt_verif` (the latent-injection mechanics
    below are why), so every consumer keys off `.date`, never an assumption
    that a state shares its background's date.
    """

    state: torch.Tensor
    date: datetime.datetime


class LatentForecastModel(abc.ABC):
    """Abstract forecast-model backend for the 4D-Var solver core.

    Every method/property here is grounded in an actual current call site in
    long_window_4dvar_utils.py / long_window_4dvar.py (see each docstring) --
    this is not a speculative "what might a model need" interface, it's the
    real AIFSModel API with the AIFS-only internals (`.runner`, `.checkpoint`,
    `.var_to_idx`, `.interface`, `._predict_step_with_grad*`, ...) collapsed
    behind a smaller, formal surface.
    """

    # ------------------------------------------------------------------
    # Fixed model properties (read once, cached by the solver at startup)
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def timestep(self) -> datetime.timedelta:
        """Fixed rollout step. dt_verif/dt_init/dt_obs (config) must all be
        multiples of this -- get_window() validates against it."""

    @property
    @abc.abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abc.abstractmethod
    def lats(self) -> np.ndarray:
        """(n_points,) -- feeds aifs_grid.GridInterpolator directly (that
        module is already backend-agnostic) and the saved netCDF's
        `latitude` coordinate."""

    @property
    @abc.abstractmethod
    def lons(self) -> np.ndarray: ...

    @property
    def n_points(self) -> int:
        return self.lats.size

    @abc.abstractmethod
    def pressure_levels(self, base: str) -> np.ndarray:
        """Ascending-or-descending level values for a pressure-level family
        (e.g. base='z' -> the geopotential levels). Callers that need a
        specific ordering (e.g. get_surface_pressure's top-of-atmosphere-
        first `searchsorted` requirement -- see CLAUDE.md's debugging notes
        on the ~230-280 hPa bias bug this caused when AIFS's surface-first
        convention was fed in unflipped) are responsible for checking/
        flipping it themselves; this method makes no ordering promise
        beyond "whatever decode_state's level axis for that family uses."""

    @property
    @abc.abstractmethod
    def latent_shape(self) -> tuple[int, int]:
        """(n_hidden_nodes, n_channels) -- the shape compute_optimal's
        latent-space control increment must have. Required, not optional:
        every backend behind this interface is assumed latent-capable (see
        module docstring) -- there is no sentinel/None value meaning "no
        latent access", that case is simply out of scope."""

    # ------------------------------------------------------------------
    # Variable/family schema -- formalizes what's currently three raw
    # dicts (_levels_by_base / _single_by_base / var_to_idx) that callers
    # in long_window_4dvar_utils.py each re-derive a lookup order for.
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def resolve_columns(self, name: str) -> list[int]:
        """Packed-state column indices for a base variable/family name (e.g.
        'z' -> all pressure-level geopotential columns, 'sp' -> the single
        surface-pressure column).

        MUST check multi-level families before any single-level/raw
        checkpoint-mapping fallback. CLAUDE.md documents a real, shipped bug
        (the `state_scales` lookup, since removed along with the rest of
        physical-space control) where checking the raw checkpoint mapping
        first silently shadowed the 14-level pressure-level 'z' family with
        a single-column raw 'z' entry (surface orography) of the same name
        -- config intended to scale/mask the whole family silently affected
        one column instead, with no error. Formalizing the correct
        precedence into ONE method closes off that entire bug class, rather
        than relying on every call site (control_variables masking,
        loss_variables validation, save_trajectory_diagnostics' netCDF
        dimension building) to each get the order right independently, the
        way long_window_4dvar_utils.py's `_resolve_control_mask` /
        `_resolve_loss_interp_specs` do today.

        Raise KeyError for an unknown name -- control_variables/
        loss_variables validation depends on this to catch a config typo.
        """

    def is_known_variable(self, name: str) -> bool:
        try:
            self.resolve_columns(name)
            return True
        except KeyError:
            return False

    # ------------------------------------------------------------------
    # State construction / decoding
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def decode_state(
        self, state: ModelState, time_index: int = -1, only: "list[str] | None" = None
    ) -> dict[str, torch.Tensor]:
        """Slice one time level into {name: tensor} -- pressure-level
        families as (n_levels, n_points), single-level fields as (n_points,).

        `only`: an optional performance hint -- restrict decoding to just
        these base names, skipping any others. A backend may ignore it (full
        decode is always a valid, if less efficient, implementation); AIFS's
        implements it because its per-step hot loop (compute_loss_4dvar)
        otherwise decodes ~15-20 families it never reads, `max_epoch x
        n_steps` times per window.

        MUST include, under exactly these canonical keys (the solver core's
        forward operator / QC code reads them unconditionally, regardless of
        ps_operator choice): 'geopotential', 'temperature',
        'specific_humidity', 'surface_pressure'. AIFS's decode_state
        produces these as aliases of its own 'z'/'t'/'q'/'sp' base-name
        columns (see aifs_model.AIFSModel.decode_state) -- a backend should
        own ALL of its aliasing internally and return canonical keys
        directly; the solver core should never need a separate alias table
        the way long_window_4dvar_utils._DECODE_ALIASES exists today only
        to reconcile AIFS's two naming styles for time-interpolation
        (_interp_decoded). Folding that into decode_state itself, backend-
        side, is one of the concrete simplifications this interface buys.
        """

    @abc.abstractmethod
    def prepare_initial_state(self, input_state: dict, date: datetime.datetime) -> ModelState:
        """Build a fresh ModelState from a generic (backend-defined) fetched
        input dict + a valid-at date. Backend-specific IC-fetching glue
        (aifs_ic.py today) is expected to call this, not the solver core
        directly -- see InitialConditionProvider below."""

    @abc.abstractmethod
    def prime_from_state(self, state: ModelState) -> ModelState:
        """Rebuild whatever internal bookkeeping `advance()` needs, starting
        from an already-packed state with no fresh fetch (the restart path:
        get_input() loads a pickled ModelState straight off disk, never
        calling prepare_initial_state -- see
        aifs_model.AIFSModel.prime_from_state's docstring for why skipping
        this crashes AIFS's first advance() call after a restart)."""

    # ------------------------------------------------------------------
    # Differentiable rollout
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def advance(
        self,
        state: ModelState,
        steps: int = 1,
        use_checkpoint: bool = True,
        latent_increment: Optional[torch.Tensor] = None,
    ) -> ModelState:
        """Differentiable multi-step rollout, `torch.utils.checkpoint`-wrapped
        per step when use_checkpoint. `latent_increment` (shape ==
        latent_shape, or None) is applied ONLY on the first of `steps`
        internal single-step advances -- see aifs_model.AIFSModel.advance's
        docstring for why (no persistent latent state carries across steps
        for AIFS; a different backend's architecture determines whether that
        same one-shot-injection contract is the only sound one for it too,
        or whether it could support per-step injection -- compute_optimal /
        compute_loss_4dvar as they exist today assume one-shot, so a backend
        supporting more would need the solver core's injection-timing logic
        revisited too, not just this method).
        """


class InitialConditionProvider(abc.ABC):
    """Sketched, not worked out in detail. Per-backend historical IC/
    verification fetching -- deliberately NOT part of LatentForecastModel,
    since it's about *data access* (ERA5 via CDS for AIFS today), not model
    architecture; a backend swap and an IC-source swap are independent
    changes that happen to be coupled today only because get_input()/
    get_verif() in long_window_4dvar_utils.py call aifs_ic.py AND reach past
    it into a couple of AIFSModel-internal escape hatches directly
    (`model.runner`, passed straight into `aifs_ic.build_input_state`/
    `read_single_date_fields`). Formalizing this boundary -- probably as
    `fetch_initial_state`/`fetch_verification` methods here, with the
    `model.runner` reach-through replaced by whatever
    LatentForecastModel-level methods those two aifs_ic functions actually
    need -- is the second half of the isolation job and hasn't been
    scoped yet.
    """

    @abc.abstractmethod
    def fetch_initial_state(self, exp: dict, logger) -> ModelState: ...

    @abc.abstractmethod
    def fetch_verification(
        self, exp: dict, logger, date_override: Optional[str] = None
    ) -> dict[str, torch.Tensor]: ...
