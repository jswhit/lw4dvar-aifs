"""
Irregular-grid geometry helpers for the AIFS 4D-Var port.

AIFS's N320 octahedral reduced-Gaussian grid is a flat list of ~542080
(lat, lon) points, not a regular lat/lon grid -- `interp2d`'s bilinear
regular-grid assumption (long_window_4dvar_utils.py) doesn't apply. This
module replaces it with a k-nearest-neighbor inverse-distance-weighted
interpolator built once (per set of observation locations) from a k-d tree
over the model grid, so per-epoch cost inside the loss function is just a
fixed-index gather + fixed-weight dot product (trivially differentiable,
since the indices/weights don't depend on the model state).
"""

import numpy as np
import torch
from scipy.spatial import cKDTree


def _lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Unit-sphere Cartesian coordinates -- avoids the antimeridian/pole
    discontinuities of raw (lon, lat) nearest-neighbor search."""
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


class GridInterpolator:
    """k-d tree over the AIFS model grid, for interpolating model fields to
    arbitrary (lon, lat) observation locations."""

    def __init__(self, model_lons: np.ndarray, model_lats: np.ndarray):
        self.model_lons = np.asarray(model_lons, dtype=np.float64)
        self.model_lats = np.asarray(model_lats, dtype=np.float64)
        self.tree = cKDTree(_lonlat_to_xyz(self.model_lons, self.model_lats))
        self.coslat = np.cos(np.radians(self.model_lats)).astype(np.float32)

    def weights(self, ob_lon: np.ndarray, ob_lat: np.ndarray, k: int = 4, device=None):
        """Precompute k-NN indices + inverse-distance weights for a batch of
        observation locations. Returns (idx, wts) torch tensors of shape
        (n_obs, k); rows for out-of-range/padding obs are harmless (weights
        still sum to 1, values simply unused downstream since `used`/QC masks
        those rows out of the loss).

        Call once per window (like get_psobs), not per epoch.
        """
        pts = _lonlat_to_xyz(np.asarray(ob_lon, dtype=np.float64), np.asarray(ob_lat, dtype=np.float64))
        dist, idx = self.tree.query(pts, k=k)
        if k == 1:
            dist = dist[:, np.newaxis]
            idx = idx[:, np.newaxis]
        # chordal distance on a unit sphere -> inverse-distance weights
        # (small epsilon guards an observation that lands exactly on a grid point)
        w = 1.0 / (dist + 1e-6)
        w = w / w.sum(axis=-1, keepdims=True)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
        wts_t = torch.as_tensor(w, dtype=torch.float32, device=device)
        return idx_t, wts_t

    def interp(self, idx: torch.Tensor, wts: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        """Apply precomputed (idx, wts) to a field.

        field : (..., n_points) tensor (e.g. (n_levels, n_points) or (n_points,))
        idx, wts : (n_obs, k)
        returns : (..., n_obs)
        """
        gathered = field[..., idx]  # (..., n_obs, k)
        return (gathered * wts).sum(dim=-1)
