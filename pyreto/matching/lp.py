"""Linear program that splits reference layers into a tower of sub-layers (lp_functions.R)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class _TowerResult:
    """Tower of granular sub-layers produced by the LP engine.

    Returned by both _solve_lp and _calculate_layer_losses.
    Fields ``exp_loss`` and ``frequency`` contain the fully filled-in values
    (i.e. ``exp_loss_new`` / ``frequency_new`` in the R source).
    """

    att: np.ndarray
    limits: np.ndarray
    exp_loss: np.ndarray
    frequency: np.ndarray
    info_available: np.ndarray  # True where original EL data existed
    status: int  # 0=OK, 1=warnings (some refs ignored), 2=error, 4=inconsistent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_tower_points(
    att_layers: np.ndarray,
    limits: np.ndarray,
    thresholds: np.ndarray | None,
) -> np.ndarray:
    """Return sorted unique breakpoints including an infinite sentinel."""
    exit_pts = att_layers + limits
    finite_exit = exit_pts[~np.isinf(exit_pts)]
    pts = np.union1d(att_layers, finite_exit)
    if thresholds is not None and len(thresholds) > 0:
        pts = np.union1d(pts, np.asarray(thresholds, dtype=float))
    pts = np.sort(pts)
    if not np.isinf(pts[-1]):
        pts = np.append(pts, np.inf)
    return pts


# ---------------------------------------------------------------------------
# Public (internal-public) API
# ---------------------------------------------------------------------------


def _solve_lp(
    att_layers: np.ndarray,
    limits: np.ndarray,
    exp_losses: np.ndarray,
    thresholds: np.ndarray | None = None,
    threshold_freqs: np.ndarray | None = None,
) -> _TowerResult:
    """Port of ``solve_lp`` from ``lp_functions.R``.

    For each sub-layer that is covered by at least one input layer, solve an LP
    that maximises its expected loss subject to:

    - equality: each input layer's expected loss equals the sum of its
      sub-layer contributions;
    - monotone RoL: rate-on-line is non-increasing across the tower;
    - frequency bounds: where threshold frequencies are supplied.

    The final sub-layer values are the average across all per-sub-layer LP
    solutions (one solve per info-available sub-layer, maximising that
    sub-layer's EL each time).

    Parameters
    ----------
    att_layers:
        Attachment point of each reference layer.
    limits:
        Cover (limit) of each reference layer.
    exp_losses:
        Expected loss of each reference layer.
    thresholds:
        Optional attachment points at which excess frequencies are known.
    threshold_freqs:
        Excess frequency at each threshold (required when thresholds is given).

    Returns
    -------
    _TowerResult
        status=0 on success, status=1 if any LP solve was infeasible.
    """
    att_layers = np.asarray(att_layers, dtype=float)
    limits = np.asarray(limits, dtype=float)
    exp_losses = np.asarray(exp_losses, dtype=float)

    all_pts = _build_tower_points(att_layers, limits, thresholds)
    tower_att = all_pts[:-1]
    tower_lim = np.diff(all_pts)
    n_att = len(tower_att)
    n_layers = len(att_layers)

    # Equality constraint matrix: A_eq[i, j] = 1 iff sub-layer j is inside
    # reference layer i (i.e. att_layers[i] <= tower_att[j] and
    # att_layers[i]+limits[i] >= all_pts[j+1]).
    a_eq = np.zeros((n_layers, n_att))
    for j in range(n_att):
        inside = (att_layers <= tower_att[j]) & (att_layers + limits >= all_pts[j + 1])
        a_eq[:, j] = inside.astype(float)
    b_eq = exp_losses

    info_available = a_eq.sum(axis=0) > 0

    # Inequality constraints (linprog convention: a_ub @ x <= b_ub)
    a_ub_rows: list[np.ndarray] = []
    b_ub_vals: list[float] = []

    # Monotone decreasing RoL:
    # x[i]/lim[i] >= x[i+1]/lim[i+1]  →  -x[i]/lim[i] + x[i+1]/lim[i+1] <= 0
    for i in range(n_att - 1):
        row = np.zeros(n_att)
        row[i] = -1.0 / tower_lim[i]
        row[i + 1] = 0.0 if np.isinf(tower_lim[i + 1]) else 1.0 / tower_lim[i + 1]
        a_ub_rows.append(row)
        b_ub_vals.append(0.0)

    # Frequency constraints at thresholds:
    freq_arr = np.full(n_att, np.nan)
    if thresholds is not None and len(thresholds) > 0 and threshold_freqs is not None:
        thr = np.asarray(thresholds, dtype=float)
        fqs = np.asarray(threshold_freqs, dtype=float)
        for thresh, freq in zip(thr, fqs, strict=True):
            idx = int(np.searchsorted(tower_att, thresh))
            if idx < n_att and tower_att[idx] == thresh:
                freq_arr[idx] = freq
                # x[idx]/lim[idx] <= freq
                row_up = np.zeros(n_att)
                row_up[idx] = 1.0 / tower_lim[idx]
                a_ub_rows.append(row_up)
                b_ub_vals.append(float(freq))
                # x[idx-1]/lim[idx-1] >= freq  →  -x[idx-1]/lim[idx-1] <= -freq
                if idx > 0:
                    row_lo = np.zeros(n_att)
                    row_lo[idx - 1] = -1.0 / tower_lim[idx - 1]
                    a_ub_rows.append(row_lo)
                    b_ub_vals.append(-float(freq))

    a_ub = np.array(a_ub_rows) if a_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    bounds = [(0.0, None)] * n_att

    # Maximise each info-available sub-layer's EL; average all solutions.
    indices = np.where(info_available)[0]
    solution_sum = np.zeros(n_att)
    n_infeasible = 0

    for idx in indices:
        c = np.zeros(n_att)
        c[idx] = -1.0  # min(-x[idx]) ≡ max(x[idx])
        res = linprog(
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if res.success:
            solution_sum += res.x
        else:
            n_infeasible += 1

    n_info = len(indices)
    if n_info > 0:
        solution = solution_sum / n_info
    else:
        solution = solution_sum

    status = 1 if n_infeasible > 0 else 0
    exp_loss_out = solution if status == 0 else np.full(n_att, np.nan)

    return _TowerResult(
        att=tower_att,
        limits=tower_lim,
        exp_loss=exp_loss_out,
        frequency=freq_arr,
        info_available=info_available,
        status=status,
    )
