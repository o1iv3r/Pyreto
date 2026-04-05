"""LP matching engine for piecewise-Pareto layer-loss matching.

Port of lp_functions.R from the R Pareto package.
Task 11a: _solve_lp and _calculate_layer_losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from pyreto.pareto import (
    pareto_extrapolation,
    pareto_find_alpha_btw_fq_layer,
    pareto_find_alpha_btw_fqs,
    pareto_find_alpha_btw_layers,
    pareto_layer_mean,
)


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


def _calculate_layer_losses(
    att_layers: np.ndarray,
    limits: np.ndarray,
    exp_losses: np.ndarray,
    thresholds: np.ndarray | None = None,
    threshold_freqs: np.ndarray | None = None,
    *,
    overlapping: bool = False,
    default_alpha: float = 2.0,
    ignore_inconsistent_references: bool = False,
) -> _TowerResult:
    """Port of ``calculate_layer_losses`` from ``lp_functions.R``.

    Builds a granular tower of sub-layers from the reference data, resolves
    any overlapping or inconsistent references via LP if necessary, then
    fills in missing expected losses by Pareto interpolation.

    Parameters
    ----------
    att_layers:
        Attachment points of reference layers with EL data (sorted ascending).
    limits:
        Covers of reference layers.
    exp_losses:
        Expected losses of reference layers.
    thresholds:
        Optional attachment points at which excess frequencies are known.
    threshold_freqs:
        Excess frequency at each threshold.
    overlapping:
        If True, call the LP solver unconditionally.  If False (default),
        build a non-overlapping tower first and only call the LP if RoLs or
        frequencies are inconsistent.
    default_alpha:
        Pareto alpha used for gap-filling when no bracketing reference exists.
    ignore_inconsistent_references:
        If True, drop inconsistent references rather than returning an error.

    Returns
    -------
    _TowerResult
        status 0=OK, 1=some refs ignored, 2=error, 4=inconsistent refs.
    """
    att_layers = np.asarray(att_layers, dtype=float)
    limits = np.asarray(limits, dtype=float)
    exp_losses = np.asarray(exp_losses, dtype=float)
    n_layers = len(att_layers)

    # ------------------------------------------------------------------
    # Step 1: build the initial tower
    # ------------------------------------------------------------------

    if overlapping:
        lp = _solve_lp(att_layers, limits, exp_losses, thresholds, threshold_freqs)
        if lp.status != 0:
            if not ignore_inconsistent_references:
                return _TowerResult(
                    att=lp.att,
                    limits=lp.limits,
                    exp_loss=lp.exp_loss,
                    frequency=lp.frequency,
                    info_available=lp.info_available,
                    status=4,
                )
        tower_att = lp.att
        tower_lim = lp.limits
        el_info = lp.info_available.copy()
        tower_el = lp.exp_loss.copy()
        tower_freq = lp.frequency.copy()
        status = lp.status
    else:
        # Non-overlapping: sort layers, add threshold rows, fill gaps.
        order = np.argsort(att_layers)
        s_att = att_layers[order]
        s_lim = limits[order]
        s_el = exp_losses[order]

        # Start tower from sorted layers
        t_att = list(s_att)
        t_lim = list(s_lim)
        t_el = list(s_el)
        t_freq = [np.nan] * n_layers
        t_info = [True] * n_layers  # EL came from input

        # Merge threshold rows
        if thresholds is not None and len(thresholds) > 0:
            thr = np.asarray(thresholds, dtype=float)
            fqs = np.asarray(threshold_freqs, dtype=float)
            for thresh, freq in zip(np.sort(thr), fqs[np.argsort(thr)], strict=True):
                match = np.searchsorted(t_att, thresh)
                if match < len(t_att) and t_att[match] == thresh:
                    t_freq[match] = float(freq)
                else:
                    # Insert new threshold-only row
                    t_att.insert(match, float(thresh))
                    t_lim.insert(match, np.nan)  # filled below
                    t_el.insert(match, np.nan)
                    t_freq.insert(match, float(freq))
                    t_info.insert(match, False)

        n = len(t_att)

        # Fill in limits for threshold-only rows
        for i in range(n):
            if np.isnan(t_lim[i]):
                if i < n - 1:
                    t_lim[i] = t_att[i + 1] - t_att[i]
                else:
                    t_lim[i] = np.inf

        # Fill gaps (sub-layers between consecutive layers that don't touch)
        new_rows: list[tuple[float, float, float, float, bool]] = []
        for i in range(n - 1):
            gap_start = t_att[i] + t_lim[i]
            gap_end = t_att[i + 1]
            if gap_start < gap_end - 1e-12:
                new_rows.append((gap_start, gap_end - gap_start, np.nan, np.nan, False))

        for att_g, lim_g, el_g, freq_g, info_g in new_rows:
            idx = np.searchsorted([r[0] for r in [(a,) for a in t_att]], att_g)
            t_att.insert(idx, att_g)
            t_lim.insert(idx, lim_g)
            t_el.insert(idx, el_g)
            t_freq.insert(idx, freq_g)
            t_info.insert(idx, info_g)

        # Ensure unlimited final row
        n = len(t_att)
        last_exit = t_att[-1] + t_lim[-1]
        if not np.isinf(last_exit):
            t_att.append(last_exit)
            t_lim.append(np.inf)
            t_el.append(np.nan)
            t_freq.append(np.nan)
            t_info.append(False)

        tower_att = np.array(t_att, dtype=float)
        tower_lim = np.array(t_lim, dtype=float)
        tower_el = np.array(t_el, dtype=float)
        tower_freq = np.array(t_freq, dtype=float)
        el_info = np.array(t_info, dtype=bool)
        n = len(tower_att)

        # Check RoL / frequency consistency: interleave freqs and RoLs and
        # verify the combined sequence is non-increasing.
        rols = np.where(np.isinf(tower_lim), 0.0, tower_el / tower_lim)
        fq_rol = np.empty(2 * n)
        fq_rol[0::2] = tower_freq
        fq_rol[1::2] = rols
        fq_rol_valid = fq_rol[~np.isnan(fq_rol)]

        status = 0
        if len(fq_rol_valid) > 1 and np.max(np.diff(fq_rol_valid)) > 0:
            # Inconsistent references — attempt LP resolution
            if not ignore_inconsistent_references:
                return _TowerResult(
                    att=tower_att,
                    limits=tower_lim,
                    exp_loss=tower_el,
                    frequency=tower_freq,
                    info_available=el_info,
                    status=4,
                )
            # Try progressively dropping inconsistent references
            use_layer = np.zeros(n_layers, dtype=bool)
            use_layer[0] = True
            for i in range(1, n_layers):
                use_layer[i] = True
                probe = _solve_lp(
                    att_layers[use_layer],
                    limits[use_layer],
                    exp_losses[use_layer],
                    thresholds,
                    threshold_freqs,
                )
                if probe.status != 0:
                    use_layer[i] = False

            n_thr = len(thresholds) if thresholds is not None else 0
            use_thr = np.zeros(n_thr, dtype=bool)
            for i in range(n_thr):
                use_thr[i] = True
                probe = _solve_lp(
                    att_layers[use_layer],
                    limits[use_layer],
                    exp_losses[use_layer],
                    thresholds[use_thr] if thresholds is not None else None,
                    threshold_freqs[use_thr] if threshold_freqs is not None else None,
                )
                if probe.status != 0:
                    use_thr[i] = False

            lp = _solve_lp(
                att_layers[use_layer],
                limits[use_layer],
                exp_losses[use_layer],
                thresholds[use_thr] if thresholds is not None and n_thr > 0 else None,
                threshold_freqs[use_thr] if threshold_freqs is not None and n_thr > 0 else None,
            )
            tower_att = lp.att
            tower_lim = lp.limits
            tower_el = lp.exp_loss.copy()
            tower_freq = lp.frequency.copy()
            el_info = lp.info_available.copy()
            status = 1  # some refs ignored

    # ------------------------------------------------------------------
    # Step 2: gap-fill missing expected losses
    # ------------------------------------------------------------------

    n = len(tower_att)
    el = tower_el.copy()
    freq = tower_freq.copy()

    info_has_data = el_info | ~np.isnan(freq)
    index_info = np.where(info_has_data)[0]

    # Handle single-row tower
    if n == 1:
        if np.isnan(freq[0]):
            freq[0] = el[0] / float(pareto_layer_mean(tower_lim[0], tower_att[0], default_alpha))
        elif not el_info[0]:
            el[0] = float(pareto_layer_mean(tower_lim[0], tower_att[0], default_alpha)) * freq[0]
        return _TowerResult(
            att=tower_att,
            limits=tower_lim,
            exp_loss=el,
            frequency=freq,
            info_available=el_info,
            status=status,
        )

    # Fill sub-layers 0..n-2 (all except the final unlimited layer)
    for i in range(n - 1):
        if el_info[i]:
            continue

        before = index_info[index_info <= i]
        after = index_info[index_info > i]
        if len(before) == 0 or len(after) == 0:
            continue
        idx1 = int(before[-1])
        idx2 = int(after[0])

        if el_info[idx1] and not np.isnan(freq[idx2]):
            # Case 1: known EL at idx1, known frequency at idx2
            if freq[idx2] > 0:
                alpha = pareto_find_alpha_btw_fq_layer(
                    tower_att[idx2],
                    freq[idx2],
                    tower_lim[idx1],
                    tower_att[idx1],
                    el[idx1],
                )
                fq1 = freq[idx2] * (tower_att[idx2] / tower_att[idx1]) ** alpha
                el[i] = fq1 * float(
                    pareto_layer_mean(tower_lim[i], tower_att[i], alpha, t=tower_att[idx1])
                )
            else:
                el[i] = float(
                    pareto_extrapolation(
                        tower_lim[idx1],
                        tower_att[idx1],
                        tower_lim[i],
                        tower_att[i],
                        default_alpha,
                        el[idx1],
                        truncation=tower_att[idx2],
                    )
                )
        elif not np.isnan(freq[idx1]) and not np.isnan(freq[idx2]):
            # Case 2: frequencies known at both bracketing points
            if freq[idx2] > 0:
                alpha = pareto_find_alpha_btw_fqs(
                    tower_att[idx1],
                    freq[idx1],
                    tower_att[idx2],
                    freq[idx2],
                )
                el[i] = freq[idx1] * float(
                    pareto_layer_mean(tower_lim[i], tower_att[i], alpha, t=tower_att[idx1])
                )
            else:
                el[i] = freq[idx1] * float(
                    pareto_layer_mean(
                        tower_lim[i],
                        tower_att[i],
                        default_alpha,
                        t=tower_att[idx1],
                        truncation=tower_att[idx2],
                    )
                )
        elif el_info[idx1] and el_info[idx2]:
            # Case 3: ELs known at both bracketing points
            if el[idx2] > 0:
                alpha = pareto_find_alpha_btw_layers(
                    tower_lim[idx1],
                    tower_att[idx1],
                    el[idx1],
                    tower_lim[idx2],
                    tower_att[idx2],
                    el[idx2],
                )
                el[i] = float(
                    pareto_extrapolation(
                        tower_lim[idx1],
                        tower_att[idx1],
                        tower_lim[i],
                        tower_att[i],
                        alpha,
                        el[idx1],
                    )
                )
            else:
                el[i] = float(
                    pareto_extrapolation(
                        tower_lim[idx1],
                        tower_att[idx1],
                        tower_lim[i],
                        tower_att[i],
                        default_alpha,
                        el[idx1],
                        truncation=tower_att[idx2],
                    )
                )
        else:
            # Case 4: frequency known at idx1, EL known at idx2
            if el[idx2] > 0:
                alpha = pareto_find_alpha_btw_fq_layer(
                    tower_att[idx1],
                    freq[idx1],
                    tower_lim[idx2],
                    tower_att[idx2],
                    el[idx2],
                )
                el[i] = freq[idx1] * float(
                    pareto_layer_mean(tower_lim[i], tower_att[i], alpha, t=tower_att[idx1])
                )
            else:
                el[i] = freq[idx1] * float(
                    pareto_layer_mean(
                        tower_lim[i],
                        tower_att[i],
                        default_alpha,
                        t=tower_att[idx1],
                        truncation=tower_att[idx2],
                    )
                )

    # Fill the final unlimited sub-layer (index n-1)
    last = n - 1
    if not el_info[last]:
        if el[last - 1] == 0.0:
            el[last] = 0.0
        elif not np.isnan(freq[last]):
            el[last] = freq[last] * float(pareto_layer_mean(np.inf, tower_att[last], default_alpha))
        else:
            if not np.isnan(freq[last - 1]):
                alpha = pareto_find_alpha_btw_fq_layer(
                    tower_att[last - 1],
                    freq[last - 1],
                    tower_lim[last - 1],
                    tower_att[last - 1],
                    el[last - 1],
                )
                freq[last] = freq[last - 1] * (tower_att[last - 1] / tower_att[last]) ** alpha
                eff_alpha = max(alpha, 1.1)
                el[last] = freq[last] * float(pareto_layer_mean(np.inf, tower_att[last], eff_alpha))
            elif n >= 3:
                alpha = pareto_find_alpha_btw_layers(
                    tower_lim[last - 2],
                    tower_att[last - 2],
                    el[last - 2],
                    tower_lim[last - 1],
                    tower_att[last - 1],
                    el[last - 1],
                )
                freq[last] = (
                    el[last - 1]
                    / float(pareto_layer_mean(tower_lim[last - 1], tower_att[last - 1], alpha))
                    * (tower_att[last - 1] / tower_att[last]) ** alpha
                )
                eff_alpha = max(alpha, 1.1)
                el[last] = freq[last] * float(pareto_layer_mean(np.inf, tower_att[last], eff_alpha))
            else:
                el[last] = float(
                    pareto_extrapolation(
                        tower_lim[last - 1],
                        tower_att[last - 1],
                        np.inf,
                        tower_att[last],
                        default_alpha,
                        el[last - 1],
                    )
                )
                freq[last] = el[last] / float(
                    pareto_layer_mean(np.inf, tower_att[last], default_alpha)
                )

    return _TowerResult(
        att=tower_att,
        limits=tower_lim,
        exp_loss=el,
        frequency=freq,
        info_available=el_info,
        status=status,
    )
