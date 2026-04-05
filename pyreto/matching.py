"""LP matching engine and alpha-fitting for piecewise-Pareto layer-loss matching.

Port of lp_functions.R and FitPP.R from the R Pareto package.
Task 11a: _solve_lp and _calculate_layer_losses.
Task 11b: _calculate_taus, _calculate_alphas, _fit_pp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, linprog, minimize_scalar

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


# ---------------------------------------------------------------------------
# Alpha-fitting engine (Task 11b) — port of FitPP.R
# ---------------------------------------------------------------------------


@dataclass
class _FitPPResult:
    """Result of the alpha-fitting engine (_fit_pp).

    Attributes
    ----------
    t:
        Breakpoints of the piecewise-Pareto distribution (same convention as
        the R Pareto package: one attachment point per segment).
    alpha:
        Pareto alpha for each segment (same length as ``t``).
    status:
        ``"OK"`` on success, or an error description string.
    """

    t: list[float]
    alpha: list[float]
    status: str = "OK"


# ---------------------------------------------------------------------------
# Private math helpers
# ---------------------------------------------------------------------------


def _ll(a: float, b: float, alpha: float) -> float:
    """Integral of (a/x)^alpha from a to b.

    Port of the inner ``LL`` function in FitPP.R / lp_functions.R.

    LL(a, b, alpha) =
        a * log(b/a)           if alpha == 1
        a/(1-alpha) * ((b/a)^(1-alpha) - 1)   otherwise
    """
    if alpha == 1.0:
        return a * (np.log(b) - np.log(a))
    return a / (1.0 - alpha) * ((b / a) ** (1.0 - alpha) - 1.0)


def _lambda_fn(
    t: float,
    alpha: float,
    s_0: float,
    s_1: float,
    a_0: float,
    a_1: float,
) -> float:
    """Expected loss of layer [a_0, a_1) xs a_0 with a two-piece Pareto.

    The break between the two pieces is at ``t`` with alphas ``alpha`` (below)
    and ``alpha_1`` (above), where ``alpha_1`` is determined by the requirement
    that frequency at ``a_1`` equals ``s_1``:

        alpha_1 = (log(s_1/s_0) - alpha*log(a_0/t)) / log(t/a_1)

    Port of the inner ``lambda`` function in FitPP.R.
    """
    log_t_a1 = np.log(t / a_1)
    if log_t_a1 == 0.0:
        alpha_1 = alpha
    else:
        alpha_1 = (np.log(s_1 / s_0) - alpha * np.log(a_0 / t)) / log_t_a1
    return s_0 * _ll(a_0, t, alpha) + s_0 * (a_0 / t) ** alpha * _ll(t, a_1, alpha_1)


def _calculate_taus(
    s_0: float,
    s_1: float,
    a_0: float,
    a_1: float,
    l_0: float,
    tolerance: float = 1e-10,
) -> tuple[float, float]:
    """Find the valid breakpoint range [tau_l, tau_u] for a single segment.

    Port of ``Calculate_taus`` from ``FitPP.R``.

    ``tau_u`` is the breakpoint at which both pieces have the *same* alpha (the
    single-alpha solution); ``tau_l`` is the breakpoint at which the first
    piece has alpha=0 (uniform distribution).

    Returns
    -------
    (tau_l, tau_u) : tuple[float, float]
    """
    delta = tolerance * (a_1 - a_0)
    tol_abs = tolerance * (a_1 - a_0)

    # --- tau_u: solve f(x) = lambda(x, log(s1/s0)/log(a0/x)) - l_0 = 0 ---
    def f(x: float) -> float:
        log_a0_x = np.log(a_0 / x)
        if log_a0_x == 0.0:
            alpha = 0.0
        else:
            alpha = np.log(s_1 / s_0) / log_a0_x
        return _lambda_fn(x, alpha, s_0, s_1, a_0, a_1) - l_0

    tau_u: float | None = None
    try:
        lo, hi = a_0 + delta, a_1 - delta
        if lo < hi and np.sign(f(lo)) != np.sign(f(hi)):
            tau_u = float(brentq(f, lo, hi, xtol=tol_abs))
    except (ValueError, ZeroDivisionError):
        pass

    if tau_u is None:
        mid = (a_0 + a_1) / 2.0
        tau_u = a_0 if f(mid) < 0 else a_1

    # --- tau_l: solve g(x) = lambda(x, 0) - l_0 = 0 ---
    def g(x: float) -> float:
        return _lambda_fn(x, 0.0, s_0, s_1, a_0, a_1) - l_0

    tau_l: float | None = None
    try:
        lo2, hi2 = a_0, a_1
        if np.sign(g(lo2)) != np.sign(g(hi2)):
            tau_l = float(brentq(g, lo2, hi2, xtol=tol_abs))
    except (ValueError, ZeroDivisionError):
        pass

    if tau_l is None:
        mid = (a_0 + a_1) / 2.0
        tau_l = a_0 if g(mid) > 0 else a_1

    return float(tau_l), float(tau_u)


def _calculate_alphas(
    s_0: float,
    s_1: float,
    a_0: float,
    a_1: float,
    l_0: float,
    t: float,
    alpha_max: float = 100.0,
    tolerance: float = 1e-10,
) -> tuple[float, float]:
    """Find (alpha_0, alpha_1) for a two-piece Pareto matching layer loss l_0.

    Port of ``Calculate_alphas`` from ``FitPP.R``.

    Given the breakpoint ``t``, alpha_0 is found by root-finding
    (lambda(t, alpha_0) == l_0) and alpha_1 is derived analytically.
    """
    tol_abs = tolerance * (a_1 - a_0)

    def f(alpha: float) -> float:
        val = _lambda_fn(t, alpha, s_0, s_1, a_0, a_1) - l_0
        if not np.isnan(val):
            return val
        # Fallback: nudge alpha down (as in R source)
        for i in range(1, 21):
            val2 = _lambda_fn(t, alpha / 1.1**i, s_0, s_1, a_0, a_1) - l_0
            if not np.isnan(val2):
                return val2
        return val  # still nan — let caller handle

    alpha_0: float
    try:
        f0, fmax = f(0.0), f(alpha_max)
        if np.sign(f0) != np.sign(fmax):
            alpha_0 = float(brentq(f, 0.0, alpha_max, xtol=tol_abs))
        else:
            # Pick the endpoint closest to zero
            alpha_0 = 0.0 if abs(f0) <= abs(fmax) else alpha_max
    except (ValueError, ZeroDivisionError):
        alpha_0 = 0.0

    log_t_a1 = np.log(t / a_1)
    if log_t_a1 == 0.0:
        alpha_1 = alpha_0
    else:
        alpha_1 = (np.log(s_1 / s_0) - alpha_0 * np.log(a_0 / t)) / log_t_a1
        alpha_1 = min(alpha_max, alpha_1)

    return float(max(alpha_0, 0.0)), float(max(alpha_1, 0.0))


def _fit_pp(
    a: np.ndarray,
    s: np.ndarray,
    layer_losses: np.ndarray,
    truncation: float | None,
    tolerance: float = 1e-10,
    alpha_max: float = 100.0,
    minimize_ratios: bool = True,
    merge_tolerance: float = 1e-6,
) -> _FitPPResult:
    """Fit a piecewise-Pareto distribution to layer frequencies and expected losses.

    Port of ``Fit_PP`` from ``FitPP.R``.

    Parameters
    ----------
    a:
        Attachment points (n values, strictly ascending, positive).
    s:
        Excess frequencies at each attachment point (n values, strictly
        descending, all positive).
    layer_losses:
        Expected loss of each contiguous layer [a[k], a[k+1]) xs a[k].
        Accepts n-1 values (finite layers only) or n values (full R convention
        where the last element is the infinite-tail layer EL).
    truncation:
        Upper truncation point, or ``None`` for no truncation.
    tolerance:
        Root-finding tolerance.
    alpha_max:
        Maximum alpha value used during root-finding.
    minimize_ratios:
        If True, choose the breakpoint within [tau_l, tau_u] that minimises
        the ratio max(alpha_0, alpha_1) / min(alpha_0, alpha_1).
    merge_tolerance:
        Consecutive segments whose alphas differ by less than this are merged.

    Returns
    -------
    _FitPPResult
        .t      - breakpoints (one per segment)
        .alpha  - Pareto alpha for each segment
        .status - "OK" or error description
    """
    a = np.asarray(a, dtype=float)
    s = np.asarray(s, dtype=float)
    ll = np.asarray(layer_losses, dtype=float)

    n = len(a)
    result_err = _FitPPResult(t=[], alpha=[], status="")

    if len(s) != n:
        result_err.status = "a and s must have same length!"
        return result_err
    # ll must have n elements: n-1 finite layers + 1 infinite-tail layer (R convention).
    # Also accepts n-1 elements; in that case the topmost alpha is derived from
    # the last two frequencies via a simple power-law.
    if len(ll) not in (n - 1, n):
        result_err.status = "layer_losses must have n-1 or n elements (n = len(a))."
        return result_err
    if n < 2 or np.min(np.diff(a)) <= 0:
        result_err.status = "a must be ascending"
        return result_err
    if np.max(np.diff(s)) >= 0:
        result_err.status = "s must be descending"
        return result_err
    if a[0] <= 0:
        result_err.status = "a must be positive."
        return result_err
    if s[n - 1] <= 0:
        result_err.status = "s must be positive."
        return result_err

    ll_has_tail = len(ll) == n  # True -> R convention; False -> tail alpha from freqs

    # Output arrays of length 2*n - 1 (as in R source).
    # Positions 0, 2, ..., 2*(n-1) hold attachment points a[k].
    # Positions 1, 3, ..., 2*(n-1)-1 hold intermediate breakpoints.
    q = 2 * n - 1
    t_arr = np.zeros(q)
    alpha_arr = np.zeros(q)

    # Place the n attachment points at even positions (0-based: 0, 2, 4, ...)
    t_arr[0::2] = a

    # Alpha for the topmost segment (position q-1)
    if ll_has_tail:
        # Full R convention: use pareto_find_alpha_btw_fq_layer with ll[n-1]
        alpha_arr[q - 1] = pareto_find_alpha_btw_fq_layer(
            a[n - 1],
            s[n - 1],
            np.inf,
            a[n - 1],
            ll[n - 1],
            max_alpha=alpha_max,
            tolerance=tolerance,
            truncation=truncation,
        )
    else:
        # Derive topmost alpha from the last two frequencies (power-law)
        log_ratio = np.log(a[n - 1] / a[n - 2])
        if log_ratio > 0:
            alpha_arr[q - 1] = float(np.log(s[n - 2] / s[n - 1]) / log_ratio)
        else:
            alpha_arr[q - 1] = alpha_max
        alpha_arr[q - 1] = max(0.0, min(alpha_max, alpha_arr[q - 1]))

    # For each consecutive pair of attachment points, find the breakpoint and alphas
    for k in range(n - 1):
        taus = list(_calculate_taus(s[k], s[k + 1], a[k], a[k + 1], ll[k], tolerance=tolerance))

        # Clamp taus to [lower_bound, upper_bound] as in R
        lower_bound = min(
            a[k] * (s[k] / s[k + 1]) ** (1.0 / alpha_max),
            (a[k] + a[k + 1]) / 2.0,
        )
        upper_bound = max(
            a[k + 1] * (s[k + 1] / s[k]) ** (1.0 / alpha_max),
            (a[k] + a[k + 1]) / 2.0,
        )

        if taus[1] < lower_bound:
            taus = [lower_bound, lower_bound]
        elif taus[0] > upper_bound:
            taus = [upper_bound, upper_bound]
        else:
            taus[0] = max(taus[0], lower_bound)
            taus[1] = min(taus[1], upper_bound)

        t_mid = (taus[0] + taus[1]) / 2.0

        if minimize_ratios and taus[0] < taus[1]:

            def _penalty(tv: float, _k: int = k) -> float:
                al0, al1 = _calculate_alphas(
                    s[_k],
                    s[_k + 1],
                    a[_k],
                    a[_k + 1],
                    ll[_k],
                    tv,
                    tolerance=tolerance,
                    alpha_max=alpha_max,
                )
                if al0 == 0.0 and al1 == 0.0:
                    return 1.0
                mn = min(al0, al1)
                mx = max(al0, al1)
                if mn == 0.0:
                    return 1000.0
                ratio = mx / mn
                return min(ratio, 1000.0)

            opt = minimize_scalar(
                _penalty,
                bounds=(taus[0], taus[1]),
                method="bounded",
                options={"xatol": tolerance * (a[k + 1] - a[k])},
            )
            t_mid = float(opt.x)
        elif not minimize_ratios:
            t_mid = taus[0]

        # t_arr index for breakpoint between a[k] and a[k+1] is 2k+1
        t_arr[2 * k + 1] = t_mid

        al0, al1 = _calculate_alphas(
            s[k],
            s[k + 1],
            a[k],
            a[k + 1],
            ll[k],
            t_mid,
            tolerance=tolerance,
            alpha_max=alpha_max,
        )
        alpha_arr[2 * k] = al0  # segment below breakpoint (starts at a[k])
        alpha_arr[2 * k + 1] = al1  # segment above breakpoint (starts at t_mid)

    # Merge consecutive segments with equal alpha
    is_equal = np.concatenate(([False], np.abs(np.diff(alpha_arr)) < merge_tolerance))
    t_out = t_arr[~is_equal].tolist()
    alpha_out = alpha_arr[~is_equal].tolist()

    return _FitPPResult(t=t_out, alpha=alpha_out, status="OK")
