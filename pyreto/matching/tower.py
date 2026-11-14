"""Fill a tower of sub-layers from reference layers (calculate_layer_losses in lp_functions.R)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pyreto.matching.lp import _solve_lp, _TowerResult
from pyreto.pareto import (
    pareto_extrapolation,
    pareto_find_alpha_btw_fq_layer,
    pareto_find_alpha_btw_fqs,
    pareto_find_alpha_btw_layers,
    pareto_layer_mean,
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

    if overlapping:
        tower = _solve_lp(att_layers, limits, exp_losses, thresholds, threshold_freqs)
    else:
        # R sorts the layers here, which also fixes the order in which
        # inconsistent references are dropped below.
        order = np.argsort(att_layers, kind="stable")
        att_layers, limits, exp_losses = att_layers[order], limits[order], exp_losses[order]
        tower = _build_non_overlapping_tower(
            att_layers, limits, exp_losses, thresholds, threshold_freqs
        )

    if tower.status != 0:
        if not ignore_inconsistent_references:
            return replace(tower, status=4)
        tower = _drop_inconsistent_references(
            att_layers, limits, exp_losses, thresholds, threshold_freqs
        )

    return _fill_missing_losses(tower, default_alpha)


def _drop_inconsistent_references(
    att_layers: np.ndarray,
    limits: np.ndarray,
    exp_losses: np.ndarray,
    thresholds: np.ndarray | None,
    threshold_freqs: np.ndarray | None,
) -> _TowerResult:
    """Greedily keep the references that are consistent with the ones kept before.

    Layers are added one at a time (first layer always kept, no thresholds yet),
    then thresholds one at a time; a reference is dropped if the LP becomes
    infeasible.  Returns the LP tower of the kept references with status 1.
    """
    thr = np.asarray(thresholds if thresholds is not None else [], dtype=float)
    thr_fq = np.asarray(threshold_freqs if threshold_freqs is not None else [], dtype=float)

    def _lp(use_layer: np.ndarray, use_thr: np.ndarray) -> _TowerResult:
        return _solve_lp(
            att_layers[use_layer],
            limits[use_layer],
            exp_losses[use_layer],
            thr[use_thr],
            thr_fq[use_thr],
        )

    use_layer = np.zeros(len(att_layers), dtype=bool)
    use_thr = np.zeros(len(thr), dtype=bool)
    use_layer[:1] = True
    for i in range(1, len(att_layers)):
        use_layer[i] = True
        if _lp(use_layer, use_thr).status != 0:
            use_layer[i] = False
    for i in range(len(thr)):
        use_thr[i] = True
        if _lp(use_layer, use_thr).status != 0:
            use_thr[i] = False

    return replace(_lp(use_layer, use_thr), status=1)


def _build_non_overlapping_tower(
    att_layers: np.ndarray,
    limits: np.ndarray,
    exp_losses: np.ndarray,
    thresholds: np.ndarray | None,
    threshold_freqs: np.ndarray | None,
) -> _TowerResult:
    """Add threshold rows and gap layers to the (sorted) layers, then check consistency.

    Returns status 4 if RoLs and frequencies are not non-increasing, else 0.
    """
    n_layers = len(att_layers)

    # Start tower from the layers
    t_att = list(att_layers)
    t_lim = list(limits)
    t_el = list(exp_losses)
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
        idx = np.searchsorted(t_att, att_g)
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

    inconsistent = len(fq_rol_valid) > 1 and np.max(np.diff(fq_rol_valid)) > 0
    status = 4 if inconsistent else 0

    return _TowerResult(
        att=tower_att,
        limits=tower_lim,
        exp_loss=tower_el,
        frequency=tower_freq,
        info_available=el_info,
        status=status,
    )


def _fill_missing_losses(tower: _TowerResult, default_alpha: float) -> _TowerResult:
    """Fill sub-layers without expected-loss information by Pareto interpolation."""
    tower_att = tower.att
    tower_lim = tower.limits
    el_info = tower.info_available.copy()
    status = tower.status
    n = len(tower_att)
    el = tower.exp_loss.copy()
    freq = tower.frequency.copy()

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
