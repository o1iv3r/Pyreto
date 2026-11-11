"""Public entry point: match a piecewise Pareto distribution to layer losses."""

from __future__ import annotations

import warnings

import numpy as np

from pyreto.matching.fit_pp import _fit_pp
from pyreto.pareto import (
    pareto_find_alpha_btw_fq_layer,
    pareto_find_alpha_btw_layers,
    pareto_layer_mean,
)
from pyreto.ppp_model import PPPModel

# ---------------------------------------------------------------------------
# Top-level matching function (Task 11c)
# ---------------------------------------------------------------------------


def piecewise_pareto_match_layer_losses(
    attachment_points: np.ndarray,
    expected_layer_losses: np.ndarray,
    *,
    frequencies: np.ndarray | None = None,
    truncation: float | None = None,
    truncation_type: str = "lp",
    dispersion: float = 1.0,
    tolerance: float = 1e-10,
    alpha_max: float = 100.0,
    merge_tolerance: float = 1e-6,
    rol_tolerance: float = 1e-6,
    minimize_ratios: bool = True,
) -> object:
    """Fit a PPPModel to reinsurance layer attachment points and expected losses.

    Port of ``PiecewisePareto_Match_Layer_Losses`` from ``Functions.R``.

    Parameters
    ----------
    attachment_points:
        Strictly ascending positive attachment points (k values).
    expected_layer_losses:
        Expected loss of each layer ``(AP[i+1] - AP[i]) xs AP[i]`` (k values).
        The last value is the expected loss of the unlimited layer ``Inf xs AP[k]``.
    frequencies:
        Optional excess frequencies at each attachment point.  ``np.nan``
        entries are filled in automatically.
    truncation:
        Upper truncation point, or ``None`` for no truncation.
    truncation_type:
        ``"lp"`` (truncated last Pareto piece) or ``"wd"`` (whole distribution
        truncated).
    dispersion:
        Variance-to-mean ratio of the claim-count distribution.
    tolerance:
        Numerical root-finding tolerance.
    alpha_max:
        Maximum Pareto alpha used in root-finding.
    merge_tolerance:
        Consecutive segments with alpha difference below this are merged.
    rol_tolerance:
        Tolerance for rate-on-line consistency checks.
    minimize_ratios:
        Passed to ``_fit_pp``.

    Returns
    -------
    PPPModel
        Fitted model.  ``model.is_valid()`` is True on success.
    """

    ap = np.asarray(attachment_points, dtype=float)
    el = np.asarray(expected_layer_losses, dtype=float)
    k = len(ap)

    # ------------------------------------------------------------------
    # Create an empty (invalid) result for error returns
    # ------------------------------------------------------------------
    def _err(msg: str, status: int = 2) -> PPPModel:
        warnings.warn(msg)
        return PPPModel(
            fq=float("nan"),
            t=ap[:1],
            alpha=np.array([2.0]),
            status=status,
            comment=msg,
            truncation=None,
            truncation_type=truncation_type,
            dispersion=dispersion,
        )

    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------
    if k < 1 or not np.all(ap > 0) or not np.all(np.isfinite(ap)):
        return _err("attachment_points must be a vector of positive finite values.")
    if len(el) != k or not np.all(el >= 0) or not np.all(np.isfinite(el)):
        return _err(
            "expected_layer_losses must be non-negative finite values,"
            " same length as attachment_points."
        )
    if k > 1 and np.min(np.diff(ap)) <= 0:
        return _err("attachment_points must be strictly increasing.")
    if truncation_type not in ("lp", "wd"):
        return _err("truncation_type must be 'lp' or 'wd'.")
    if truncation is not None:
        if not (np.isfinite(truncation) and truncation > 0):
            return _err("truncation must be a positive finite number.")
        if truncation <= ap[-1]:
            return _err("truncation must be greater than max(attachment_points).")

    # ------------------------------------------------------------------
    # Build ELL (expected losses of contiguous layers) and limits
    # ------------------------------------------------------------------
    ell = el.copy()  # el[k-1] is the infinite-tail layer

    limits = np.empty(k, dtype=float)
    if k > 1:
        limits[: k - 1] = np.diff(ap)
    limits[k - 1] = np.inf

    # ------------------------------------------------------------------
    # Handle all-zero EL edge case
    # ------------------------------------------------------------------
    if np.min(ell) == 0.0:
        idx_zero = int(np.argmax(ell == 0))  # first zero position
        if idx_zero == 0:
            # First EL is zero -> FQ = 0
            return PPPModel(
                fq=0.0,
                t=ap[:1],
                alpha=np.array([2.0]),
                status=0,
                comment="OK",
                truncation=truncation,
                truncation_type=truncation_type,
                dispersion=dispersion,
            )
        # Non-zero ELs after a zero -> error
        if idx_zero < k - 1 and np.any(ell[idx_zero + 1 :] > 0):
            return _err(f"expected_layer_losses[{idx_zero}] == 0 but later values are > 0.")
        # Truncate at zero point
        k = idx_zero
        ell = ell[:k]
        ap = ap[:k]
        limits = limits[:k]
        limits[k - 1] = np.inf

    # ------------------------------------------------------------------
    # Handle k=1 (single-layer) edge case
    # ------------------------------------------------------------------
    if k == 1:
        fq_known = (
            float(frequencies[0])
            if frequencies is not None and not np.isnan(frequencies[0])
            else float("nan")
        )
        if np.isnan(fq_known):
            alpha_val = 2.0
            fq_val = float(
                ell[0] / pareto_layer_mean(np.inf, ap[0], alpha_val, truncation=truncation)
            )
        else:
            try:
                alpha_val = pareto_find_alpha_btw_fq_layer(
                    ap[0],
                    fq_known,
                    np.inf,
                    ap[0],
                    ell[0],
                    max_alpha=alpha_max,
                    tolerance=tolerance,
                    truncation=truncation,
                )
            except Exception:
                return _err("truncation too low.")
            if np.isnan(alpha_val):
                return _err("truncation too low.")
            fq_val = fq_known
        return PPPModel(
            fq=fq_val,
            t=ap,
            alpha=np.array([alpha_val]),
            status=0,
            comment="OK",
            truncation=truncation,
            truncation_type=truncation_type,
            dispersion=dispersion,
        )

    # ------------------------------------------------------------------
    # Initialise frequencies array
    # ------------------------------------------------------------------
    fqs = np.full(k, np.nan, dtype=float)
    if frequencies is not None:
        freq_in = np.asarray(frequencies, dtype=float)
        if len(freq_in) == k:
            fqs[:k] = freq_in[:k]

    # ------------------------------------------------------------------
    # Handle truncation_type="wd": add extra attachment point at truncation
    # ------------------------------------------------------------------
    truncation_wd = False
    original_truncation: float | None = truncation
    if truncation is not None and truncation_type == "wd":
        truncation_wd = True
        k += 1
        ap = np.append(ap, float(truncation))
        truncation = None

        # Adjust limit of penultimate layer
        limits[k - 2] = ap[k - 1] - ap[k - 2]
        limits = np.append(limits, np.inf)

        fqs = np.append(fqs, np.nan)

        # Frequency at last attachment point (before truncation) if unknown
        if np.isnan(fqs[k - 2]):
            try:
                alpha_wd = pareto_find_alpha_btw_layers(
                    limits[k - 3],
                    ap[k - 3],
                    ell[k - 3],
                    limits[k - 2],
                    ap[k - 2],
                    ell[k - 2],
                    truncation=ap[k - 1],
                )
                fqs[k - 2] = float(
                    ell[k - 2]
                    / pareto_layer_mean(limits[k - 2], ap[k - 2], alpha_wd, truncation=ap[k - 1])
                )
            except Exception:
                return _err("truncation too low.")

        try:
            alpha_wd = pareto_find_alpha_btw_fq_layer(
                ap[k - 2],
                fqs[k - 2],
                limits[k - 2],
                ap[k - 2],
                ell[k - 2],
                max_alpha=alpha_max,
                tolerance=tolerance,
                truncation=ap[k - 1],
            )
        except Exception:
            return _err("truncation too low.")
        alpha_wd = max(alpha_wd, 1.2)

        fqs[k - 1] = 0.0  # frequency at truncation point is zero
        f_wd = (ap[k - 2] / ap[k - 1]) ** alpha_wd
        add_fq = float(fqs[k - 2]) * f_wd / (1.0 - f_wd)

        fqs = fqs + add_fq
        ell_ext = ell + limits[: k - 1] * add_fq
        tail_ell = fqs[k - 1] * float(pareto_layer_mean(np.inf, ap[k - 1], alpha_wd))
        ell = np.append(ell_ext, tail_ell)

    # ------------------------------------------------------------------
    # Compute RoLs and merge inconsistent layers
    # ------------------------------------------------------------------
    rols = ell / limits
    merged_layer = np.zeros(k, dtype=bool)

    if k > 1 and np.max(rols[1:k] / rols[: k - 1]) >= 1.0 - rol_tolerance:
        warnings.warn("RoLs not strictly decreasing. Layers have been merged.")
        while k >= 3:
            ratios = rols[1:k] / rols[: k - 1]
            if np.max(ratios) < 1.0 - rol_tolerance:
                break
            pos = int(np.argmax(ratios))  # 0-based index into rols[1:]
            # Merge layers pos and pos+1 (0-based)
            ell[pos] = ell[pos] + ell[pos + 1]
            ell = np.delete(ell, pos + 1)
            ap = np.delete(ap, pos + 1)
            limits[pos] = limits[pos] + limits[pos + 1]
            limits = np.delete(limits, pos + 1)
            fqs = np.delete(fqs, pos + 1)
            merged_layer[pos] = True
            merged_layer = np.delete(merged_layer, pos + 1)
            k -= 1
            rols = ell / limits

    # ------------------------------------------------------------------
    # Clamp known frequencies to valid ranges
    # ------------------------------------------------------------------
    for i in range(k - 1):
        if not np.isnan(fqs[i]) and fqs[i] < rols[i] * (1.0 + rol_tolerance / 2.0):
            fqs[i] = rols[i] * (1.0 + rol_tolerance / 2.0)
        if not np.isnan(fqs[i + 1]) and fqs[i + 1] > rols[i] * (1.0 - rol_tolerance / 2.0):
            fqs[i + 1] = rols[i] * (1.0 - rol_tolerance / 2.0)

    # ------------------------------------------------------------------
    # Fill missing frequencies (forward pass: fill fqs[1] ... fqs[k-1])
    # ------------------------------------------------------------------
    for i in range(k - 1):
        if np.isnan(fqs[i + 1]):
            if i + 1 < k - 1:
                # Interior: derive alpha from two consecutive finite layers
                try:
                    alpha_btw = pareto_find_alpha_btw_layers(
                        limits[i],
                        ap[i],
                        ell[i],
                        limits[i + 1],
                        ap[i + 1],
                        ell[i + 1],
                    )
                    if not np.isnan(alpha_btw):
                        fqs[i + 1] = float(
                            ell[i + 1] / pareto_layer_mean(limits[i + 1], ap[i + 1], alpha_btw)
                        )
                    else:
                        fqs[i + 1] = (rols[i] + rols[i + 1]) / 2.0
                except Exception:
                    fqs[i + 1] = (rols[i] + rols[i + 1]) / 2.0
            else:
                # Last: derive alpha from finite layer and infinite tail
                trunc_last = truncation if not truncation_wd else None
                try:
                    alpha_btw = pareto_find_alpha_btw_layers(
                        limits[i],
                        ap[i],
                        ell[i],
                        np.inf,
                        ap[i + 1],
                        ell[i + 1],
                        truncation=trunc_last,
                    )
                    if not np.isnan(alpha_btw):
                        fqs[i + 1] = float(
                            ell[i + 1]
                            / pareto_layer_mean(np.inf, ap[i + 1], alpha_btw, truncation=trunc_last)
                        )
                    else:
                        fqs[i + 1] = rols[i] / 2.0
                except Exception:
                    return _err("truncation too low.")
            # Apply merged-layer adjustments (matching R)
            if merged_layer[i] and not merged_layer[i + 1 if i + 1 < k else i]:
                fqs[i + 1] = rols[i] * (1.0 - rol_tolerance / 2.0)
            elif not merged_layer[i] and merged_layer[min(i + 1, k - 1)]:
                fqs[i + 1] = rols[i + 1] * (1.0 + rol_tolerance / 2.0) if i + 1 < k else fqs[i + 1]

    # Fill fqs[0] if still missing
    if np.isnan(fqs[0]):
        try:
            alpha_btw = pareto_find_alpha_btw_layers(
                limits[0],
                ap[0],
                ell[0],
                limits[1],
                ap[1],
                ell[1],
            )
            fqs[0] = float(ell[0] / pareto_layer_mean(limits[0], ap[0], alpha_btw))
        except Exception:
            fqs[0] = rols[0] * (1.0 + rol_tolerance / 2.0)
        if merged_layer[0]:
            fqs[0] = rols[0] * (1.0 + rol_tolerance / 2.0)

    # Ensure fqs[0] > rols[0] (frequency at first AP must exceed RoL of first layer)
    if fqs[0] < rols[0] * (1.0 + rol_tolerance / 2.0):
        fqs[0] = rols[0] * (1.0 + rol_tolerance / 2.0)

    # ------------------------------------------------------------------
    # Check truncation feasibility
    # ------------------------------------------------------------------
    if truncation is not None:
        try:
            alpha_test = pareto_find_alpha_btw_fq_layer(
                ap[k - 1],
                fqs[k - 1],
                np.inf,
                ap[k - 1],
                ell[k - 1],
                max_alpha=alpha_max,
                tolerance=tolerance,
                truncation=truncation,
            )
            if np.isnan(alpha_test):
                return _err("truncation too low.")
        except Exception:
            return _err("truncation too low.")

    # ------------------------------------------------------------------
    # Normalise and call _fit_pp
    # ------------------------------------------------------------------
    fq0 = float(fqs[0])
    s_norm = fqs / fq0
    l_norm = ell / fq0

    fit_result = _fit_pp(
        ap,
        s_norm,
        l_norm,
        truncation,
        tolerance=tolerance,
        alpha_max=alpha_max,
        minimize_ratios=minimize_ratios,
        merge_tolerance=merge_tolerance,
    )
    if fit_result.status != "OK":
        return _err(f"_fit_pp: {fit_result.status}")

    t_out = np.array(fit_result.t, dtype=float)
    alpha_out = np.array(fit_result.alpha, dtype=float)

    fq_out = fq0 if not truncation_wd else float(fqs[0] - fqs[k - 1])

    return PPPModel(
        fq=fq_out,
        t=t_out,
        alpha=alpha_out,
        truncation=original_truncation,
        truncation_type=truncation_type,
        dispersion=dispersion,
        status=0,
        comment="OK",
    )
