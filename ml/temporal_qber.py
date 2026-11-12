"""
Windowing + QBER time-series generation (spec Section 4.1).

Given a sifted key of length N, T = floor(N / W) non-overlapping windows are
formed; window t covers sifted bits (t-1)W .. tW-1 and its local QBER is
    q_t = sum(alice XOR bob) / W.
Per-basis QBER sequences are computed from the rectilinear-only and the
diagonal-only bits inside every window (NaN when a window has none of that
basis, imputed later with the sequence mean).
"""
from __future__ import annotations

import numpy as np

from config import settings as S


class TemporalQBER:
    def __init__(self, window_size: int = S.WINDOW_SIZE):
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.W = int(window_size)

    @staticmethod
    def _window_qber(errors: np.ndarray, W: int) -> np.ndarray:
        T = len(errors) // W
        if T == 0:
            return np.zeros(0)
        return errors[: T * W].reshape(T, W).mean(axis=1)

    def compute(self, alice_sifted: np.ndarray, bob_sifted: np.ndarray,
                sifted_bases: np.ndarray, matching_indices: np.ndarray | None = None,
                lost_mask: np.ndarray | None = None) -> dict:
        """
        Args:
            alice_sifted, bob_sifted: sifted key bits (1-D arrays)
            sifted_bases: basis of each sifted bit (0 = rect, 1 = diag)
            matching_indices / lost_mask (optional): raw-qubit bookkeeping from
                the simulator, used to fill 'loss_per_window'.
        """
        a = np.asarray(alice_sifted, dtype=np.int8)
        b = np.asarray(bob_sifted, dtype=np.int8)
        bases = np.asarray(sifted_bases, dtype=np.int8)
        if not (len(a) == len(b) == len(bases)):
            raise ValueError("alice_sifted, bob_sifted and sifted_bases must have equal length")
        W = self.W
        errors = (a != b).astype(float)
        T = len(errors) // W
        q = self._window_qber(errors, W)

        q_rect = np.full(T, np.nan)
        q_diag = np.full(T, np.nan)
        if T:
            e_w = errors[: T * W].reshape(T, W)
            b_w = bases[: T * W].reshape(T, W)
            rect = b_w == 0
            n_rect = rect.sum(axis=1)
            n_diag = W - n_rect
            with np.errstate(invalid="ignore", divide="ignore"):
                q_rect = np.where(n_rect > 0, (e_w * rect).sum(axis=1) / np.maximum(n_rect, 1), np.nan)
                q_diag = np.where(n_diag > 0, (e_w * ~rect).sum(axis=1) / np.maximum(n_diag, 1), np.nan)

        loss_per_window = np.zeros(T)
        if matching_indices is not None and lost_mask is not None and T:
            loss_per_window = self.loss_per_window(np.asarray(matching_indices),
                                                   np.asarray(lost_mask, dtype=bool), W, T)

        return {
            "qber_sequence": q,
            "qber_rect_sequence": q_rect,
            "qber_diag_sequence": q_diag,
            "n_windows": int(T),
            "error_positions": np.flatnonzero(errors),
            "loss_per_window": loss_per_window,
            "window_size": W,
            # kept so that the feature engine can recompute other scales
            "alice_sifted": a,
            "bob_sifted": b,
            "sifted_bases": bases,
        }

    @staticmethod
    def loss_per_window(matching_indices: np.ndarray, lost_mask: np.ndarray,
                        W: int, T: int) -> np.ndarray:
        """Fraction of raw photons lost inside the raw-index span covered by each
        sifted window."""
        out = np.zeros(T)
        if len(matching_indices) < T * W:
            return out
        cum_lost = np.concatenate([[0], np.cumsum(lost_mask)])
        for t in range(T):
            lo = matching_indices[t * W]
            hi = matching_indices[(t + 1) * W - 1] + 1
            out[t] = (cum_lost[hi] - cum_lost[lo]) / max(hi - lo, 1)
        return out
