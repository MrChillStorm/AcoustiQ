"""
acoustiq_kernels.py — Numba JIT kernels for acoustiq's hot path.

Kept in a separate, properly importable module so that Numba's disk cache
(cache=True) works correctly.  When @njit functions live inside a file that is
loaded via importlib.util.exec_module() the module name is '<dynamic>' or
similar; Numba records that name in the cache pickle and then fails to
reconstruct it in spawned worker processes with:

    ModuleNotFoundError: No module named '<dynamic>'

By placing the kernels here — a real file that Python can import by name —
Numba stores 'acoustiq_kernels' in the cache pickle, which every worker can
resolve normally.  The first run compiles and writes .nbi/.nbc files next to
this script; all subsequent processes (workers, reruns) load from disk in
milliseconds instead of recompiling.
"""

import math
import numpy as np
import numba
from numba import njit


# ─────────────────────────────────────────────────────────────────────────────
# Biquad coefficient kernels
# ─────────────────────────────────────────────────────────────────────────────

@njit(fastmath=True, cache=True)
def _peaking_nb(fc, Q, gain_db, fs):
    A      = 10.0 ** (gain_db / 40.0)
    w0     = 2.0 * math.pi * fc / fs
    sin_w0 = math.sin(w0)
    cos_w0 = math.cos(w0)
    alpha  = sin_w0 / (2.0 * Q)
    a0     = 1.0 + alpha / A
    b0     = (1.0 + alpha * A) / a0
    b1     = (-2.0 * cos_w0)   / a0
    b2     = (1.0 - alpha * A) / a0
    a1     = (-2.0 * cos_w0)   / a0
    a2     = (1.0 - alpha / A) / a0
    return b0, b1, b2, 1.0, a1, a2


@njit(fastmath=True, cache=True)
def _low_shelf_nb(fc, Q, gain_db, fs):
    A      = 10.0 ** (gain_db / 40.0)
    w0     = 2.0 * math.pi * fc / fs
    sin_w0 = math.sin(w0)
    cos_w0 = math.cos(w0)
    alpha  = sin_w0 / (2.0 * Q)
    sqrtA  = math.sqrt(A)
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha
    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0


@njit(fastmath=True, cache=True)
def _high_shelf_nb(fc, Q, gain_db, fs):
    A      = 10.0 ** (gain_db / 40.0)
    w0     = 2.0 * math.pi * fc / fs
    sin_w0 = math.sin(w0)
    cos_w0 = math.cos(w0)
    alpha  = sin_w0 / (2.0 * Q)
    sqrtA  = math.sqrt(A)
    b0 =  A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 =  A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha
    return b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0


# ─────────────────────────────────────────────────────────────────────────────
# Full-stack frequency response kernels
# ─────────────────────────────────────────────────────────────────────────────

@njit(fastmath=True, cache=True)
def _total_response_nb(fcs, Qs, gs, states, z1, z2,
                       n_filters, use_high_shelf, use_low_shelf, fs):
    """Full filter-stack frequency response — entire hot loop in one JIT kernel.

    Parameters
    ----------
    fcs, Qs, gs    : float64[n_filters]  — filter parameters
    states         : bool[n_filters]     — True = ON, False = skip
    z1, z2         : complex128[n_freq]  — precomputed z^-1, z^-2
    use_high_shelf : bool — last filter is a high shelf
    use_low_shelf  : bool — first filter is a low shelf
    fs             : float64 — sample rate

    Returns
    -------
    H : complex128[n_freq]
    """
    n_freq = z1.shape[0]
    H = np.ones(n_freq, dtype=numba.complex128)

    for i in range(n_filters):
        if not states[i]:
            continue

        fc      = fcs[i]
        Q       = Qs[i]
        gain_db = gs[i]

        if use_low_shelf and i == 0:
            b0, b1, b2, a0, a1, a2 = _low_shelf_nb(fc, Q, gain_db, fs)
        elif use_high_shelf and i == n_filters - 1:
            b0, b1, b2, a0, a1, a2 = _high_shelf_nb(fc, Q, gain_db, fs)
        else:
            b0, b1, b2, a0, a1, a2 = _peaking_nb(fc, Q, gain_db, fs)

        for k in range(n_freq):
            z1k = z1[k]
            z2k = z2[k]
            num = b0 + b1 * z1k + b2 * z2k
            den = a0 + a1 * z1k + a2 * z2k
            H[k] *= num / den

    return H


@njit(fastmath=True, cache=True)
def _static_response_nb(fcs, Qs, gs, ftypes, z1, z2, n_static, fs):
    """Static (pinned) filter stack — same structure, ftype encoded as int.

    ftype encoding: 0 = PK, 1 = HS, 2 = LS
    """
    n_freq = z1.shape[0]
    H = np.ones(n_freq, dtype=numba.complex128)

    for i in range(n_static):
        fc      = fcs[i]
        Q       = Qs[i]
        gain_db = gs[i]
        ft      = ftypes[i]

        if ft == 1:
            b0, b1, b2, a0, a1, a2 = _high_shelf_nb(fc, Q, gain_db, fs)
        elif ft == 2:
            b0, b1, b2, a0, a1, a2 = _low_shelf_nb(fc, Q, gain_db, fs)
        else:
            b0, b1, b2, a0, a1, a2 = _peaking_nb(fc, Q, gain_db, fs)

        for k in range(n_freq):
            z1k = z1[k]
            z2k = z2[k]
            num = b0 + b1 * z1k + b2 * z2k
            den = a0 + a1 * z1k + a2 * z2k
            H[k] *= num / den

    return H


# ─────────────────────────────────────────────────────────────────────────────
# Warm-up — called once at import time.
# With cache=True this compiles on the very first import and writes .nbi/.nbc
# files next to this script.  Every subsequent import (including all worker
# processes) loads the compiled bytecode from disk in <100 ms.
# ─────────────────────────────────────────────────────────────────────────────

def _warmup():
    _z   = np.exp(-1j * 2.0 * np.pi
                  * np.array([100.0, 1000.0, 5000.0, 10000.0]) / 48000.0)
    _z1  = np.ascontiguousarray(_z ** -1)
    _z2  = np.ascontiguousarray(_z ** -2)
    _fcs = np.array([200.0, 1000.0, 5000.0], dtype=np.float64)
    _Qs  = np.array([0.707, 0.707,  0.5],    dtype=np.float64)
    _gs  = np.array([3.0,  -3.0,    1.0],    dtype=np.float64)
    _st  = np.ones(3, dtype=np.bool_)
    _total_response_nb(_fcs, _Qs, _gs, _st, _z1, _z2, 3, True, True, 48000.0)
    _ftypes = np.array([0, 1, 2], dtype=np.int64)
    _static_response_nb(_fcs, _Qs, _gs, _ftypes, _z1, _z2, 3, 48000.0)

_warmup()
