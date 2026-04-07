from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dsp_functions import (
    Signal,
    f_classify_signal_type,
    f_detect_pulses,
    f_dwt_energy,
    f_fit_sync_pulse,
    f_gauss_fit,
    f_moving_energy,
    f_pulse_grouping,
    f_pulse_matching,
    f_remove_dc,
    f_signal_dwt,
)


def _prepare_signal(signal: Signal) -> Signal:
    """Prepare input signal with the same pre-processing as the classic pipeline."""
    sig = Signal(np.asarray(signal.data, dtype=float).copy(), float(signal.fs), dict(signal.meta))
    sig.data = -sig.data
    return f_remove_dc(sig, plot=False)


def _process_one_fragment(
    idx: int,
    sig_frag: Signal,
    *,
    dwt_wavelet: str,
    dwt_level: int,
    plot_dwt_energy: bool,
    verbose: bool,
) -> Tuple[int, str, float, float]:
    """
    Process one fragment: DWT -> type classification -> fit.
    Return (index, fragment_type, t_scan_global, t_sync_global), using NaN for invalid values.
    """
    t_offset = float(sig_frag.meta.get("t_start_s", 0.0))
    t_scan = float("nan")
    t_sync = float("nan")

    sig_frag_dwt = f_signal_dwt(sig_frag, wavelet=dwt_wavelet, level=dwt_level, plot=False)
    sig_frag_dwt_eng = f_dwt_energy(
        sig_frag_dwt,
        mode="peak_window",
        half_win=5,
        plot=plot_dwt_energy,
        normalize=False,
    )
    sig_frag_type = f_classify_signal_type(sig_frag_dwt_eng)

    if sig_frag_type == "scan":
        try:
            fit_paras = f_gauss_fit(sig_frag, algorithm="lsqcurvefit", plot=False)
            t_scan = float(fit_paras["mu"]) + t_offset
        except Exception as ex:
            if verbose:
                print(f"warning: fragment {idx + 1} (scan) fit failed: {ex}")
    elif sig_frag_type == "sync":
        try:
            fit_paras = f_fit_sync_pulse(sig_frag, vbase0=0.0, plot=False)
            t_sync = float(fit_paras["t0"]) + t_offset
        except Exception as ex:
            if verbose:
                print(f"warning: fragment {idx + 1} (sync) fit failed: {ex}")

    return idx, sig_frag_type, t_scan, t_sync


def _finalize_result(
    signal: Signal,
    *,
    n: int,
    type_all: List[str],
    t_scan_all: np.ndarray,
    t_sync_all: np.ndarray,
    verbose: bool,
) -> Dict[str, Any]:
    """Post-process typed timestamps into grouped/matched station results."""
    type_arr = np.asarray(type_all, dtype=object)
    t_scan = t_scan_all[type_arr == "scan"]
    t_sync = t_sync_all[type_arr == "sync"]

    t_scan_group = f_pulse_grouping(
        t_scan,
        period_tol=0.0005,
        phase_tol=0.01,
        rpm_min=1500,
        rpm_max=3000,
        min_count=4,
        plot=False,
        verbose=False,
    )
    t_sync_group = f_pulse_grouping(
        t_sync,
        period_tol=0.0005,
        phase_tol=0.01,
        rpm_min=1500,
        rpm_max=3000,
        min_count=4,
        plot=False,
        verbose=verbose,
    )
    stations = f_pulse_matching(t_scan_group, t_sync_group, period_tol=0.005, verbose=verbose)

    return {
        "source": signal.meta.get("source", "<in-memory>"),
        "n_fragments": n,
        "t_scan": t_scan,
        "t_sync": t_sync,
        "scan_groups": t_scan_group,
        "sync_groups": t_sync_group,
        "stations": stations,
    }


def process_one_classic(
    signal: Signal,
    *,
    energy_win_sec: float = 1e-6,
    detect_threshold: float = 5.0,
    dwt_level: int = 5,
    dwt_wavelet: str = "db4",
    plot_dwt_energy: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Classic processing pipeline equivalent to process_signal.process_one_file,
    but accepts an in-memory Signal object directly.
    """
    sig = _prepare_signal(signal)
    sig_energy = f_moving_energy(sig, energy_win_sec, method="mean", plot=False)
    sig_frags = f_detect_pulses(
        sig,
        ref_sig=sig_energy,
        method="mean_std",
        threshold=detect_threshold,
        plot=False,
        clilog=False,
        min_width=0.8,
    )

    n = len(sig_frags)
    t_scan_all = np.full(n, np.nan, dtype=float)
    t_sync_all = np.full(n, np.nan, dtype=float)
    type_all: List[str] = ["" for _ in range(n)]

    for i, sig_frag in enumerate(sig_frags):
        idx, frag_type, t_scan, t_sync = _process_one_fragment(
            i,
            sig_frag,
            dwt_wavelet=dwt_wavelet,
            dwt_level=dwt_level,
            plot_dwt_energy=plot_dwt_energy,
            verbose=verbose,
        )
        type_all[idx] = frag_type
        t_scan_all[idx] = t_scan
        t_sync_all[idx] = t_sync

    return _finalize_result(
        signal,
        n=n,
        type_all=type_all,
        t_scan_all=t_scan_all,
        t_sync_all=t_sync_all,
        verbose=verbose,
    )


def process_one_classic_parallel(
    signal: Signal,
    *,
    energy_win_sec: float = 1e-6,
    detect_threshold: float = 5.0,
    dwt_level: int = 5,
    dwt_wavelet: str = "db4",
    plot_dwt_energy: bool = False,
    verbose: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Parallel variant of process_one_classic.
    Uses a thread pool to process each detected fragment independently.
    """
    sig = _prepare_signal(signal)
    sig_energy = f_moving_energy(sig, energy_win_sec, method="mean", plot=False)
    sig_frags = f_detect_pulses(
        sig,
        ref_sig=sig_energy,
        method="mean_std",
        threshold=detect_threshold,
        plot=False,
        clilog=False,
        min_width=0.8,
    )

    n = len(sig_frags)
    t_scan_all = np.full(n, np.nan, dtype=float)
    t_sync_all = np.full(n, np.nan, dtype=float)
    type_all: List[str] = ["" for _ in range(n)]

    if n == 0:
        return _finalize_result(
            signal,
            n=n,
            type_all=type_all,
            t_scan_all=t_scan_all,
            t_sync_all=t_sync_all,
            verbose=verbose,
        )

    # Matplotlib calls are not thread-safe, so keep energy plotting in serial mode.
    if plot_dwt_energy:
        if verbose:
            print("warning: plot_dwt_energy=True; fallback to serial fragment processing.")
        for i, sig_frag in enumerate(sig_frags):
            idx, frag_type, t_scan, t_sync = _process_one_fragment(
                i,
                sig_frag,
                dwt_wavelet=dwt_wavelet,
                dwt_level=dwt_level,
                plot_dwt_energy=True,
                verbose=verbose,
            )
            type_all[idx] = frag_type
            t_scan_all[idx] = t_scan
            t_sync_all[idx] = t_sync
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _process_one_fragment,
                    i,
                    sig_frag,
                    dwt_wavelet=dwt_wavelet,
                    dwt_level=dwt_level,
                    plot_dwt_energy=False,
                    verbose=verbose,
                )
                for i, sig_frag in enumerate(sig_frags)
            ]
            for fut in concurrent.futures.as_completed(futures):
                idx, frag_type, t_scan, t_sync = fut.result()
                type_all[idx] = frag_type
                t_scan_all[idx] = t_scan
                t_sync_all[idx] = t_sync

    return _finalize_result(
        signal,
        n=n,
        type_all=type_all,
        t_scan_all=t_scan_all,
        t_sync_all=t_sync_all,
        verbose=verbose,
    )
