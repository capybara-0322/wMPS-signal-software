from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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


def build_path(pattern: str, filenum: int) -> Path:
    if "%" in pattern:
        return Path(pattern % filenum)
    return Path(pattern.format(num=filenum))


def process_one_file(
    file_path: Path,
    channel: str = "ch1",
    energy_win_sec: float = 1e-6,
    detect_threshold: float = 5.0,
    dwt_level: int = 5,
    dwt_wavelet: str = "db4",
    plot_dwt_energy: bool = False,
) -> Dict[str, Any]:
    sig = Signal.from_file(file_path, channel=channel)
    sig.data = -sig.data
    sig = f_remove_dc(sig, plot=False)

    sig_energy = f_moving_energy(sig, energy_win_sec, method="mean", plot=False)
    sig_frags = f_detect_pulses(
        sig,
        ref_sig=sig_energy,
        method="mean_std",
        threshold=detect_threshold,
        plot=False,
        clilog=False,
    )

    n = len(sig_frags)
    t_scan_all = np.full(n, np.nan, dtype=float)
    t_sync_all = np.full(n, np.nan, dtype=float)
    type_all: List[str] = ["" for _ in range(n)]

    for i, sig_frag in enumerate(sig_frags):
        t_offset = float(sig_frag.meta.get("t_start_s", 0.0))
        sig_frag_dwt = f_signal_dwt(sig_frag, wavelet=dwt_wavelet, level=dwt_level, plot=False)
        sig_frag_dwt_eng = f_dwt_energy(
            sig_frag_dwt,
            mode="peak_window",
            half_win=5,
            plot=plot_dwt_energy,
            normalize=False,
        )
        sig_frag_type = f_classify_signal_type(sig_frag_dwt_eng)
        type_all[i] = sig_frag_type

        if sig_frag_type == "scan":
            try:
                fit_paras = f_gauss_fit(sig_frag, algorithm="lsqcurvefit", plot=False)
                t_scan_all[i] = float(fit_paras["mu"]) + t_offset
            except Exception as ex:
                print(f"warning: fragment {i+1} (scan) fit failed: {ex}")
        elif sig_frag_type == "sync":
            try:
                fit_paras = f_fit_sync_pulse(sig_frag, vbase0=0.0, plot=False)
                t_sync_all[i] = float(fit_paras["t0"]) + t_offset
            except Exception as ex:
                print(f"warning: fragment {i+1} (sync) fit failed: {ex}")

    type_arr = np.asarray(type_all, dtype=object)
    t_scan = t_scan_all[type_arr == "scan"]
    t_sync = t_sync_all[type_arr == "sync"]

    t_scan_group = f_pulse_grouping(
        t_scan,
        period_tol=0.0005,
        phase_tol=0.01,
        rpm_min=1500,
        rpm_max=3000,
        plot=False,
        verbose=False,
    )
    t_sync_group = f_pulse_grouping(
        t_sync,
        period_tol=0.0005,
        phase_tol=0.01,
        rpm_min=1500,
        rpm_max=3000,
        plot=False,
        verbose=False,
    )

    stations = f_pulse_matching(t_scan_group, t_sync_group, period_tol=0.0005, verbose=False)

    return {
        "file": str(file_path),
        "n_fragments": n,
        "t_scan": t_scan,
        "t_sync": t_sync,
        "scan_groups": t_scan_group,
        "sync_groups": t_sync_group,
        "stations": stations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Process signal files with logic equivalent to dsp_functions_matlab/test.m")
    parser.add_argument("--files", nargs="+", type=int, default=[7], help="File numbers to process, e.g. --files 7 8")
    parser.add_argument(
        "--pattern",
        type=str,
        default="./04-04overlap_python/00%02d.npz",
        help="Input path pattern, supports printf style (e.g. ./dir/00%02d.npz) or format style (e.g. ./dir/{num:04d}.npz)",
    )
    parser.add_argument("--channel", type=str, default="ch1")
    parser.add_argument("--plot-dwt-energy", action="store_true")
    args = parser.parse_args()

    t0 = time.time()

    for filenum in args.files:
        file_path = build_path(args.pattern, filenum)
        if not file_path.exists():
            print(f"skip: file not found: {file_path}")
            continue

        print(f"processing: {file_path}")
        try:
            result = process_one_file(
                file_path=file_path,
                channel=args.channel,
                plot_dwt_energy=args.plot_dwt_energy,
            )
        except ImportError as ex:
            print(f"error: {ex}")
            print("install suggestion: conda install -n dsp pywavelets")
            break

        t_scan = result["t_scan"]
        t_sync = result["t_sync"]
        valid_sync = int(np.sum(np.isfinite(t_sync)))
        valid_scan = int(np.sum(np.isfinite(t_scan)))

        print(
            f"sync: {valid_sync} valid / {t_sync.size} fragments; "
            f"scan: {valid_scan} valid / {t_scan.size} fragments"
        )
        print(
            f"groups(scan/sync): {result['scan_groups']['summary'].shape[0]}/{result['sync_groups']['summary'].shape[0]}; "
            f"stations matched: {len(result['stations'])}"
        )

    print(f"done in {time.time() - t0:.3f}s")


if __name__ == "__main__":
    main()
