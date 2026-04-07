from __future__ import annotations

import sys
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
    # 复制一份输入信号，避免原始对象被就地修改
    sig = Signal(np.asarray(signal.data, dtype=float).copy(), float(signal.fs), dict(signal.meta))
    # 与 MATLAB/现有脚本保持一致：先做反相，再去直流
    sig.data = -sig.data
    sig = f_remove_dc(sig, plot=False)

    # 计算移动窗能量，并基于能量信号做脉冲片段检测
    sig_energy = f_moving_energy(sig, energy_win_sec, method="mean", plot=False)
    sig_frags = f_detect_pulses(
        sig,
        ref_sig=sig_energy,
        method="mean_std",
        threshold=detect_threshold,
        plot=False,
        clilog=False,
        min_width=0.8
    )

    n = len(sig_frags)
    # 分别记录每个片段拟合得到的 scan/sync 时间（无效值先用 NaN 占位）
    t_scan_all = np.full(n, np.nan, dtype=float)
    t_sync_all = np.full(n, np.nan, dtype=float)
    type_all: List[str] = ["" for _ in range(n)]

    for i, sig_frag in enumerate(sig_frags):
        # 片段时间需加上原始信号中的起始偏移，得到全局时间戳
        t_offset = float(sig_frag.meta.get("t_start_s", 0.0))
        # 片段级 DWT + 能量特征，用于分类该片段类型
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
            # scan 脉冲：使用高斯模型拟合中心位置 mu
            try:
                fit_paras = f_gauss_fit(sig_frag, algorithm="lsqcurvefit", plot=False)
                t_scan_all[i] = float(fit_paras["mu"]) + t_offset
            except Exception as ex:
                if verbose:
                    print(f"warning: fragment {i + 1} (scan) fit failed: {ex}")
        elif sig_frag_type == "sync":
            # sync 脉冲：使用同步脉冲模型拟合起点 t0
            try:
                fit_paras = f_fit_sync_pulse(sig_frag, vbase0=0.0, plot=False)
                t_sync_all[i] = float(fit_paras["t0"]) + t_offset
            except Exception as ex:
                if verbose:
                    print(f"warning: fragment {i + 1} (sync) fit failed: {ex}")

    # 按分类结果拆分两类时间序列
    type_arr = np.asarray(type_all, dtype=object)
    t_scan = t_scan_all[type_arr == "scan"]
    t_sync = t_sync_all[type_arr == "sync"]

    # 对两类脉冲分别做周期/相位分组
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

    # 将 scan 组与 sync 组进行匹配，得到 station 级结果
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
