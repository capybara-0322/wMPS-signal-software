from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Signal:
    """One-dimensional sampled signal with sampling rate metadata."""

    data: np.ndarray
    fs: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("Signal data must not be empty")
        if self.fs <= 0:
            raise ValueError("fs must be positive")
        self.data = arr
        self.fs = float(self.fs)
        self.meta = dict(self.meta or {})

    @property
    def N(self) -> int:
        return int(self.data.size)

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def duration(self) -> float:
        return self.N / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.N, dtype=float) / self.fs

    @classmethod
    def from_file(cls, filepath: str | Path, channel: str = "ch1") -> "Signal":
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(str(path))

        if path.suffix.lower() == ".npz":
            raw = _load_npz(path)
        else:
            raw = _load_mat(path)

        if "fs" not in raw:
            raise KeyError("missing key: fs")
        fs = float(np.asarray(raw["fs"]).reshape(-1)[0])

        q_field = f"q_{channel}"
        if q_field in raw:
            q_data = np.asarray(raw[q_field], dtype=float).reshape(-1)
        elif "q_values" in raw:
            q_data = np.asarray(raw["q_values"], dtype=float).reshape(-1)
        elif "q_ch1" in raw:
            q_data = np.asarray(raw["q_ch1"], dtype=float).reshape(-1)
        else:
            raise KeyError(f"missing channel key: {q_field}")

        scale = float(np.asarray(raw.get("scale", 1.0)).reshape(-1)[0])
        quant_mode = str(_scalar(raw.get("quant_mode", ""))).strip()

        if quant_mode and quant_mode != "int16_symmetric_v1":
            pass

        data = q_data * scale

        meta: Dict[str, Any] = {"source": str(path), "channel": channel}
        if "meta_json" in raw:
            try:
                meta["device"] = json.loads(str(_scalar(raw["meta_json"])))
            except Exception:
                pass

        return cls(data=data, fs=fs, meta=meta)

    def plot(self, max_points: int = 10_000_000, title: str = "") -> None:
        import matplotlib.pyplot as plt

        t_plot, y_plot, note = _downsample_for_plot(self.t, self.data, max_points)
        fig_title = title or f"fs={self.fs/1e6:.3g} MHz N={self.N}{note}"
        plt.figure()
        plt.plot(t_plot * 1e3, y_plot, linewidth=0.8)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (V)")
        plt.title(fig_title)
        plt.grid(True)
        plt.tight_layout()

    def psd(self) -> None:
        import matplotlib.pyplot as plt
        from scipy.signal import welch

        f, pxx = welch(self.data, fs=self.fs)
        plt.figure()
        plt.semilogy(f, pxx + np.finfo(float).eps)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.title(f"PSD fs={self.fs/1e6:.3g} MHz")
        plt.grid(True)
        plt.tight_layout()

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.data**2)))

    def crop(self, t1: float, t2: float) -> "Signal":
        if t1 < 0 or t2 <= t1:
            raise ValueError("require 0 <= t1 < t2")
        i1 = max(0, int(round(t1 * self.fs)))
        i2 = min(self.N, int(round(t2 * self.fs)) + 1)
        return Signal(self.data[i1:i2], self.fs, self.meta)

    def resample(self, fs_new: float) -> "Signal":
        from fractions import Fraction
        from scipy.signal import resample_poly

        if fs_new <= 0:
            raise ValueError("fs_new must be positive")
        frac = Fraction(fs_new / self.fs).limit_denominator(1_000_000)
        new_data = resample_poly(self.data, frac.numerator, frac.denominator)
        return Signal(new_data, float(fs_new), self.meta)


def f_remove_dc(sig: Signal, plot: bool = False) -> Signal:
    dc = float(np.mean(sig.data))
    data_out = sig.data - dc

    meta = dict(sig.meta)
    meta["dc_removed"] = dc
    meta["parent_source"] = sig.meta.get("source", "")
    meta["source"] = "f_remove_dc"

    out = Signal(data_out, sig.fs, meta)

    if plot:
        import matplotlib.pyplot as plt

        t_ms = sig.t * 1e3
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(t_ms, sig.data, linewidth=0.8, color="tab:blue", label="raw")
        ax2.plot(t_ms, data_out, linewidth=0.8, color="tab:red", label="dc_removed")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Raw (V)")
        ax2.set_ylabel("DC removed (V)")
        plt.title(f"DC removed: {dc:.6g} V")
        plt.tight_layout()

    return out


def f_moving_energy(sig: Signal, win_sec: float, method: str = "mean", plot: bool = False) -> Signal:
    if win_sec <= 0:
        raise ValueError("win_sec must be positive")

    method = method.lower()
    if method not in {"mean", "rms", "sum"}:
        raise ValueError("method must be one of: mean, rms, sum")

    win_pts = int(round(win_sec * sig.fs))
    if win_pts < 1:
        raise ValueError("window too short")
    win_pts = min(win_pts, sig.N)

    x2 = sig.data**2
    half = win_pts // 2
    x2_pad = np.concatenate([np.zeros(half), x2, np.zeros(half)])
    cs = np.concatenate([[0.0], np.cumsum(x2_pad)])
    win_sum = cs[win_pts : win_pts + sig.N] - cs[: sig.N]

    if method == "mean":
        energy = win_sum / win_pts
        unit = "V^2"
    elif method == "rms":
        energy = np.sqrt(win_sum / win_pts)
        unit = "V"
    else:
        energy = win_sum
        unit = "V^2*pts"

    meta = {
        "source": "f_moving_energy",
        "parent_source": sig.meta.get("source", ""),
        "method": method,
        "unit": unit,
        "win_sec": float(win_sec),
        "win_pts": int(win_pts),
    }
    out = Signal(energy, sig.fs, meta)

    if plot:
        import matplotlib.pyplot as plt

        t_ms = sig.t * 1e3
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(t_ms, sig.data, linewidth=0.8, color="tab:blue")
        ax2.plot(t_ms, energy, linewidth=0.8, color="tab:red")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Signal (V)")
        ax2.set_ylabel(f"{method} ({unit})")
        plt.title("Moving window energy")
        plt.tight_layout()

    return out


def f_signal_dwt(sig: Signal, wavelet: str = "db4", level: int = 0, plot: bool = False) -> Dict[str, Any]:
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "f_signal_dwt requires PyWavelets. Install it with: conda install -n dsp pywavelets"
        ) from exc

    x = sig.data
    fs = sig.fs

    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    if level == 0:
        L = max_level
    else:
        L = int(round(level))
        if L < 1:
            raise ValueError("level must be >=1 or 0 for auto")
        L = min(L, max_level)

    coeffs = pywt.wavedec(x, wavelet, level=L)

    cA: List[np.ndarray] = []
    cD: List[np.ndarray] = []
    t_k: List[np.ndarray] = []
    freq_band: List[np.ndarray] = []
    fs_k = np.zeros(L, dtype=float)

    for k in range(1, L + 1):
        cD_k = np.asarray(coeffs[-k], dtype=float)
        cA_k = np.asarray(pywt.downcoef("a", x, wavelet, level=k), dtype=float)
        fs_level = fs / (2**k)
        t_level = np.arange(cD_k.size, dtype=float) / fs_level
        f_lo = fs / (2 ** (k + 1))
        f_hi = fs / (2**k)

        cA.append(cA_k)
        cD.append(cD_k)
        fs_k[k - 1] = fs_level
        t_k.append(t_level)
        freq_band.append(np.array([f_lo, f_hi], dtype=float))

    result = {
        "wavelet": wavelet,
        "level": L,
        "fs": fs,
        "cA": cA,
        "cD": cD,
        "fs_k": fs_k,
        "t_k": t_k,
        "freq_band": freq_band,
    }

    if plot:
        _plot_dwt(result, sig)

    return result


def f_dwt_energy(
    dwt_result: Dict[str, Any],
    mode: str = "full",
    half_win: int = 500,
    thresh_n: float = 3.0,
    win_time: float = 100e-6,
    ref_fs: float = 50e6,
    normalize: bool = False,
    plot: bool = False,
) -> Dict[str, Any]:
    required = ["level", "fs", "cA", "cD", "fs_k", "t_k", "freq_band"]
    for key in required:
        if key not in dwt_result:
            raise KeyError(f"missing key: {key}")

    mode = mode.lower()
    if mode not in {"full", "peak_window", "threshold", "time_window"}:
        raise ValueError("unsupported mode")

    L = int(dwt_result["level"])
    fs_ori = float(dwt_result["fs"])

    result: Dict[str, Any] = {
        "mode": mode,
        "normalize": bool(normalize),
        "ref_fs": float(ref_fs),
        "fc": np.zeros(L, dtype=float),
        "energy_cD": np.zeros(L, dtype=float),
        "win_idx_cD": [],
        "win_time_cD": [],
        "thresh_fallback_cD": np.zeros(L, dtype=bool),
    }

    for k in range(1, L + 1):
        coef = np.asarray(dwt_result["cD"][k - 1], dtype=float).reshape(-1)
        t_k = np.asarray(dwt_result["t_k"][k - 1], dtype=float).reshape(-1)
        fs_k = float(dwt_result["fs_k"][k - 1])
        fband = np.asarray(dwt_result["freq_band"][k - 1], dtype=float)

        if fband[0] <= 0:
            fc = fband[1] / 2.0
        else:
            fc = math.sqrt(fband[0] * fband[1])

        win_idx, win_time, energy, fallback = _process_layer(
            coef,
            t_k,
            fs_k,
            mode=mode,
            half_win=half_win,
            thresh_n=thresh_n,
            win_time=win_time,
            ref_fs=ref_fs,
            normalize=normalize,
        )

        result["fc"][k - 1] = fc
        result["energy_cD"][k - 1] = energy
        result["win_idx_cD"].append(win_idx)
        result["win_time_cD"].append(win_time)
        result["thresh_fallback_cD"][k - 1] = fallback

    coef_a = np.asarray(dwt_result["cA"][L - 1], dtype=float).reshape(-1)
    t_a = np.asarray(dwt_result["t_k"][L - 1], dtype=float).reshape(-1)
    fs_a = float(dwt_result["fs_k"][L - 1])

    win_idx_a, win_time_a, energy_a, fallback_a = _process_layer(
        coef_a,
        t_a,
        fs_a,
        mode=mode,
        half_win=half_win,
        thresh_n=thresh_n,
        win_time=win_time,
        ref_fs=ref_fs,
        normalize=normalize,
    )

    result["fc_approx"] = fs_ori / (2 ** (L + 2))
    result["win_idx_cA"] = win_idx_a
    result["win_time_cA"] = win_time_a
    result["energy_cA"] = energy_a
    result["thresh_fallback_cA"] = fallback_a

    if plot:
        _plot_energy(dwt_result, result)

    return result


def f_dwt_energy_new(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return f_dwt_energy(*args, **kwargs)


def f_gauss_fit(
    sig: Signal,
    algorithm: str = "lsqcurvefit",
    x0: Optional[Sequence[float]] = None,
    baseline: bool = False,
    plot: bool = False,
) -> Dict[str, Any]:
    from scipy.optimize import least_squares, minimize

    t_raw = sig.t
    y_raw = sig.data

    n_param = 4 if baseline else 3
    t_offset = float(np.mean(t_raw))
    t_scale = float(np.max(np.abs(t_raw - t_offset)))
    y_scale = float(np.max(np.abs(y_raw)))
    if t_scale <= np.finfo(float).eps:
        raise ValueError("time span too small for fitting")
    if y_scale <= np.finfo(float).eps:
        raise ValueError("signal magnitude too small for fitting")

    t_norm = (t_raw - t_offset) / t_scale
    y_norm = y_raw / y_scale

    if x0 is None:
        x0_norm = np.array([1.0, 0.0, 0.1, 0.0] if baseline else [1.0, 0.0, 0.1], dtype=float)
    else:
        x0_arr = np.asarray(x0, dtype=float).reshape(-1)
        if x0_arr.size != n_param:
            raise ValueError(f"x0 length must be {n_param}")
        arr = np.zeros(n_param, dtype=float)
        arr[0] = x0_arr[0] / y_scale
        arr[1] = (x0_arr[1] - t_offset) / t_scale
        arr[2] = x0_arr[2] / t_scale
        if baseline:
            arr[3] = x0_arr[3] / y_scale
        x0_norm = arr

    def gauss_fn(p: np.ndarray, t: np.ndarray) -> np.ndarray:
        if baseline:
            return p[0] * np.exp(-((t - p[1]) ** 2) / (2.0 * (p[2] ** 2))) + p[3]
        return p[0] * np.exp(-((t - p[1]) ** 2) / (2.0 * (p[2] ** 2)))

    def residual_fn(p: np.ndarray) -> np.ndarray:
        p2 = p.copy()
        p2[2] = abs(p2[2])
        return gauss_fn(p2, t_norm) - y_norm

    algo = algorithm.lower()
    converged = False
    p_fit = x0_norm.copy()

    if algo in {"lsqcurvefit", "fit"}:
        sol = least_squares(
            residual_fn,
            x0_norm,
            bounds=(np.array([-np.inf, -np.inf, 1e-9] + ([-np.inf] if baseline else [])), np.inf),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=5000,
        )
        p_fit = sol.x
        converged = bool(sol.success)
        algo = "lsqcurvefit" if algorithm.lower() == "lsqcurvefit" else "fit"
    else:
        sol = minimize(lambda p: float(np.sum(residual_fn(p) ** 2)), x0_norm, method="Nelder-Mead")
        p_fit = sol.x
        converged = bool(sol.success)
        algo = "fminsearch"

    p_fit[2] = abs(p_fit[2])
    A_fit = p_fit[0] * y_scale
    mu_fit = p_fit[1] * t_scale + t_offset
    sigma_fit = p_fit[2] * t_scale
    C_fit = p_fit[3] * y_scale if baseline else 0.0

    if baseline:
        y_hat = A_fit * np.exp(-((t_raw - mu_fit) ** 2) / (2.0 * sigma_fit**2)) + C_fit
    else:
        y_hat = A_fit * np.exp(-((t_raw - mu_fit) ** 2) / (2.0 * sigma_fit**2))

    residuals = y_raw - y_hat
    rmse = float(np.sqrt(np.mean(residuals**2)))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_raw - np.mean(y_raw)) ** 2))
    r_squared = 0.0 if ss_tot <= np.finfo(float).eps else 1.0 - ss_res / ss_tot

    result: Dict[str, Any] = {
        "A": float(A_fit),
        "mu": float(mu_fit),
        "sigma": float(sigma_fit),
        "fwhm": float(2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_fit),
        "residual": rmse,
        "r_squared": float(r_squared),
        "algorithm": algo,
        "converged": converged,
        "x_fit": t_raw,
        "y_fit": y_hat,
    }
    if baseline:
        result["C"] = float(C_fit)

    if plot:
        import matplotlib.pyplot as plt

        t_ms = t_raw * 1e3
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(t_ms, y_raw, color="0.6", linewidth=0.8, label="raw")
        axs[0].plot(t_ms, y_hat, "r-", linewidth=1.5, label="fit")
        axs[0].axvline(mu_fit * 1e3, linestyle="--", color="b")
        axs[0].legend()
        axs[0].grid(True)
        axs[1].plot(t_ms, residuals, color="tab:green", linewidth=0.8)
        axs[1].axhline(0.0, linestyle="--", color="k", linewidth=0.8)
        axs[1].set_xlabel("Time (ms)")
        axs[1].grid(True)
        plt.tight_layout()

    return result


def f_fit_sync_pulse(
    sig: Signal,
    t0_0: float = math.nan,
    pmax0: float = math.nan,
    tr0: float = 0.1e-6,
    tau_r0: float = math.nan,
    tau_f0: float = 0.5e-6,
    vbase0: float = math.nan,
    algorithm: str = "trust-region-reflective",
    plot: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    from scipy.optimize import least_squares

    t = sig.t
    v = sig.data

    n_base = min(max(round(sig.N * 0.05), 10), 100)
    if math.isnan(vbase0):
        vbase0 = float(np.mean(v[:n_base]))

    v_nb = v - vbase0
    idx_pk = int(np.argmax(v_nb))
    v_pk = float(v_nb[idx_pk])

    if math.isnan(pmax0):
        pmax0 = max(v_pk, np.finfo(float).eps)
    if math.isnan(t0_0):
        t0_0 = float(t[idx_pk] - tr0)
    if math.isnan(tau_r0):
        tau_r0 = tr0 / 3.0

    tau_r0 = min(tau_r0, tr0 * 0.9)

    p0 = np.array([t0_0, tr0, tau_r0, tau_f0, pmax0, vbase0], dtype=float)

    dt = sig.dt
    lb = np.array([t[0] - dt, dt, dt, dt, 0.0, -np.inf], dtype=float)
    ub = np.array([t[-1], 10e-6, 5e-6, 50e-6, np.inf, np.inf], dtype=float)
    p0 = np.clip(p0, lb + np.finfo(float).eps, ub - np.finfo(float).eps)

    def model(p: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
        return _pulse_model(p, t_vec)

    def residual_fn(p: np.ndarray) -> np.ndarray:
        return model(p, t) - v

    method = "trf" if algorithm == "trust-region-reflective" else "lm"
    if method == "lm":
        sol = least_squares(residual_fn, p0, method="lm", max_nfev=5000, ftol=1e-12, xtol=1e-14, gtol=1e-12)
    else:
        sol = least_squares(
            residual_fn,
            p0,
            bounds=(lb, ub),
            method="trf",
            max_nfev=5000,
            ftol=1e-12,
            xtol=1e-14,
            gtol=1e-12,
        )

    p_fit = sol.x
    residuals = sol.fun
    rms_res = float(np.sqrt(np.mean(residuals**2)))
    v_fitted = model(p_fit, t)

    result = {
        "t0": float(p_fit[0]),
        "Tr": float(p_fit[1]),
        "tau_r": float(p_fit[2]),
        "tau_f": float(p_fit[3]),
        "Pmax": float(p_fit[4]),
        "Vbase": float(p_fit[5]),
        "t_peak": float(p_fit[0] + p_fit[1]),
        "residual": rms_res,
        "exitflag": int(sol.status),
        "fit_t": t,
        "fit_y": v_fitted,
    }

    if verbose:
        print("===== Sync Pulse Fit =====")
        print(f"algorithm: {algorithm}")
        print(f"exitflag: {sol.status}")
        print(f"residual RMS: {rms_res:.4g} V")
        print(result)

    if plot:
        import matplotlib.pyplot as plt

        t_us = t * 1e6
        t_dense = np.linspace(t[0], t[-1], max(sig.N * 4, 2000))
        v_dense = model(p_fit, t_dense)

        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].plot(t_us, v, color="0.6", linewidth=0.8, label="raw")
        axs[0].plot(t_dense * 1e6, v_dense, "r-", linewidth=1.5, label="fit")
        axs[0].axvline(result["t0"] * 1e6, linestyle="--", color="b")
        axs[0].axvline(result["t_peak"] * 1e6, linestyle="--", color="g")
        axs[0].legend()
        axs[0].grid(True)
        axs[1].plot(t_us, residuals, color="tab:blue", linewidth=0.8)
        axs[1].axhline(0.0, color="k", linewidth=0.8)
        axs[1].grid(True)
        axs[2].plot(t_us, v - v_fitted, color="tab:orange", linewidth=0.8)
        axs[2].axhline(0.0, color="k", linewidth=0.8)
        axs[2].set_xlabel("Time (us)")
        axs[2].grid(True)
        plt.tight_layout()

    return result


def f_detect_pulses(
    sig: Signal,
    ref_sig: Optional[Signal] = None,
    method: str = "absolute",
    threshold: float = 0.5,
    min_width: float = 1.0,
    margin: float = 5.0,
    plot: bool = False,
    clilog: bool = False,
) -> List[Signal]:
    if ref_sig is not None:
        if not isinstance(ref_sig, Signal):
            raise TypeError("ref_sig must be Signal")
        if ref_sig.fs != sig.fs:
            raise ValueError("ref_sig.fs mismatch")
        if ref_sig.N != sig.N:
            raise ValueError("ref_sig length mismatch")
        det_sig = ref_sig
    else:
        det_sig = sig

    x = det_sig.data
    x_out = sig.data
    fs = sig.fs
    N = sig.N

    method = method.lower()
    if method == "absolute":
        thr = threshold
    elif method == "mean_std":
        thr = float(np.mean(x) + threshold * np.std(x))
    elif method == "peak_ratio":
        if not (0.0 < threshold < 1.0):
            raise ValueError("peak_ratio threshold must be in (0,1)")
        thr = float(np.max(np.abs(x)) * threshold)
    elif method == "median_mad":
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        thr = float(med + threshold * mad)
    else:
        raise ValueError("unsupported method")

    min_samples = max(1, int(round(min_width * 1e-6 * fs)))
    margin_samples = int(round(margin * 1e-6 * fs))

    above = x >= thr
    d_above = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    rise_idx = np.where(d_above == 1)[0]
    fall_idx = np.where(d_above == -1)[0] - 1

    if rise_idx.size == 0:
        return []

    widths = fall_idx - rise_idx + 1
    valid = widths >= min_samples
    rise_idx = rise_idx[valid]
    fall_idx = fall_idx[valid]

    if rise_idx.size == 0:
        return []

    i = 0
    while i < rise_idx.size - 1:
        gap = rise_idx[i + 1] - fall_idx[i] - 1
        if gap < min_samples:
            fall_idx[i] = fall_idx[i + 1]
            rise_idx = np.delete(rise_idx, i + 1)
            fall_idx = np.delete(fall_idx, i + 1)
        else:
            i += 1

    segments: List[Signal] = []
    for k in range(rise_idx.size):
        i1 = max(0, int(rise_idx[k] - margin_samples))
        i2 = min(N - 1, int(fall_idx[k] + margin_samples))

        seg_meta = dict(sig.meta)
        seg_meta["pulse_index"] = k + 1
        seg_meta["pulse_count"] = int(rise_idx.size)
        seg_meta["threshold"] = float(thr)
        seg_meta["method"] = method
        seg_meta["t_start_s"] = i1 / fs
        seg_meta["t_end_s"] = i2 / fs
        seg_meta["source"] = f"pulse_{k + 1}_of_{rise_idx.size}"
        if ref_sig is not None:
            seg_meta["ref_source"] = ref_sig.meta.get("source", "RefSig")

        segments.append(Signal(x_out[i1 : i2 + 1], fs, seg_meta))

    if plot:
        import matplotlib.pyplot as plt

        t_ms = sig.t * 1e3
        plt.figure(figsize=(10, 4))
        plt.plot(t_ms, x_out, color="0.3", linewidth=0.8, label="sig")
        if ref_sig is not None:
            plt.plot(t_ms, x, color="tab:blue", linewidth=0.8, label="ref_sig")
        plt.axhline(thr, linestyle="--", color="r", linewidth=1.0, label="threshold")
        for k in range(rise_idx.size):
            i1 = max(0, int(rise_idx[k] - margin_samples))
            i2 = min(N - 1, int(fall_idx[k] + margin_samples))
            plt.axvspan(i1 / fs * 1e3, i2 / fs * 1e3, color="tab:orange", alpha=0.15)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (V)")
        plt.title(f"Detected pulses: {rise_idx.size}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    if clilog:
        print(f"[f_detect_pulses] method={method} threshold={thr:.4g} count={len(segments)}")

    return segments


def f_classify_signal_type(
    sig_frag_dwt_eng: Dict[str, Any],
    w: Sequence[float] = (-5.06, 0.24, 1.72, 7.08, 6.96, -1.33),
    b: float = -0.488,
) -> str:
    if "energy_cD" not in sig_frag_dwt_eng or "energy_cA" not in sig_frag_dwt_eng:
        raise KeyError("require fields: energy_cD, energy_cA")

    energy_cD = np.asarray(sig_frag_dwt_eng["energy_cD"], dtype=float).reshape(-1)
    energy_cA = float(np.asarray(sig_frag_dwt_eng["energy_cA"]).reshape(-1)[0])

    if energy_cD.size != 5:
        raise ValueError("energy_cD must contain 5 levels")

    x = np.concatenate([energy_cD, np.array([energy_cA], dtype=float)])
    s = float(np.sum(x))
    if s <= 0:
        raise ValueError("sum of energies must be positive")

    x_norm = x / s
    score = float(np.dot(np.asarray(w, dtype=float).reshape(-1), x_norm) + b)
    return "sync" if score > 0 else "scan"


def f_pulse_grouping(
    pulses: Sequence[float],
    period_tol: float = 0.3,
    phase_tol: float = 0.2,
    min_count: int = 5,
    min_pulses: int = 3,
    plot: bool = False,
    verbose: bool = False,
    rpm_min: Optional[float] = None,
    rpm_max: Optional[float] = None,
) -> Dict[str, Any]:
    pulses_arr = np.asarray(pulses, dtype=float).reshape(-1)
    pulses_arr = np.sort(pulses_arr)
    n = int(pulses_arr.size)

    t_min = 60.0 / rpm_max if (rpm_max is not None and rpm_max > 0) else 0.0
    t_max = 60.0 / rpm_min if (rpm_min is not None and rpm_min > 0) else float("inf")

    if n < 2:
        out_empty = {
            "pulses": pulses_arr,
            "group_id": np.zeros(n, dtype=int),
            "periods": np.array([], dtype=float),
            "phases": np.array([], dtype=float),
            "freq_labels": np.zeros(n, dtype=int),
            "base_periods": np.array([], dtype=float),
            "summary": np.zeros((0, 4), dtype=float),
            "diffs": np.array([], dtype=float),
        }
        if plot:
            _plot_pulse_grouping(out_empty)
        return out_empty

    # Pairwise positive diffs and source indices.
    ii, jj = np.triu_indices(n, k=1)
    diffs = pulses_arr[jj] - pulses_arr[ii]

    diffs_sorted = np.sort(diffs)
    cluster_id = np.ones(diffs_sorted.size, dtype=int)
    cid = 1
    for i in range(1, diffs_sorted.size):
        if diffs_sorted[i] - diffs_sorted[i - 1] > period_tol:
            cid += 1
        cluster_id[i] = cid

    n_clusters = int(cluster_id.max())
    cluster_info = np.zeros((n_clusters, 3), dtype=float)  # [cid, count, median]
    for c in range(1, n_clusters + 1):
        m = cluster_id == c
        cluster_info[c - 1, 0] = c
        cluster_info[c - 1, 1] = np.sum(m)
        cluster_info[c - 1, 2] = float(np.median(diffs_sorted[m]))

    if cluster_info.size > 0:
        cluster_info = cluster_info[np.argsort(-cluster_info[:, 1])]
        cluster_info = cluster_info[cluster_info[:, 1] >= float(min_count)]
        in_range = (cluster_info[:, 2] >= t_min) & (cluster_info[:, 2] <= t_max)
        cluster_info = cluster_info[in_range]

    candidate_periods = cluster_info[:, 2] if cluster_info.size else np.array([], dtype=float)
    candidate_counts = cluster_info[:, 1] if cluster_info.size else np.array([], dtype=float)

    base_periods: List[float] = []
    base_counts: List[float] = []
    for k, t_now in enumerate(candidate_periods):
        is_harmonic = False
        for m in range(len(base_periods)):
            tb = base_periods[m]
            ratio = t_now / tb if tb != 0 else float("inf")
            if ratio > 0.5:
                nearest_int = round(ratio)
                if nearest_int >= 2 and abs(ratio - nearest_int) < (period_tol / tb):
                    is_harmonic = True
                    break
            ratio_inv = tb / t_now if t_now != 0 else float("inf")
            if ratio_inv >= 2 and abs(ratio_inv - round(ratio_inv)) < (period_tol / t_now):
                base_periods[m] = float(t_now)
                base_counts[m] = float(max(base_counts[m], candidate_counts[k]))
                is_harmonic = True
                break
        if not is_harmonic:
            base_periods.append(float(t_now))
            base_counts.append(float(candidate_counts[k]))

    if base_periods:
        order = np.argsort(-np.asarray(base_counts, dtype=float))
        base_periods_arr = np.asarray(base_periods, dtype=float)[order]
    else:
        base_periods_arr = np.array([], dtype=float)

    freq_labels = np.zeros(n, dtype=int)
    assigned_period = np.zeros(n, dtype=float)

    for k, t_now in enumerate(base_periods_arr, start=1):
        unassigned = np.where(freq_labels == 0)[0]
        if unassigned.size < min_pulses:
            break
        tol_abs = phase_tol * t_now
        remaining_mask = np.ones(unassigned.size, dtype=bool)

        while np.sum(remaining_mask) >= min_pulses:
            rem_idx = np.where(remaining_mask)[0]
            rem_phases = np.mod(pulses_arr[unassigned[rem_idx]], t_now)

            neighbor_count = np.zeros(rem_idx.size, dtype=int)
            for i in range(rem_idx.size):
                cdist = np.abs(rem_phases - rem_phases[i])
                cdist = np.minimum(cdist, t_now - cdist)
                neighbor_count[i] = int(np.sum(cdist < tol_abs))

            seed_local = int(np.argmax(neighbor_count))
            max_neighbors = int(neighbor_count[seed_local])
            if max_neighbors < min_pulses:
                break

            seed_phase = rem_phases[seed_local]
            cdist = np.abs(rem_phases - seed_phase)
            cdist = np.minimum(cdist, t_now - cdist)
            in_cluster = cdist < tol_abs

            cluster_global = unassigned[rem_idx[in_cluster]]
            cluster_times = np.sort(pulses_arr[cluster_global])
            intervals = np.diff(cluster_times)
            if intervals.size == 0:
                valid_count = 0
            else:
                ratios = intervals / t_now
                valid_count = int(np.sum(np.abs(ratios - np.round(ratios)) < phase_tol))

            if valid_count >= (min_pulses - 1):
                freq_labels[cluster_global] = k
                assigned_period[cluster_global] = t_now
                remaining_mask[rem_idx[in_cluster]] = False
            else:
                remaining_mask[rem_idx[seed_local]] = False

    group_id = np.zeros(n, dtype=int)
    group_periods: List[float] = []
    group_phases: List[float] = []
    gid = 0

    for k, t_now in enumerate(base_periods_arr, start=1):
        members = np.where(freq_labels == k)[0]
        if members.size == 0:
            continue

        ph = np.mod(pulses_arr[members], t_now)
        assigned = np.zeros(members.size, dtype=bool)
        tol_abs = phase_tol * t_now

        while np.sum(~assigned) >= min_pulses:
            rem = np.where(~assigned)[0]
            rem_ph = ph[rem]

            nc = np.zeros(rem.size, dtype=int)
            for i in range(rem.size):
                cd = np.abs(rem_ph - rem_ph[i])
                cd = np.minimum(cd, t_now - cd)
                nc[i] = int(np.sum(cd < tol_abs))

            seed_idx = int(np.argmax(nc))
            if int(nc[seed_idx]) < min_pulses:
                break

            seed_ph = rem_ph[seed_idx]
            cd = np.abs(rem_ph - seed_ph)
            cd = np.minimum(cd, t_now - cd)
            in_cluster = cd < tol_abs

            gid += 1
            cluster_member_idx = members[rem[in_cluster]]
            group_id[cluster_member_idx] = gid
            assigned[rem[in_cluster]] = True

            cluster_ph = rem_ph[in_cluster]
            mean_phase = np.mod(
                math.atan2(
                    float(np.mean(np.sin(2.0 * np.pi * cluster_ph / t_now))),
                    float(np.mean(np.cos(2.0 * np.pi * cluster_ph / t_now))),
                )
                * t_now
                / (2.0 * np.pi),
                t_now,
            )
            group_periods.append(float(t_now))
            group_phases.append(float(mean_phase))

    summary_rows: List[List[float]] = []
    for g in range(1, gid + 1):
        cnt = int(np.sum(group_id == g))
        summary_rows.append([float(g), float(group_periods[g - 1]), float(group_phases[g - 1]), float(cnt)])

    if verbose:
        print(f"[f_pulse_grouping] pulses={n} groups={gid} unassigned={int(np.sum(group_id == 0))}")

    out = {
        "pulses": pulses_arr,
        "group_id": group_id,
        "periods": np.asarray(group_periods, dtype=float),
        "phases": np.asarray(group_phases, dtype=float),
        "freq_labels": freq_labels,
        "base_periods": base_periods_arr,
        "summary": np.asarray(summary_rows, dtype=float) if summary_rows else np.zeros((0, 4), dtype=float),
        "diffs": diffs,
        "assigned_period": assigned_period,
    }

    if plot:
        _plot_pulse_grouping(out)

    return out


def f_pulse_matching(
    t_scan_group: Dict[str, Any],
    t_sync_group: Dict[str, Any],
    period_tol: float = 0.5,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    scan_summary = np.asarray(t_scan_group.get("summary", np.zeros((0, 4))), dtype=float)
    sync_summary = np.asarray(t_sync_group.get("summary", np.zeros((0, 4))), dtype=float)

    scan_group_id = np.asarray(t_scan_group.get("group_id", []), dtype=int).reshape(-1)
    sync_group_id = np.asarray(t_sync_group.get("group_id", []), dtype=int).reshape(-1)
    scan_pulses = np.asarray(t_scan_group.get("pulses", []), dtype=float).reshape(-1)
    sync_pulses = np.asarray(t_sync_group.get("pulses", []), dtype=float).reshape(-1)

    scan_info: List[Dict[str, Any]] = []
    for row in scan_summary:
        gid = int(row[0])
        t_now = float(row[1])
        idx = scan_group_id == gid
        scan_info.append({"group_id": gid, "period": t_now, "pulses": np.sort(scan_pulses[idx])})

    sync_info: List[Dict[str, Any]] = []
    for row in sync_summary:
        gid = int(row[0])
        t_now = float(row[1])
        idx = sync_group_id == gid
        sync_info.append({"group_id": gid, "period": t_now, "pulses": np.sort(sync_pulses[idx])})

    all_entries: List[List[float]] = []
    for i, g in enumerate(scan_info):
        all_entries.append([g["period"], 1.0, float(i)])
    for i, g in enumerate(sync_info):
        all_entries.append([g["period"], 2.0, float(i)])

    if not all_entries:
        return []

    all_arr = np.asarray(all_entries, dtype=float)
    all_arr = all_arr[np.argsort(all_arr[:, 0])]

    station_labels = np.ones(all_arr.shape[0], dtype=int)
    sid = 1
    for i in range(1, all_arr.shape[0]):
        if all_arr[i, 0] - all_arr[i - 1, 0] > period_tol:
            sid += 1
        station_labels[i] = sid

    stations_raw: List[Dict[str, Any]] = []
    n_stations = int(station_labels.max())

    for s in range(1, n_stations + 1):
        mask = station_labels == s
        entries = all_arr[mask]

        scan_idx = entries[entries[:, 1] == 1, 2].astype(int)
        sync_idx = entries[entries[:, 1] == 2, 2].astype(int)
        mean_period = float(np.mean(entries[:, 0]))

        if sync_idx.size < 1 or scan_idx.size < 2:
            continue

        if sync_idx.size == 1:
            sync_sel = int(sync_idx[0])
        else:
            sync_counts = np.array([len(sync_info[i]["pulses"]) for i in sync_idx], dtype=int)
            sync_sel = int(sync_idx[int(np.argmax(sync_counts))])

        if scan_idx.size == 2:
            scan_sel = scan_idx
        else:
            scan_counts = np.array([len(scan_info[i]["pulses"]) for i in scan_idx], dtype=int)
            order = np.argsort(-scan_counts)
            scan_sel = scan_idx[order[:2]]

        t_sync = np.sort(np.asarray(sync_info[int(sync_sel)]["pulses"], dtype=float))
        t_scan_a = np.sort(np.asarray(scan_info[int(scan_sel[0])]["pulses"], dtype=float))
        t_scan_b = np.sort(np.asarray(scan_info[int(scan_sel[1])]["pulses"], dtype=float))

        t1_list: List[float] = []
        t2_list: List[float] = []
        cycles: List[Dict[str, float]] = []

        for i in range(t_sync.size - 1):
            t01 = float(t_sync[i])
            t02 = float(t_sync[i + 1])
            dt = t02 - t01
            if dt < mean_period * 0.5 or dt > mean_period * 1.5:
                continue

            s_a = t_scan_a[(t_scan_a > t01) & (t_scan_a < t02)]
            s_b = t_scan_b[(t_scan_b > t01) & (t_scan_b < t02)]
            if s_a.size != 1 or s_b.size != 1:
                continue

            ts1 = float(min(s_a[0], s_b[0]))
            ts2 = float(max(s_a[0], s_b[0]))
            t1_norm = (ts1 - t01) / dt
            t2_norm = (ts2 - t01) / dt
            if t1_norm <= 0 or t1_norm >= 1 or t2_norm <= 0 or t2_norm >= 1 or t1_norm >= t2_norm:
                continue

            t1_list.append(float(t1_norm))
            t2_list.append(float(t2_norm))
            cycles.append(
                {
                    "t01": t01,
                    "ts1": ts1,
                    "ts2": ts2,
                    "t02": t02,
                    "t1_norm": float(t1_norm),
                    "t2_norm": float(t2_norm),
                }
            )

        if not t1_list:
            continue

        t1_arr = np.asarray(t1_list, dtype=float)
        t2_arr = np.asarray(t2_list, dtype=float)
        st = {
            "period": mean_period,
            "rpm": 60.0 / mean_period,
            "scan_groups": [
                int(scan_info[int(scan_sel[0])]["group_id"]),
                int(scan_info[int(scan_sel[1])]["group_id"]),
            ],
            "sync_group": int(sync_info[int(sync_sel)]["group_id"]),
            "n_cycles": int(t1_arr.size),
            "t1_all": t1_arr,
            "t2_all": t2_arr,
            "t1_mean": float(np.mean(t1_arr)),
            "t2_mean": float(np.mean(t2_arr)),
            "t1_std": float(np.std(t1_arr, ddof=0)),
            "t2_std": float(np.std(t2_arr, ddof=0)),
            "cycles": cycles,
        }
        stations_raw.append(st)

    if not stations_raw:
        return []

    stations = sorted(stations_raw, key=lambda x: x["rpm"])
    if verbose:
        print(f"[f_pulse_matching] stations={len(stations)}")
    return stations


def _downsample_for_plot(t: np.ndarray, x: np.ndarray, max_pts: int) -> Tuple[np.ndarray, np.ndarray, str]:
    if x.size > max_pts:
        step = max(1, x.size // max_pts)
        idx = np.arange(0, x.size, step)
        return t[idx], x[idx], f" (show {idx.size}/{x.size})"
    return t, x, ""


def _process_layer(
    coef: np.ndarray,
    t_k: np.ndarray,
    fs_k: float,
    mode: str,
    half_win: int,
    thresh_n: float,
    win_time: float,
    ref_fs: float,
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    n = coef.size
    fallback = False

    if mode == "full":
        idx = np.array([0, n - 1], dtype=int)
        times = t_k[idx]
        energy = _compute_energy(coef, fs_k, normalize)
        return idx, times, energy, fallback

    if mode == "peak_window":
        ip = int(np.argmax(np.abs(coef)))
        hpts = max(1, int(round(half_win * fs_k / ref_fs)))
        i1 = max(0, ip - hpts)
        i2 = min(n - 1, ip + hpts)
        idx = np.array([i1, i2], dtype=int)
        times = t_k[idx]
        energy = _compute_energy(coef[i1 : i2 + 1], fs_k, normalize)
        return idx, times, energy, fallback

    if mode == "threshold":
        thr = float(thresh_n * np.std(coef))
        idx = np.where(np.abs(coef) >= thr)[0]
        if idx.size == 0:
            fallback = True
            idx = np.arange(n)
        times = t_k[idx]
        energy = _compute_energy(coef[idx], fs_k, normalize)
        return idx, times, energy, fallback

    if mode == "time_window":
        ip = int(np.argmax(np.abs(coef)))
        hpts = max(1, int(round((win_time / 2.0) * fs_k)))
        i1 = max(0, ip - hpts)
        i2 = min(n - 1, ip + hpts)
        idx = np.array([i1, i2], dtype=int)
        times = t_k[idx]
        energy = _compute_energy(coef[i1 : i2 + 1], fs_k, normalize)
        return idx, times, energy, fallback

    idx = np.array([0, n - 1], dtype=int)
    times = t_k[idx]
    energy = _compute_energy(coef, fs_k, normalize)
    return idx, times, energy, fallback


def _compute_energy(seg: np.ndarray, fs_k: float, normalize: bool) -> float:
    e = float(np.sum(seg**2))
    if normalize and e != 0.0:
        dur = seg.size / fs_k
        e = e / dur
    return e


def _plot_dwt(result: Dict[str, Any], sig: Signal) -> None:
    import matplotlib.pyplot as plt

    L = int(result["level"])
    n_rows = L + 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, max(4, n_rows * 1.5)), sharex=False)

    axes[0].plot(sig.t * 1e3, sig.data, color="0.2", linewidth=0.7)
    axes[0].set_ylabel("raw")
    axes[0].grid(True)

    for k in range(1, L + 1):
        ax = axes[k]
        ax.plot(result["t_k"][k - 1] * 1e3, result["cD"][k - 1], linewidth=0.7)
        ax.set_ylabel(f"cD{k}")
        ax.grid(True)

    axa = axes[-1]
    t_ca = np.arange(result["cA"][L - 1].size, dtype=float) / result["fs_k"][L - 1] * 1e3
    axa.plot(t_ca, result["cA"][L - 1], color="tab:purple", linewidth=0.8)
    axa.set_ylabel(f"cA{L}")
    axa.set_xlabel("Time (ms)")
    axa.grid(True)

    fig.suptitle(f"DWT wavelet={result['wavelet']} level={L}")
    plt.tight_layout()


def _plot_energy(dw: Dict[str, Any], res: Dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    L = int(dw["level"])
    labels = [f"cD{k}" for k in range(1, L + 1)] + [f"cA{L}"]
    vals = list(np.asarray(res["energy_cD"], dtype=float)) + [float(res["energy_cA"])]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, vals)
    plt.ylabel("Power" if res["normalize"] else "Energy")
    plt.title(f"DWT layer energy ({res['mode']})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()


def _plot_pulse_grouping(results: Dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    pulses = np.asarray(results.get("pulses", []), dtype=float).reshape(-1)
    group_id = np.asarray(results.get("group_id", []), dtype=int).reshape(-1)
    diffs = np.asarray(results.get("diffs", []), dtype=float).reshape(-1)
    base_periods = np.asarray(results.get("base_periods", []), dtype=float).reshape(-1)
    periods = np.asarray(results.get("periods", []), dtype=float).reshape(-1)
    phases = np.asarray(results.get("phases", []), dtype=float).reshape(-1)

    n_groups = int(group_id.max()) if group_id.size else 0
    fig = plt.figure(figsize=(13, 8.5))
    colors = plt.cm.get_cmap("tab10", max(n_groups, 1))

    ax1 = fig.add_subplot(2, 2, 1)
    if diffs.size:
        ax1.hist(diffs, bins=min(300, max(20, diffs.size // 4)), color=(0.4, 0.6, 0.9), edgecolor="none")
    for t_now in base_periods:
        ax1.axvline(t_now, color="r", linewidth=1.2)
    ax1.set_xlabel("Delta t")
    ax1.set_ylabel("Count")
    ax1.set_title("Pairwise time differences")
    ax1.grid(True)

    ax2 = fig.add_subplot(2, 2, 2)
    for i, t_now in enumerate(pulses):
        g = int(group_id[i]) if i < group_id.size else 0
        if g == 0:
            ax2.plot(t_now, 0, "kx", markersize=5)
        else:
            c = colors(g - 1)
            ax2.plot(t_now, g, "o", color=c, markerfacecolor=c, markersize=4)
    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Group ID")
    ax2.set_title("Pulse grouping")
    ax2.grid(True)

    ax3 = fig.add_subplot(2, 2, 3)
    for g in range(1, n_groups + 1):
        if g - 1 >= periods.size:
            continue
        t_now = periods[g - 1]
        idx = group_id == g
        ph = np.mod(pulses[idx], t_now) / t_now
        c = colors(g - 1)
        ax3.plot(ph, np.full_like(ph, g), "o", color=c, markerfacecolor=c, markersize=4)
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("Normalized phase")
    ax3.set_ylabel("Group ID")
    ax3.set_title("Phase distribution")
    ax3.grid(True)

    ax4 = fig.add_subplot(2, 2, 4)
    for g in range(1, n_groups + 1):
        if g - 1 >= periods.size or g - 1 >= phases.size:
            continue
        t_now = periods[g - 1]
        ph = phases[g - 1]
        idx = np.where(group_id == g)[0]
        tg = pulses[idx]
        residuals = tg - ph - np.round((tg - ph) / t_now) * t_now
        c = colors(g - 1)
        ax4.plot(tg, residuals, "o", color=c, markerfacecolor=c, markersize=4)
    ax4.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax4.set_xlabel("Time t")
    ax4.set_ylabel("Residual")
    ax4.set_title("Residual to nearest lattice")
    ax4.grid(True)

    fig.suptitle("f_pulse_grouping")
    fig.tight_layout()


def _pulse_model(p: Sequence[float], t: np.ndarray) -> np.ndarray:
    t0, tr, tau_r, tau_f, pmax, vbase = [float(v) for v in p]
    y = np.full_like(t, vbase, dtype=float)

    m_r = (t > t0) & (t < t0 + tr)
    if np.any(m_r):
        denom = 1.0 - math.exp(-tr / tau_r)
        denom = max(denom, 1e-15)
        y[m_r] = vbase + pmax * (1.0 - np.exp(-(t[m_r] - t0) / tau_r)) / denom

    m_f = t >= (t0 + tr)
    if np.any(m_f):
        y[m_f] = vbase + pmax * np.exp(-(t[m_f] - t0 - tr) / tau_f)

    return y


def _scalar(v: Any) -> Any:
    arr = np.asarray(v)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0]
    return v


def _load_npz(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as z:
        for k in z.files:
            out[k] = z[k]
    return out


def _load_mat(path: Path) -> Dict[str, Any]:
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise ImportError("Loading .mat requires scipy") from exc

    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in raw.items() if not k.startswith("__")}
