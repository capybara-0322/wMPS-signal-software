from __future__ import annotations

"""Read/write waveform data in npz format, compatible with dsp_functions.Signal."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dsp_functions import Signal
except Exception:
    Signal = None  # type: ignore[assignment]


def save_waveform_int16_npz(
    file_path: str | Path,
    data: Iterable[int],
    *,
    fs: float,
    scale: float = 1.0,
    meta: Dict[str, Any] | None = None,
    quant_mode: str = "int16_symmetric_v1",
) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    q_values = np.asarray(list(data), dtype=np.int16)
    payload = {
        "q_values": q_values,
        "q_ch1": q_values,
        "fs": np.float64(fs),
        "scale": np.float64(scale),
        "quant_mode": np.array(quant_mode, dtype=object),
        "meta_json": np.array(json.dumps(meta or {}, ensure_ascii=False), dtype=object),
    }
    np.savez_compressed(path, **payload)


def load_waveform_int16_npz(file_path: str | Path) -> Tuple[List[int], float, float, str, Dict[str, Any]]:
    path = Path(file_path)

    with np.load(path, allow_pickle=True) as z:
        if "q_values" in z:
            q_values = z["q_values"].astype(np.int16)
        elif "q_ch1" in z:
            q_values = z["q_ch1"].astype(np.int16)
        else:
            raise KeyError("Neither 'q_values' nor 'q_ch1' found in npz.")

        fs = float(z["fs"]) if "fs" in z else 0.0
        scale = float(z["scale"]) if "scale" in z else 1.0
        quant_mode = str(z["quant_mode"]) if "quant_mode" in z else "int16_symmetric_v1"

        if "meta_json" in z:
            try:
                meta = json.loads(str(z["meta_json"]))
            except Exception:
                meta = {}
        else:
            meta = {}

    return q_values.tolist(), fs, scale, quant_mode, meta


def load_npz_payload(file_path: str | Path) -> Dict[str, Any]:
    """Load raw NPZ payload as a normal dict."""
    path = Path(file_path)
    out: Dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as z:
        for k in z.files:
            out[k] = z[k]
    return out


def load_signal_from_npz(file_path: str | Path, channel: str = "ch1"):
    """Load npz file into a dsp_functions.Signal object."""
    if Signal is None:
        raise ImportError("dsp_functions.Signal is not available in current environment")
    return Signal.from_file(file_path, channel=channel)
