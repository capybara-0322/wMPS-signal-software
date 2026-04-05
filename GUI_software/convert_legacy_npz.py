from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# 用法示例：
#
# # 单文件转换
# python
# GUI_software / convert_legacy_npz.py - -input
# old.npz - -output
# new.npz
#
# # 不写 --output 时，默认生成 old_new.npz
# python
# GUI_software / convert_legacy_npz.py - -input
# old.npz
#
# # 批量转换目录
# python
# GUI_software / convert_legacy_npz.py - -input - dir. / Signals - -pattern
# "*.npz" - -recursive
#
# # 批量输出到新目录
# python
# GUI_software / convert_legacy_npz.py - -input - dir. / Signals - -output - dir. / Signals_new - -recursive






STD_QUANT_MODE = "int16_symmetric_v1"


def _first_existing(payload: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in payload:
            return payload[k]
    return None


def _scalar(v: Any, default: Any = None) -> Any:
    if v is None:
        return default
    arr = np.asarray(v)
    if arr.size == 0:
        return default
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0]
    return v


def _load_any_npz(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as z:
        for k in z.files:
            out[k] = z[k]
    return out


def _parse_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta_json = _first_existing(payload, ["meta_json", "metadata_json", "meta"])
    if meta_json is None:
        return {}

    v = _scalar(meta_json, default={})
    if isinstance(v, dict):
        return dict(v)
    try:
        return json.loads(str(v))
    except Exception:
        return {"legacy_meta_raw": str(v)}


def _load_legacy_signal(payload: Dict[str, Any]) -> Tuple[np.ndarray, float, float, str, Dict[str, Any]]:
    q_raw = _first_existing(payload, ["q_values", "q_ch1", "q", "data", "signal", "y", "ch1"])
    if q_raw is None:
        raise KeyError("No signal data found. Tried keys: q_values/q_ch1/q/data/signal/y/ch1")

    arr = np.asarray(q_raw).reshape(-1)
    if arr.size == 0:
        raise ValueError("Empty signal array")

    fs_raw = _first_existing(payload, ["fs", "Fs", "sample_rate", "sampling_rate"])
    fs = float(_scalar(fs_raw, default=0.0))
    if fs <= 0:
        raise ValueError("Missing valid fs (sampling rate)")

    scale_raw = _first_existing(payload, ["scale", "yscale", "vertical_scale", "factor"])
    scale = float(_scalar(scale_raw, default=1.0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    quant_mode_raw = _first_existing(payload, ["quant_mode", "quant", "mode"])
    quant_mode = str(_scalar(quant_mode_raw, default=STD_QUANT_MODE))
    if not quant_mode:
        quant_mode = STD_QUANT_MODE

    meta = _parse_meta(payload)

    # If already quantized int16-like data, keep as-is; otherwise quantize from float data.
    if np.issubdtype(arr.dtype, np.integer):
        q_values = np.clip(arr.astype(np.int64), -32768, 32767).astype(np.int16)
    else:
        # Treat arr as physical values, quantize using scale.
        q_float = np.round(arr.astype(np.float64) / scale)
        q_values = np.clip(q_float, -32768, 32767).astype(np.int16)

    return q_values, fs, scale, quant_mode, meta


def _save_new_npz(path: Path, q_values: np.ndarray, fs: float, scale: float, quant_mode: str, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "q_values": np.asarray(q_values, dtype=np.int16),
        "q_ch1": np.asarray(q_values, dtype=np.int16),
        "fs": np.float64(fs),
        "scale": np.float64(scale),
        "quant_mode": np.array(quant_mode or STD_QUANT_MODE, dtype=object),
        "meta_json": np.array(json.dumps(meta or {}, ensure_ascii=False), dtype=object),
    }
    np.savez_compressed(path, **payload)


def convert_file(input_path: Path, output_path: Path, overwrite: bool = False) -> str:
    if output_path.exists() and not overwrite:
        return f"SKIP  {input_path} -> {output_path} (exists)"

    payload = _load_any_npz(input_path)
    q_values, fs, scale, quant_mode, meta = _load_legacy_signal(payload)
    _save_new_npz(output_path, q_values, fs, scale, quant_mode, meta)

    return f"OK    {input_path} -> {output_path}  (N={q_values.size}, fs={fs:g}, scale={scale:g})"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert legacy NPZ files to new Signal-compatible NPZ format")
    p.add_argument("--input", type=str, help="Single input .npz file")
    p.add_argument("--output", type=str, help="Single output .npz file")
    p.add_argument("--input-dir", type=str, help="Input directory for batch conversion")
    p.add_argument("--output-dir", type=str, help="Output directory for batch conversion")
    p.add_argument("--pattern", type=str, default="*.npz", help="Glob pattern for batch mode (default: *.npz)")
    p.add_argument("--recursive", action="store_true", help="Use recursive glob in batch mode")
    p.add_argument("--suffix", type=str, default="_new", help="Output filename suffix in batch mode")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            print(f"ERROR input not found: {in_path}")
            return 2
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = in_path.with_name(f"{in_path.stem}_new{in_path.suffix}")
        try:
            print(convert_file(in_path, out_path, overwrite=args.overwrite))
            return 0
        except Exception as exc:
            print(f"ERROR {in_path}: {exc}")
            return 1

    if args.input_dir:
        in_dir = Path(args.input_dir)
        if not in_dir.exists():
            print(f"ERROR input-dir not found: {in_dir}")
            return 2

        out_dir = Path(args.output_dir) if args.output_dir else in_dir
        glob_func = in_dir.rglob if args.recursive else in_dir.glob
        files = sorted([p for p in glob_func(args.pattern) if p.is_file()])
        if not files:
            print("No files matched.")
            return 0

        ok = 0
        fail = 0
        for src in files:
            rel = src.relative_to(in_dir)
            dst_name = f"{src.stem}{args.suffix}{src.suffix}"
            dst = (out_dir / rel).with_name(dst_name)
            try:
                print(convert_file(src, dst, overwrite=args.overwrite))
                ok += 1
            except Exception as exc:
                print(f"ERROR {src}: {exc}")
                fail += 1

        print(f"DONE ok={ok} fail={fail}")
        return 1 if fail else 0

    print("ERROR specify --input or --input-dir")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
