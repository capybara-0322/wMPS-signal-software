from __future__ import annotations

import math
import tkinter as tk
from tkinter import filedialog, ttk
from typing import List, Optional

from local_file_saver import load_waveform_int16_npz, save_waveform_int16_npz
from rpaq_client import CaptureConfig, GenerateConfig, RpaqClient

MIN_X_SCALE = 0.01
MAX_X_SCALE = 8_000_000.0
MIN_Y_SCALE = 0.01
MAX_Y_SCALE = 8_000_000.0
MIN_SAMPLES_PER_PIXEL = 0.0001
DEFAULT_IP = "192.168.1.104"


class WaveViewerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("wMPS Signal Viewer CUIZHE")
        self.root.geometry("1000x700")
        self.root.configure(bg="#c4bfd3")

        self.rpaq_client: Optional[RpaqClient] = None

        self.x_scale = 1.0
        self.y_scale = 1.0
        self.signal_data: List[int] = []
        self.max_abs_value = 1.0

        self.last_drag_x = 0.0
        self.last_drag_y = 0.0
        self.last_offset_samples = 0.0
        self.last_y_offset_pixels = 0.0
        self.x_offset_samples = 0.0
        self.y_offset_pixels = 0.0

        self._build_layout()
        self._setup_decimation_frequency()
        self._load_placeholder_signal()
        self._draw_signal()

    def _build_layout(self) -> None:
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        left = tk.Frame(self.root, bg="#c4bfd3", width=220)
        left.grid(row=0, column=0, sticky="ns", padx=12, pady=12)
        left.grid_propagate(False)

        center = tk.Frame(self.root, bg="#c4bfd3")
        center.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        center.grid_columnconfigure(0, weight=1)
        center.grid_rowconfigure(0, weight=1)

        right = tk.Frame(self.root, bg="#c4bfd3", width=260)
        right.grid(row=0, column=2, sticky="ns", padx=(0, 12), pady=12)
        right.grid_propagate(False)

        self._build_left_controls(left)
        self._build_wave_canvas(center)
        self._build_right_panels(right)

    def _build_left_controls(self, parent: tk.Frame) -> None:
        def section(title: str) -> tk.Frame:
            wrap = tk.Frame(parent, bg="#c4bfd3")
            wrap.pack(fill="x", pady=8)
            tk.Label(wrap, text=title, bg="#c4bfd3").pack(anchor="w")
            return wrap

        conn = section("Connection")
        row = tk.Frame(conn, bg="#c4bfd3")
        row.pack(fill="x", pady=4)
        tk.Button(row, text="Connect", command=self.on_connect_click).pack(side="left")
        self.ip_var = tk.StringVar(value=DEFAULT_IP)
        tk.Entry(row, textvariable=self.ip_var, width=14).pack(side="left", padx=8)
        self.connection_canvas = tk.Canvas(row, width=16, height=16, bg="#c4bfd3", highlightthickness=0)
        self.connection_canvas.pack(side="left")
        self.connection_indicator = self.connection_canvas.create_oval(2, 2, 14, 14, fill="#9e9e9e", outline="#9e9e9e")

        view = section("View Control")
        row = tk.Frame(view, bg="#c4bfd3")
        row.pack(fill="x", pady=4)
        tk.Button(row, text="Clear", command=self.on_clear_click).pack(side="left")
        tk.Button(row, text="Reset Zoom", command=self.on_reset_zoom_click).pack(side="left", padx=8)

        gen = section("Signal Generate")
        tk.Button(gen, text="Generate Sin", command=self.on_generate_click).pack(anchor="w", pady=3)
        tk.Label(gen, text="Frequency", bg="#c4bfd3").pack(anchor="w")
        self.frequency_var = tk.StringVar()
        tk.Entry(gen, textvariable=self.frequency_var).pack(fill="x", pady=2)
        tk.Label(gen, text="Amplitude", bg="#c4bfd3").pack(anchor="w")
        self.amplitude_var = tk.StringVar()
        tk.Entry(gen, textvariable=self.amplitude_var).pack(fill="x", pady=2)
        tk.Label(gen, text="Bias", bg="#c4bfd3").pack(anchor="w")
        self.bias_var = tk.StringVar()
        tk.Entry(gen, textvariable=self.bias_var).pack(fill="x", pady=2)

        cap = section("Signal Capture")
        cap_top_row = tk.Frame(cap, bg="#c4bfd3")
        cap_top_row.pack(fill="x", pady=3)
        tk.Button(cap_top_row, text="Load Signal", command=self.on_load_click).pack(side="left")
        self.capture_channel_var = tk.StringVar(value="chan1")
        self.capture_channel_box = ttk.Combobox(
            cap_top_row,
            textvariable=self.capture_channel_var,
            state="readonly",
            values=["chan1", "chan2"],
            width=7,
        )
        self.capture_channel_box.pack(side="left", padx=(8, 0))

        dec_row = tk.Frame(cap, bg="#c4bfd3")
        dec_row.pack(fill="x", pady=3)
        left_col = tk.Frame(dec_row, bg="#c4bfd3")
        left_col.pack(side="left", fill="x", expand=True)
        right_col = tk.Frame(dec_row, bg="#c4bfd3")
        right_col.pack(side="left", fill="x", expand=True, padx=(8, 0))

        tk.Label(left_col, text="Decimation", bg="#c4bfd3").pack(anchor="w")
        self.decimation_var = tk.StringVar(value="1")
        self.decimation_box = ttk.Combobox(
            left_col,
            textvariable=self.decimation_var,
            state="readonly",
            values=["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048"],
            width=8,
        )
        self.decimation_box.pack(anchor="w")

        tk.Label(right_col, text="Fs (Hz)", bg="#c4bfd3").pack(anchor="w")
        self.capture_rate_var = tk.StringVar()
        tk.Entry(right_col, textvariable=self.capture_rate_var, state="readonly", width=12).pack(anchor="w")

        tk.Label(cap, text="Samples", bg="#c4bfd3").pack(anchor="w")
        self.samples_var = tk.StringVar()
        tk.Entry(cap, textvariable=self.samples_var).pack(fill="x", pady=2)

        save = section("Save Data")
        row = tk.Frame(save, bg="#c4bfd3")
        row.pack(fill="x", pady=2)
        self.save_path_var = tk.StringVar()
        tk.Entry(row, textvariable=self.save_path_var).pack(side="left", fill="x", expand=True)
        tk.Button(row, text="Browse", command=self.on_browse_save_path_click).pack(side="left", padx=6)
        tk.Button(save, text="Save to File", command=self.on_save_to_file_click).pack(anchor="w", pady=3)

        load = section("Load Data")
        row = tk.Frame(load, bg="#c4bfd3")
        row.pack(fill="x", pady=2)
        self.load_path_var = tk.StringVar()
        tk.Entry(row, textvariable=self.load_path_var).pack(side="left", fill="x", expand=True)
        tk.Button(row, text="Browse", command=self.on_browse_load_path_click).pack(side="left", padx=6)
        tk.Button(load, text="Load from File", command=self.on_load_from_file_click).pack(anchor="w", pady=3)

    def _build_wave_canvas(self, parent: tk.Frame) -> None:
        frame = tk.Frame(parent, bg="#ffffff")
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.wave_canvas = tk.Canvas(frame, bg="white", highlightthickness=0)
        self.wave_canvas.grid(row=0, column=0, sticky="nsew")

        vbar = tk.Scrollbar(frame, orient="vertical", command=self.wave_canvas.yview)
        hbar = tk.Scrollbar(frame, orient="horizontal", command=self.wave_canvas.xview)
        self.wave_canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        self.wave_canvas.configure(scrollregion=(0, 0, 1, 1))

        self.wave_canvas.bind("<Configure>", lambda _e: self._draw_signal())
        self.wave_canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.wave_canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.wave_canvas.bind("<Control-MouseWheel>", self._on_zoom_wheel)
        self.wave_canvas.bind("<Shift-MouseWheel>", self._on_zoom_wheel)

    def _build_right_panels(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="Status", bg="#c4bfd3").pack(anchor="w")
        self.status_area = tk.Text(parent, height=4, wrap="word")
        self.status_area.pack(fill="x", pady=(2, 8))

        tk.Label(parent, text="Info", bg="#c4bfd3").pack(anchor="w")
        self.info_area = tk.Text(parent, height=8, wrap="word")
        self.info_area.pack(fill="x", pady=(2, 8))

        tk.Label(parent, text="Log", bg="#c4bfd3").pack(anchor="w")
        self.log_area = tk.Text(parent, height=14, wrap="word")
        self.log_area.pack(fill="both", expand=True, pady=(2, 0))

        for area in (self.status_area, self.info_area, self.log_area):
            area.configure(state="disabled")

    def _setup_decimation_frequency(self) -> None:
        self.decimation_box.bind("<<ComboboxSelected>>", lambda _e: self._update_capture_rate())
        self._update_capture_rate()

    def _update_capture_rate(self) -> None:
        try:
            decimation = int(self.decimation_var.get().strip())
            rate = 125_000_000 // decimation if decimation > 0 else ""
        except ValueError:
            rate = ""
        self.capture_rate_var.set(str(rate) if rate != "" else "")

    def _set_text(self, area: tk.Text, text: str) -> None:
        area.configure(state="normal")
        area.delete("1.0", tk.END)
        area.insert(tk.END, text)
        area.configure(state="disabled")

    def _append_text(self, area: tk.Text, text: str) -> None:
        area.configure(state="normal")
        area.insert(tk.END, text)
        area.configure(state="disabled")

    def on_connect_click(self) -> None:
        ip = self.ip_var.get().strip() or DEFAULT_IP
        self.ip_var.set(ip)
        try:
            if self.rpaq_client is not None:
                self.rpaq_client.close()
            self.rpaq_client = RpaqClient(ip, 2333, 1000)
            self._set_text(self.status_area, f"Connected to {ip}")
            self._set_connection_status(True)
        except Exception as exc:
            self._set_text(self.status_area, f"Failed to connect: {exc}")
            self._set_connection_status(False)

    def on_clear_click(self) -> None:
        self.wave_canvas.delete("all")

    def on_reset_zoom_click(self) -> None:
        self._set_x_scale(1.0)
        self._set_y_scale(1.0)
        self.x_offset_samples = 0.0
        self.y_offset_pixels = 0.0
        self._draw_signal()

    def on_load_click(self) -> None:
        if self.rpaq_client is None:
            self._set_text(self.status_area, "Not connected. Please click Connect first.")
            return
        try:
            samples = int((self.samples_var.get() or "0").strip())
        except ValueError:
            samples = 0

        cfg = CaptureConfig(
            decimation=int(self.decimation_var.get() or "1"),
            samples=samples,
            chan=2 if self.capture_channel_var.get().strip().lower() == "chan2" else 1,
        )

        try:
            result = self.rpaq_client.capture(cfg)
        except Exception as exc:
            self._set_text(self.status_area, f"Capture failed: {exc}")
            return

        self._set_text(self.info_area, str(result.info_messages))
        self._set_signal_data(result.data)
        self._draw_signal()

    def on_generate_click(self) -> None:
        if self.rpaq_client is None:
            self._set_text(self.status_area, "Not connected. Please click Connect first.")
            return

        cfg = GenerateConfig(
            amplify=self._read_float(self.amplitude_var.get(), 0.8),
            frequent=self._read_int(self.frequency_var.get(), 10000),
            offset=self._read_float(self.bias_var.get(), 0.0),
        )
        try:
            result = self.rpaq_client.generate(cfg)
        except Exception as exc:
            self._set_text(self.status_area, f"Generate failed: {exc}")
            return
        self._set_text(self.info_area, str(result.info_messages))

    def on_browse_save_path_click(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save Signal Data",
            defaultextension=".npz",
            filetypes=[("NumPy NPZ (*.npz)", "*.npz"), ("All Files", "*.*")],
            initialfile="signal.npz",
        )
        if path:
            self.save_path_var.set(path)

    def on_browse_load_path_click(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Signal Data",
            filetypes=[("NumPy NPZ (*.npz)", "*.npz"), ("All Files", "*.*")],
        )
        if path:
            self.load_path_var.set(path)

    def on_save_to_file_click(self) -> None:
        if not self.signal_data:
            self._set_text(self.status_area, "No data to save.")
            return
        path = self.save_path_var.get().strip()
        if not path:
            self._set_text(self.status_area, "Please choose a file path.")
            return
        try:
            save_waveform_int16_npz(
                path,
                self.signal_data,
                fs=self._current_fs(),
                scale=1.0,
                meta=self._build_npz_meta(),
                quant_mode="int16_symmetric_v1",
            )
            self._set_text(self.status_area, f"Saved {len(self.signal_data)} samples to {path}")
        except Exception as exc:
            self._set_text(self.status_area, f"Failed to save: {exc}")

    def on_load_from_file_click(self) -> None:
        path = self.load_path_var.get().strip()
        if not path:
            self._set_text(self.status_area, "Please choose a file path.")
            return
        try:
            data, fs, scale, quant_mode, meta = load_waveform_int16_npz(path)
        except Exception as exc:
            self._set_text(self.status_area, f"Failed to load: {exc}")
            return
        self._set_signal_data(data)
        self._draw_signal()
        if fs > 0:
            self.capture_rate_var.set(str(fs))
        self._set_text(
            self.info_area,
            str(
                {
                    "fs": fs,
                    "scale": scale,
                    "quant_mode": quant_mode,
                    "meta": meta,
                }
            ),
        )
        self._set_text(self.status_area, f"Loaded {len(data)} samples from {path}")

    def _set_connection_status(self, connected: bool) -> None:
        color = "#43a047" if connected else "#9e9e9e"
        self.connection_canvas.itemconfigure(self.connection_indicator, fill=color, outline=color)

    def _on_mouse_press(self, event: tk.Event) -> None:
        self.last_drag_x = event.x
        self.last_drag_y = event.y
        self.last_offset_samples = self.x_offset_samples
        self.last_y_offset_pixels = self.y_offset_pixels

    def _on_mouse_drag(self, event: tk.Event) -> None:
        width = max(self.wave_canvas.winfo_width(), 2)
        height = max(self.wave_canvas.winfo_height(), 2)
        if len(self.signal_data) <= 1 or width <= 1 or height <= 1:
            return

        dx = event.x - self.last_drag_x
        dy = event.y - self.last_drag_y

        base_samples_per_pixel = (len(self.signal_data) - 1) / (width - 1)
        samples_per_pixel = base_samples_per_pixel / self.x_scale

        new_offset = self.last_offset_samples - dx * samples_per_pixel
        max_offset = max(0.0, len(self.signal_data) - 1 - samples_per_pixel * (width - 1))
        self.x_offset_samples = self._clamp(new_offset, 0.0, max_offset)
        self.y_offset_pixels = self.last_y_offset_pixels + dy
        self._draw_signal()

    def _on_zoom_wheel(self, event: tk.Event) -> None:
        zoom_x = bool(event.state & 0x0004)
        zoom_y = bool(event.state & 0x0001)
        if not zoom_x and not zoom_y:
            return

        delta = getattr(event, "delta", 0)
        if delta == 0:
            return

        factor = math.pow(1.1, delta / 40.0)
        width = max(self.wave_canvas.winfo_width(), 2)
        data_len = len(self.signal_data)
        base_samples_per_pixel = (data_len - 1) / (width - 1) if data_len > 1 and width > 1 else 1.0

        old_samples_per_pixel = base_samples_per_pixel / self.x_scale
        old_visible_samples = old_samples_per_pixel * max(1.0, width - 1)
        center_sample = self.x_offset_samples + old_visible_samples / 2.0

        if zoom_x:
            self._set_x_scale(self.x_scale * factor)
        if zoom_y:
            self._set_y_scale(self.y_scale * factor)

        new_samples_per_pixel = base_samples_per_pixel / self.x_scale
        new_visible_samples = new_samples_per_pixel * max(1.0, width - 1)
        self.x_offset_samples = center_sample - new_visible_samples / 2.0
        self.x_offset_samples = self._clamp(self.x_offset_samples, 0.0, max(0.0, data_len - 1 - new_visible_samples))

        self._draw_signal()

    def _set_x_scale(self, scale: float) -> None:
        self.x_scale = max(MIN_X_SCALE, min(self._get_max_x_scale(), scale))

    def _set_y_scale(self, scale: float) -> None:
        self.y_scale = max(MIN_Y_SCALE, min(MAX_Y_SCALE, scale))

    def _load_placeholder_signal(self) -> None:
        points = 200000
        data = []
        for i in range(points):
            t = (2 * math.pi * i) / 2000.0
            v = math.sin(6 * t) * 0.7 + math.sin(20 * t) * 0.2
            data.append(int(round(v * 10000.0)))
        self._set_signal_data(data)

    def _set_signal_data(self, data: List[int]) -> None:
        self.signal_data = data or []
        self.max_abs_value = 1.0
        self.x_offset_samples = 0.0
        self.y_offset_pixels = 0.0
        self._set_x_scale(self.x_scale)
        for v in self.signal_data:
            abs_v = abs(float(v))
            if abs_v > self.max_abs_value:
                self.max_abs_value = abs_v

    def _draw_signal(self) -> None:
        self.wave_canvas.delete("all")
        width = max(self.wave_canvas.winfo_width(), 1)
        height = max(self.wave_canvas.winfo_height(), 1)

        if not self.signal_data or width <= 1 or height <= 1:
            return

        for x in range(0, width + 1, 100):
            self.wave_canvas.create_line(x, 0, x, height, fill="#cfd8dc", width=1)
        for y in range(0, height + 1, 100):
            self.wave_canvas.create_line(0, y, width, y, fill="#cfd8dc", width=1)

        center_y = height / 2.0 + self.y_offset_pixels
        y_scale_px = (height * 0.45) * self.y_scale / self.max_abs_value

        data_len = len(self.signal_data)
        pixel_width = max(int(width), 1)
        if pixel_width <= 1 or data_len <= 1:
            return

        base_samples_per_pixel = (data_len - 1) / float(pixel_width - 1)
        samples_per_pixel = base_samples_per_pixel / self.x_scale
        visible_samples = samples_per_pixel * (pixel_width - 1)
        max_offset = max(0.0, data_len - 1 - visible_samples)
        self.x_offset_samples = self._clamp(self.x_offset_samples, 0.0, max_offset)

        if samples_per_pixel < 1.0:
            coords = []
            prev_y = center_y - self._sample_at(self.x_offset_samples) * y_scale_px
            coords.extend([0.0, prev_y])
            for x in range(1, pixel_width):
                sample_index = self.x_offset_samples + x * samples_per_pixel
                y = center_y - self._sample_at(sample_index) * y_scale_px
                coords.extend([float(x), y])
            self.wave_canvas.create_line(*coords, fill="#1565c0", width=2, smooth=False)
        else:
            for x in range(pixel_width):
                sample_start = self.x_offset_samples + x * samples_per_pixel
                sample_end = self.x_offset_samples + (x + 1) * samples_per_pixel
                start = max(0, int(math.floor(sample_start)))
                end = min(data_len - 1, max(start, int(math.floor(sample_end))))
                min_v = self.signal_data[start]
                max_v = min_v
                for i in range(start, end + 1):
                    v = self.signal_data[i]
                    if v < min_v:
                        min_v = v
                    elif v > max_v:
                        max_v = v
                y1 = center_y - max_v * y_scale_px
                y2 = center_y - min_v * y_scale_px
                self.wave_canvas.create_line(x + 0.5, y1, x + 0.5, y2, fill="#1565c0", width=2)

    def _sample_at(self, index: float) -> float:
        if not self.signal_data:
            return 0.0
        if index <= 0:
            return float(self.signal_data[0])
        last = len(self.signal_data) - 1
        if index >= last:
            return float(self.signal_data[last])
        i0 = int(math.floor(index))
        i1 = i0 + 1
        t = index - i0
        return self.signal_data[i0] * (1.0 - t) + self.signal_data[i1] * t

    def _get_max_x_scale(self) -> float:
        if len(self.signal_data) <= 1:
            return MAX_X_SCALE
        width = self.wave_canvas.winfo_width()
        if width <= 1.0:
            return MAX_X_SCALE
        base_samples_per_pixel = (len(self.signal_data) - 1) / (width - 1)
        max_from_pixel = base_samples_per_pixel / MIN_SAMPLES_PER_PIXEL
        return min(MAX_X_SCALE, max(MIN_X_SCALE, max_from_pixel))

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        if value < min_value:
            return min_value
        if value > max_value:
            return max_value
        return value

    @staticmethod
    def _read_int(raw: str, default: int) -> int:
        s = raw.strip()
        if not s:
            return default
        try:
            return int(s)
        except ValueError:
            return default

    @staticmethod
    def _read_float(raw: str, default: float) -> float:
        s = raw.strip()
        if not s:
            return default
        try:
            return float(s)
        except ValueError:
            return default

    def _current_fs(self) -> float:
        try:
            fs = float(self.capture_rate_var.get().strip())
            if fs > 0:
                return fs
        except ValueError:
            pass
        return 125_000_000.0

    def _build_npz_meta(self) -> dict:
        fs = self._current_fs()
        sample_interval = 1.0 / fs if fs > 0 else 0.0
        record_length = len(self.signal_data)
        return {
            "Model": "GUI_software",
            "Firmware Version": "",
            "Waveform Type": "ANALOG",
            "Point Format": "Y",
            "Horizontal Units": "s",
            "Horizontal Scale": "",
            "Horizontal Delay": "",
            "Sample Interval": f"{sample_interval:.12g}",
            "Record Length": str(record_length),
            "Gating": "0.0% to 100.0%",
            "Probe Attenuation": "1",
            "Vertical Units": "V",
            "Vertical Offset": self.bias_var.get().strip() or "0",
            "Vertical Scale": "",
            "Vertical Position": "0",
            "Source IP": self.ip_var.get().strip(),
            "Decimation": str(self._read_int(self.decimation_var.get(), 1)),
            "Capture Fs": f"{fs:.12g}",
        }


if __name__ == "__main__":
    root = tk.Tk()
    app = WaveViewerApp(root)
    root.mainloop()
