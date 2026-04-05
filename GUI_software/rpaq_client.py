"""RPAQ protocol client aligned with Java implementation."""

from __future__ import annotations

import re
import socket
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

MAGIC_RPAQ = 0x52504151
VERSION = 1
DATA_CHUNK = 128 * 1024


class MsgType(IntEnum):
    DATA = 1
    ACK = 2
    ERR = 3
    INFO = 4


@dataclass
class CaptureConfig:
    decimation: int = 1
    samples: int = 60000
    chan: int = 1
    addr: Optional[int] = None
    bytes: Optional[int] = None
    trig: str = "NOW"

    def to_command(self) -> str:
        parts = [
            "CAPTURE",
            f"dec={self.decimation}",
            f"samples={self.samples}",
            f"chan={self.chan}",
        ]
        if self.addr is not None:
            parts.append(f"addr={self.addr}")
        if self.bytes is not None:
            parts.append(f"bytes={self.bytes}")
        if self.trig:
            parts.append(f"trig={self.trig}")
        return " ".join(parts)


@dataclass
class GenerateConfig:
    chan: int = 1
    frequent: int = 1000
    amplify: float = 0.2
    offset: float = 0.0

    def to_command(self) -> str:
        return f"GEN chan={self.chan} freq={self.frequent} amp={self.amplify} offset={self.offset}"


@dataclass
class CaptureResult:
    data: List[int]
    info_messages: List[str]


@dataclass
class CommonResult:
    info_messages: List[str]


_BYTES_PATTERN = re.compile(r"bytes=(\d+)")


class RpaqClient:
    def __init__(self, host: str, port: int, connect_timeout_ms: int) -> None:
        timeout_s = max(connect_timeout_ms, 1) / 1000.0
        self._sock = socket.create_connection((host, port), timeout=timeout_s)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self) -> None:
        self._sock.close()

    def set_read_timeout_ms(self, timeout_ms: int) -> None:
        self._sock.settimeout(max(timeout_ms, 1) / 1000.0)

    def ping(self) -> str:
        self._send_line("PING")
        msg_type, _, payload = self._read_packet()
        if msg_type == MsgType.ACK:
            return payload.decode("utf-8", errors="replace")
        if msg_type == MsgType.ERR:
            raise OSError(f"Server error: {payload.decode('utf-8', errors='replace')}")
        raise OSError(f"Unexpected response to PING: {msg_type}")

    def quit(self) -> None:
        self._send_line("QUIT")
        msg_type, _, payload = self._read_packet()
        if msg_type == MsgType.ERR:
            raise OSError(f"Server error: {payload.decode('utf-8', errors='replace')}")

    def capture(self, cfg: CaptureConfig) -> CaptureResult:
        self._send_line(cfg.to_command())
        info: List[str] = []
        expected_bytes = -1
        data_buf = bytearray()

        while True:
            msg_type, _, payload = self._read_packet()
            if msg_type == MsgType.ERR:
                raise OSError(f"Server error: {payload.decode('utf-8', errors='replace')}")
            if msg_type == MsgType.INFO:
                msg = payload.decode("utf-8", errors="replace")
                info.append(msg)
                m = _BYTES_PATTERN.search(msg)
                if m:
                    expected_bytes = int(m.group(1))
                continue
            if msg_type == MsgType.ACK:
                info.append(payload.decode("utf-8", errors="replace"))
                continue
            if msg_type == MsgType.DATA:
                data_buf.extend(payload)
                if expected_bytes >= 0 and len(data_buf) >= expected_bytes:
                    return CaptureResult(_to_shorts_le(bytes(data_buf[:expected_bytes])), info)
                if expected_bytes < 0 and len(payload) < DATA_CHUNK:
                    return CaptureResult(_to_shorts_le(bytes(data_buf)), info)
                continue
            raise OSError(f"Unhandled packet type: {msg_type}")

    def generate(self, cfg: GenerateConfig) -> CommonResult:
        self._send_line(cfg.to_command())
        info: List[str] = []
        while True:
            msg_type, _, payload = self._read_packet()
            if msg_type == MsgType.ERR:
                raise OSError(f"Server error: {payload.decode('utf-8', errors='replace')}")
            if msg_type == MsgType.INFO:
                info.append(payload.decode("utf-8", errors="replace"))
                continue
            if msg_type == MsgType.ACK:
                info.append(payload.decode("utf-8", errors="replace"))
                return CommonResult(info)
            raise OSError(f"Unhandled packet type: {msg_type}")

    def _send_line(self, line: str) -> None:
        self._sock.sendall((line + "\n").encode("utf-8"))

    def _read_exact(self, n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise EOFError("Connection closed while reading packet")
            data.extend(chunk)
        return bytes(data)

    def _read_packet(self) -> tuple[MsgType, int, bytes]:
        header = self._read_exact(16)
        magic, version, type_code, seq, payload_len = struct.unpack(">IHHII", header)
        if magic != MAGIC_RPAQ:
            raise OSError(f"Bad magic: 0x{magic:08x}")
        if version != VERSION:
            raise OSError(f"Unsupported version: {version}")
        try:
            msg_type = MsgType(type_code)
        except ValueError as ex:
            raise OSError(f"Unknown message type: {type_code}") from ex
        payload = self._read_exact(payload_len)
        return msg_type, seq, payload


def _to_shorts_le(data: bytes) -> List[int]:
    if len(data) < 2:
        return []
    size = len(data) // 2
    return [v[0] for v in struct.iter_unpack("<h", data[: size * 2])]
