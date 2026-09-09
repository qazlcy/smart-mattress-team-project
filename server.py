"""Smart mattress replay service (standard-library only).

Set MATTRESS_DATA_DIR to the extracted `睡姿数据` directory.  Raw course data
is deliberately kept out of this repository.
"""
from __future__ import annotations

import json
import os
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROWS, COLS = 56, 24
ROOT = Path(__file__).parent
DATA_DIR = Path(os.environ.get("MATTRESS_DATA_DIR", ROOT / "data"))


def parse_frames(path: Path, limit: int = 180) -> list[list[list[int]]]:
    """Read consecutive 24-value rows into 56x24 pressure frames."""
    rows: list[list[int]] = []
    for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        values = [int(value) for value in re.findall(r"\d+", line)]
        if len(values) == COLS:
            rows.append(values)
    return [[rows[i + r] for r in range(ROWS)] for i in range(0, len(rows) - ROWS + 1, ROWS)][:limit]


def demo_frames() -> list[list[list[int]]]:
    frames = []
    for tick in range(90):
        frame = []
        for row in range(ROWS):
            values = []
            for col in range(COLS):
                shoulder = max(0, 190 - ((row - 15) ** 2 * 3 + (col - 12) ** 2 * 7))
                hip = max(0, 235 - ((row - 36) ** 2 * 3 + (col - 12) ** 2 * 6))
                values.append(max(0, int((shoulder + hip) * (0.92 + (tick % 8) / 100))))
            frame.append(values)
        frames.append(frame)
    return frames


def users() -> list[str]:
    if not DATA_DIR.is_dir():
        return ["demo"]
    result = sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
    return result or ["demo"]


def frames_for(user: str, sequence: int) -> tuple[list[list[list[int]]], str]:
    folder = DATA_DIR / user
    candidate = folder / f"{user}_{sequence}.txt"
    if candidate.is_file():
        frames = parse_frames(candidate)
        if frames:
            return frames, candidate.name
    return demo_frames(), "内置演示数据（未找到本地采集文件）"


def metrics(frame: list[list[int]]) -> dict:
    flat = [value for row in frame for value in row]
    active = [value for value in flat if value >= 15]
    total = sum(flat)
    weighted_col = sum(value * col for row in frame for col, value in enumerate(row)) / max(total, 1)
    # This is a transparent rule-based integration fallback, not a trained model.
    posture = "右侧卧" if weighted_col < 10.5 else "左侧卧" if weighted_col > 12.5 else "仰卧"
    bands = [(0, 14), (14, 28), (28, 42), (42, 56)]
    airbags = []
    for index, (start, end) in enumerate(bands, 1):
        mean = sum(sum(row) for row in frame[start:end]) / ((end - start) * COLS)
        airbags.append({"id": f"A{index}", "pressure": round(mean, 1), "state": "充气" if mean > 45 else "保持"})
    return {
        "maxPressure": max(flat), "averagePressure": round(sum(active) / max(len(active), 1), 1),
        "contactAreaIndex": round(len(active) / len(flat) * 100, 1), "posture": posture,
        "postureSource": "规则回退（等待睡姿模型接入）", "airbags": airbags,
    }


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/users":
            return self.respond({"users": users(), "dataDirectory": str(DATA_DIR), "dataAvailable": DATA_DIR.is_dir()})
        if parsed.path == "/api/replay":
            query = parse_qs(parsed.query)
            user = query.get("user", ["demo"])[0]
            try:
                sequence = max(1, min(21, int(query.get("sequence", ["1"])[0])))
            except ValueError:
                sequence = 1
            frames, source = frames_for(user, sequence)
            return self.respond({"user": user, "sequence": sequence, "source": source, "shape": [ROWS, COLS], "frames": frames, "metrics": [metrics(frame) for frame in frames]})
        return super().do_GET()

    def respond(self, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    os.chdir(ROOT / "web")
    print("Open http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
