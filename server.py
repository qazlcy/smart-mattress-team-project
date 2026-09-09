"""Smart mattress replay service.

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

# Must match the 44 x 24 tensor expected by sleep_posture.CNN.
ROWS, COLS = 44, 24
ROOT = Path(__file__).parent
DATA_DIR = Path(os.environ.get("MATTRESS_DATA_DIR", ROOT / "data"))


def parse_frames(path: Path, limit: int = 180) -> list[list[list[int]]]:
    """Read consecutive 24-value rows into 44x24 pressure frames."""
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
    if folder.is_dir():
        # A few user folders have a filename prefix different from the folder name.
        candidates = sorted(path for path in folder.glob("*.txt") if path.stem.rsplit("_", 1)[-1] == str(sequence))
        if candidates:
            frames = parse_frames(candidates[0])
            if frames:
                return frames, candidates[0].name
    return demo_frames(), "内置演示数据（未找到本地采集文件）"


_predictor = None


def predict_posture(frame: list[list[int]]) -> tuple[str | None, str]:
    """Predict a coarse posture when a locally trained CNN checkpoint is supplied.

    Set MATTRESS_CNN_WEIGHTS to a checkpoint containing a CNN ``state_dict``.
    No model artifact is committed with the course source, so the replay stays
    usable with an explicit rule fallback until a checkpoint is provided.
    """
    global _predictor
    weights = os.environ.get("MATTRESS_CNN_WEIGHTS")
    if not weights:
        return None, "规则回退（未配置 CNN 权重）"
    try:
        if _predictor is None:
            import torch
            from sleep_posture import config, labels
            from sleep_posture.models import CNN
            model = CNN()
            state = torch.load(weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
            model.eval()
            _predictor = (model, torch, config, labels)
        model, torch, config, labels = _predictor
        tensor = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor = torch.clamp(tensor / config.PRESSURE_FULL_SCALE, 0.0, 1.0)
        with torch.no_grad():
            predicted = int(model(tensor).argmax(dim=1).item())
        return labels.CLASS_NAMES[predicted], "CNN 睡姿模型"
    except Exception as error:  # Keep the live dashboard available if a local artifact is invalid.
        return None, f"规则回退（CNN 不可用：{type(error).__name__}）"


def metrics(frame: list[list[int]]) -> dict:
    flat = [value for row in frame for value in row]
    active = [value for value in flat if value >= 15]
    total = sum(flat)
    weighted_col = sum(value * col for row in frame for col, value in enumerate(row)) / max(total, 1)
    fallback = "右侧卧" if weighted_col < 10.5 else "左侧卧" if weighted_col > 12.5 else "仰卧"
    posture, posture_source = predict_posture(frame)
    posture = posture or fallback
    bands = [(0, 11), (11, 22), (22, 33), (33, 44)]
    airbags = []
    for index, (start, end) in enumerate(bands, 1):
        mean = sum(sum(row) for row in frame[start:end]) / ((end - start) * COLS)
        airbags.append({"id": f"A{index}", "pressure": round(mean, 1), "state": "充气" if mean > 45 else "保持"})
    return {
        "maxPressure": max(flat), "averagePressure": round(sum(active) / max(len(active), 1), 1),
        "contactAreaIndex": round(len(active) / len(flat) * 100, 1), "posture": posture,
        "postureSource": posture_source, "airbags": airbags,
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
