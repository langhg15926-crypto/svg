from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont


# -----------------------------
# Utils
# -----------------------------
def now_ts() -> float:
    return time.time()


def mm_to_pt(mm: float) -> float:
    # 1 inch = 25.4mm, 1pt = 1/72 inch
    return mm * 72.0 / 25.4


def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def safe_int(x: float) -> int:
    return int(round(float(x)))


def gray_to_hex(g: float) -> str:
    v = int(round(clamp(g, 0, 255)))
    return f"#{v:02x}{v:02x}{v:02x}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Config models
# -----------------------------
class StyleConfig(BaseModel):
    max_width_mm: float = 80.0
    font_family: str = "'FZShuSong-Z01', 'Times New Roman', 'SimSun', serif"
    axis_font_size_pt: float = 9.0
    label_font_size_pt: float = 9.0
    stroke_main_pt: float = 0.6
    stroke_axis_pt: float = 0.6
    stroke_dash_pt: float = 0.6
    stroke_color: str = "#000000"
    dash_array_default_pt: str = "2.0,2.0"


class LLMConfig(BaseModel):
    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_sec: int = 120
    max_retries: int = 1

    # OCR / refine switches
    use_llm_ocr: bool = True
    use_llm_refine: bool = False  # 结构化二次修正（可选）

    # capability flags from UI (not strictly enforced, used for warnings)
    vision: bool = True
    web: bool = True
    reasoning: bool = True
    tools: bool = True


class DetectConfig(BaseModel):
    # Binarize
    adaptive_block_size: int = 35
    adaptive_C: int = 10

    # Hough on skeleton (auto-tuned if auto_tune=True)
    auto_tune: bool = True
    hough_threshold: int = 40
    hough_min_line_length: int = 12
    hough_max_line_gap: int = 4

    # Merge
    merge_angle_deg: float = 2.0
    merge_dist_px: float = 2.0

    # Endpoints
    near_dist_px: float = 2.0
    endpoint_extend_px: int = 12
    snap_axis_eps_px: float = 0.35

    # Dashed detection
    dash_min_switches: int = 6
    dash_noise_run_px: int = 2

    # Filled rect detection (bars)
    filled_min_area: int = 140
    filled_min_fill_ratio: float = 0.35
    filled_min_d90_px: float = 2.2

    # Text boxes
    text_dilate_px: int = 2
    text_min_area: int = 18
    text_max_box_cover_ratio: float = 0.55  # reject huge boxes
    text_merge_gap_px: int = 4
    text_pad_px: int = 2

    # Arrow removal: here we interpret it as “no arrowheads/shapes”.
    # We do NOT output arrow markers; we also avoid outputting small filled triangles by filled_min_area threshold.
    remove_arrowheads: bool = True


class AppConfig(BaseModel):
    style: StyleConfig = Field(default_factory=StyleConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)


# -----------------------------
# Job state
# -----------------------------
def mk_step(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "status": "pending",   # pending|running|done|error
        "t_start": None,
        "t_end": None,
        "message": "",
    }


@dataclass
class JobItemState:
    index: int
    filename: str
    status: str = "pending"  # pending|processing|done|error
    current_step: str = ""
    steps: List[Dict[str, Any]] = None
    outputs: Dict[str, str] = None
    error: Optional[str] = None
    logs: List[str] = None
    llm_calls: int = 0

    def __post_init__(self):
        self.steps = []
        self.outputs = {}
        self.logs = []


@dataclass
class JobState:
    job_id: str
    status: str = "running"  # running|done|error
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    config_summary: Dict[str, Any] = None
    items: List[JobItemState] = None
    logs: List[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        self.config_summary = {}
        self.items = []
        self.logs = []


class JobStore:
    def __init__(self):
        self._lock = Lock()
        self._jobs: Dict[str, JobState] = {}

    def create_job(self, config: AppConfig, filenames: List[str]) -> JobState:
        job_id = uuid.uuid4().hex[:12]
        st = JobState(job_id=job_id, created_at=now_ts(), started_at=now_ts())
        st.config_summary = {
            "llm_enabled": config.llm.enabled,
            "llm_model": config.llm.model,
            "use_llm_ocr": config.llm.use_llm_ocr,
            "use_llm_refine": config.llm.use_llm_refine,
            "max_width_mm": config.style.max_width_mm,
        }
        st.items = [JobItemState(index=i + 1, filename=fn) for i, fn in enumerate(filenames)]
        with self._lock:
            self._jobs[job_id] = st
        return st

    def get(self, job_id: str) -> JobState:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    def add_job_log(self, job_id: str, msg: str) -> None:
        with self._lock:
            st = self._jobs[job_id]
            st.logs.append(msg)

    def add_item_log(self, job_id: str, idx: int, msg: str) -> None:
        with self._lock:
            st = self._jobs[job_id]
            st.items[idx].logs.append(msg)

    def step_start(self, job_id: str, idx: int, name: str, message: str = "") -> None:
        with self._lock:
            item = self._jobs[job_id].items[idx]
            item.current_step = name
            step = mk_step(name)
            step["status"] = "running"
            step["t_start"] = now_ts()
            step["message"] = message
            item.steps.append(step)

    def step_done(self, job_id: str, idx: int, name: str, message: str = "") -> None:
        with self._lock:
            item = self._jobs[job_id].items[idx]
            for s in reversed(item.steps):
                if s["name"] == name and s["status"] == "running":
                    s["status"] = "done"
                    s["t_end"] = now_ts()
                    if message:
                        s["message"] = message
                    break

    def step_error(self, job_id: str, idx: int, name: str, err: str) -> None:
        with self._lock:
            item = self._jobs[job_id].items[idx]
            for s in reversed(item.steps):
                if s["name"] == name and s["status"] == "running":
                    s["status"] = "error"
                    s["t_end"] = now_ts()
                    s["message"] = err
                    break
            item.status = "error"
            item.error = err

    def set_item_status(self, job_id: str, idx: int, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            item = self._jobs[job_id].items[idx]
            item.status = status
            if error:
                item.error = error

    def set_item_outputs(self, job_id: str, idx: int, outputs: Dict[str, str]) -> None:
        with self._lock:
            item = self._jobs[job_id].items[idx]
            item.outputs.update(outputs)

    def inc_item_llm_calls(self, job_id: str, idx: int, n: int = 1) -> None:
        with self._lock:
            self._jobs[job_id].items[idx].llm_calls += n

    def finalize_job(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            st = self._jobs[job_id]
            st.status = status
            st.finished_at = now_ts()
            st.error = error


STORE = JobStore()


# -----------------------------
# OpenAI-compatible client
# -----------------------------
class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def list_models(self) -> List[str]:
        # try /v1/models then /models
        for path in ("/v1/models", "/models"):
            url = self.base_url + path
            r = requests.get(url, headers=self._headers(), timeout=self.timeout)
            if r.status_code >= 400:
                continue
            data = r.json()
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                ids = []
                for m in data["data"]:
                    mid = m.get("id")
                    if mid:
                        ids.append(mid)
                return sorted(set(ids))
            if isinstance(data, list):
                # some proxies return list of model ids
                return sorted(set([str(x) for x in data]))
        raise RuntimeError("Failed to list models from this base_url")

    def chat(self, model: str, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        url = self.base_url + "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        r = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"chat/completions error {r.status_code}: {r.text[:500]}")
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data)[:500]

    def vision_chat(self, model: str, prompt: str, image_png_bytes: bytes) -> str:
        b64 = base64.b64encode(image_png_bytes).decode("utf-8")
        url_data = f"data:image/png;base64,{b64}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url_data}},
                ],
            }
        ]
        return self.chat(model=model, messages=messages, temperature=0.0)


def extract_first_json_object(text: str) -> Optional[dict]:
    # Try to find the first {...} block that can be parsed.
    # This is intentionally conservative.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except Exception:
        # try removing code fences
        blob2 = re.sub(r"```(?:json)?", "", blob).replace("```", "")
        try:
            return json.loads(blob2)
        except Exception:
            return None


# -----------------------------
# Vectorization core
# -----------------------------
def binarize(gray: np.ndarray, block_size: int, C: int) -> np.ndarray:
    bs = block_size
    if bs % 2 == 0:
        bs += 1
    bs = max(3, bs)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        bs,
        C,
    )
    return bw


def skeletonize(binary: np.ndarray) -> np.ndarray:
    img = binary.copy()
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def estimate_stroke_width_px(mask: np.ndarray) -> float:
    # distanceTransform measures distance to background; thin lines => small distances
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    vals = dist[dist > 0]
    if vals.size == 0:
        return 1.0
    # typical half-width around median
    return float(np.median(vals) * 2.0)


def tune_hough(dc: DetectConfig, w: int, h: int, stroke_width: float) -> Tuple[int, int, int]:
    # produce (threshold, minLen, maxGap)
    if not dc.auto_tune:
        return dc.hough_threshold, dc.hough_min_line_length, dc.hough_max_line_gap

    m = min(w, h)
    sw = clamp(stroke_width, 1.0, 12.0)
    thr = max(20, int(m * 0.12 + sw * 2.0))
    min_len = max(10, int(m * 0.08 + sw * 4.0))
    max_gap = max(2, int(m * 0.02 + sw * 2.0))
    return thr, min_len, max_gap


def point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    # distance from p to infinite line through a-b
    ab = b - a
    if np.linalg.norm(ab) < 1e-6:
        return float(np.linalg.norm(p - a))
    return float(np.abs(np.cross(ab, p - a)) / np.linalg.norm(ab))


def gather_points_near_segment(
    pts: np.ndarray, a: np.ndarray, b: np.ndarray, near_dist: float
) -> np.ndarray:
    # pts: Nx2
    # quick bbox filter first
    x1, y1 = a
    x2, y2 = b
    xmin = min(x1, x2) - near_dist
    xmax = max(x1, x2) + near_dist
    ymin = min(y1, y2) - near_dist
    ymax = max(y1, y2) + near_dist
    in_box = (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) & (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax)
    cand = pts[in_box]
    if cand.size == 0:
        return cand
    # distance to infinite line
    dists = np.array([point_line_distance(p, a, b) for p in cand], dtype=np.float32)
    return cand[dists <= near_dist + 1e-6]


def tls_fit_line(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # returns (center, direction_unit)
    c = points.mean(axis=0)
    X = points - c
    # SVD on covariance
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    direction = vt[0]
    n = np.linalg.norm(direction)
    if n < 1e-8:
        direction = np.array([1.0, 0.0], dtype=np.float32)
    else:
        direction = direction / n
    return c.astype(np.float32), direction.astype(np.float32)


def refine_endpoints_to_mask(
    mask: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    max_extend: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    # Move endpoints along the line direction until leaving ink; clamp inside.
    h, w = mask.shape[:2]
    v = p2 - p1
    ln = float(np.linalg.norm(v))
    if ln < 1e-6:
        return p1, p2
    d = v / ln

    def has_ink(pt: np.ndarray) -> bool:
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        return mask[y, x] > 0

    # extend backward for p1
    q1 = p1.copy()
    for _ in range(max_extend):
        cand = q1 - d
        if has_ink(cand):
            q1 = cand
        else:
            break

    # extend forward for p2
    q2 = p2.copy()
    for _ in range(max_extend):
        cand = q2 + d
        if has_ink(cand):
            q2 = cand
        else:
            break

    # clamp
    q1[0] = clamp(q1[0], 0, w - 1)
    q1[1] = clamp(q1[1], 0, h - 1)
    q2[0] = clamp(q2[0], 0, w - 1)
    q2[1] = clamp(q2[1], 0, h - 1)

    return q1, q2


def snap_axis(p1: np.ndarray, p2: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    # If nearly horizontal or vertical, snap to crisp pixel lines.
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    if abs(dy) <= eps:
        y = (p1[1] + p2[1]) * 0.5
        p1[1] = y
        p2[1] = y
    if abs(dx) <= eps:
        x = (p1[0] + p2[0]) * 0.5
        p1[0] = x
        p2[0] = x
    return p1, p2


def segment_angle_deg(p1: np.ndarray, p2: np.ndarray) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    ang = math.degrees(math.atan2(dy, dx))
    # normalize to [0,180)
    ang = (ang + 180.0) % 180.0
    return ang


def seg_len(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p2 - p1))


def segment_projection_interval(p1: np.ndarray, p2: np.ndarray, direction: np.ndarray) -> Tuple[float, float]:
    d = direction / max(1e-6, float(np.linalg.norm(direction)))
    t1 = float(p1 @ d)
    t2 = float(p2 @ d)
    return (min(t1, t2), max(t1, t2))


def merge_collinear_segments(segments: List[Tuple[np.ndarray, np.ndarray]], angle_tol_deg: float, dist_tol_px: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Simple greedy merge: repeatedly merge if angles close and endpoints close
    out = segments[:]
    changed = True
    while changed:
        changed = False
        used = [False] * len(out)
        new_list: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(out)):
            if used[i]:
                continue
            a1, a2 = out[i]
            ang1 = segment_angle_deg(a1, a2)
            merged = (a1.copy(), a2.copy())
            used[i] = True
            for j in range(i + 1, len(out)):
                if used[j]:
                    continue
                b1, b2 = out[j]
                ang2 = segment_angle_deg(b1, b2)
                da = min(abs(ang1 - ang2), 180 - abs(ang1 - ang2))
                if da > angle_tol_deg:
                    continue
                # check endpoint proximity
                # If any endpoints are close, consider merge
                dmin = min(
                    np.linalg.norm(merged[0] - b1),
                    np.linalg.norm(merged[0] - b2),
                    np.linalg.norm(merged[1] - b1),
                    np.linalg.norm(merged[1] - b2),
                )
                if dmin > dist_tol_px:
                    continue
                # ensure projection overlap or very small gap
                direction = merged[1] - merged[0]
                if np.linalg.norm(direction) < 1e-6:
                    continue
                a_min, a_max = segment_projection_interval(merged[0], merged[1], direction)
                b_min, b_max = segment_projection_interval(b1, b2, direction)
                gap = max(a_min, b_min) - min(a_max, b_max)
                if gap > dist_tol_px:
                    continue
                # merge by projecting all 4 endpoints onto merged direction
                pts = np.vstack([merged[0], merged[1], b1, b2])
                c, d = tls_fit_line(pts)
                t = (pts - c) @ d
                p_min = c + d * float(t.min())
                p_max = c + d * float(t.max())
                merged = (p_min.astype(np.float32), p_max.astype(np.float32))
                used[j] = True
                changed = True
            new_list.append(merged)
        out = new_list
    return out


def detect_dashed(mask: np.ndarray, p1: np.ndarray, p2: np.ndarray, noise_run: int, min_switches: int) -> bool:
    # Sample along line; count transitions between ink/background
    h, w = mask.shape[:2]
    v = p2 - p1
    L = float(np.linalg.norm(v))
    if L < 10:
        return False
    d = v / L
    n = int(max(40, min(600, L * 2)))
    run = []
    last = None
    cnt = 0
    switches = 0

    def sample(pt: np.ndarray) -> int:
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        if x < 0 or x >= w or y < 0 or y >= h:
            return 0
        return 1 if mask[y, x] > 0 else 0

    for i in range(n):
        t = (L * i) / (n - 1)
        pt = p1 + d * t
        s = sample(pt)
        if last is None:
            last = s
            cnt = 1
            continue
        if s == last:
            cnt += 1
        else:
            # ignore tiny noise runs
            if cnt >= noise_run:
                switches += 1
            last = s
            cnt = 1
    return switches >= min_switches


def detect_filled_rects(gray: np.ndarray, mask: np.ndarray, dc: DetectConfig) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    # Return rect objects and filled_mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape[:2]
    filled = np.zeros_like(mask)
    rects: List[Dict[str, Any]] = []

    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < dc.filled_min_area:
            continue
        bbox_area = max(1, bw * bh)
        fill_ratio = float(area) / float(bbox_area)
        if fill_ratio < dc.filled_min_fill_ratio:
            continue

        comp = (labels == i).astype(np.uint8) * 255
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        vals = dist[dist > 0]
        if vals.size == 0:
            continue
        d90 = float(np.percentile(vals, 90))
        if d90 < dc.filled_min_d90_px:
            continue

        # sample fill color from original gray inside component
        crop_g = gray[y:y+bh, x:x+bw]
        crop_m = comp[y:y+bh, x:x+bw] > 0
        if crop_m.sum() < 10:
            continue
        med = float(np.median(crop_g[crop_m]))
        fill_hex = gray_to_hex(med)

        rects.append({
            "type": "rect",
            "id": f"rect_{len(rects)+1:04d}",
            "x": int(x),
            "y": int(y),
            "w": int(bw),
            "h": int(bh),
            "fill": fill_hex,
            "role": "fill",
        })
        filled[labels == i] = 255

    return rects, filled


def detect_lines_from_skeleton(
    mask_lines: np.ndarray,
    dc: DetectConfig,
    stroke_width: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    h, w = mask_lines.shape[:2]
    skel = skeletonize(mask_lines)
    ys, xs = np.where(skel > 0)
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1) if xs.size else np.zeros((0,2), np.float32)

    thr, min_len, max_gap = tune_hough(dc, w, h, stroke_width)
    sw = max(1.0, stroke_width)
    near_dist = max(dc.near_dist_px, sw * 0.6)
    endpoint_extend = max(dc.endpoint_extend_px, int(round(sw * 6.0)))
    snap_axis_eps = max(dc.snap_axis_eps_px, sw * 0.15)
    merge_dist = max(dc.merge_dist_px, sw * 0.6)
    lines = cv2.HoughLinesP(
        skel,
        rho=1,
        theta=np.pi / 180.0,
        threshold=thr,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )

    segs: List[Tuple[np.ndarray, np.ndarray]] = []
    if lines is None:
        return segs

    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, l.tolist())
        a = np.array([x1, y1], dtype=np.float32)
        b = np.array([x2, y2], dtype=np.float32)

        near = gather_points_near_segment(pts, a, b, near_dist)
        if near.shape[0] >= 8:
            c, d = tls_fit_line(near)
            t = (near - c) @ d
            p1 = c + d * float(t.min())
            p2 = c + d * float(t.max())
        else:
            p1, p2 = a, b

        p1, p2 = refine_endpoints_to_mask(mask_lines, p1, p2, max_extend=endpoint_extend)
        p1, p2 = snap_axis(p1, p2, eps=snap_axis_eps)

        if seg_len(p1, p2) >= 6.0:
            segs.append((p1, p2))

    # merge collinear
    segs = merge_collinear_segments(segs, angle_tol_deg=dc.merge_angle_deg, dist_tol_px=merge_dist)

    # clamp again
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for p1, p2 in segs:
        p1[0] = clamp(p1[0], 0, w - 1)
        p1[1] = clamp(p1[1], 0, h - 1)
        p2[0] = clamp(p2[0], 0, w - 1)
        p2[1] = clamp(p2[1], 0, h - 1)
        if seg_len(p1, p2) >= 6.0:
            out.append((p1, p2))
    return out


def rasterize_lines(shape: Tuple[int, int], segs: List[Tuple[np.ndarray, np.ndarray]], thickness: int) -> np.ndarray:
    h, w = shape
    img = np.zeros((h, w), dtype=np.uint8)
    for p1, p2 in segs:
        cv2.line(img, (safe_int(p1[0]), safe_int(p1[1])), (safe_int(p2[0]), safe_int(p2[1])), 255, thickness=thickness)
    return img


def merge_text_boxes(boxes: List[Tuple[int,int,int,int]], gap: int) -> List[Tuple[int,int,int,int]]:
    # merge horizontally if same row overlap and close
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged: List[Tuple[int,int,int,int]] = []

    def overlap_y(a, b) -> float:
        ay1, ay2 = a[1], a[1]+a[3]
        by1, by2 = b[1], b[1]+b[3]
        inter = max(0, min(ay2, by2) - max(ay1, by1))
        return inter / max(1, min(a[3], b[3]))

    cur = boxes[0]
    for b in boxes[1:]:
        if overlap_y(cur, b) >= 0.55:
            # close enough?
            cur_right = cur[0] + cur[2]
            b_left = b[0]
            if b_left - cur_right <= gap:
                # merge
                x1 = min(cur[0], b[0])
                y1 = min(cur[1], b[1])
                x2 = max(cur[0] + cur[2], b[0] + b[2])
                y2 = max(cur[1] + cur[3], b[1] + b[3])
                cur = (x1, y1, x2-x1, y2-y1)
                continue
        merged.append(cur)
        cur = b
    merged.append(cur)
    return merged


def detect_text_boxes(mask: np.ndarray, segs: List[Tuple[np.ndarray, np.ndarray]], filled_mask: np.ndarray, dc: DetectConfig) -> List[Dict[str, Any]]:
    h, w = mask.shape[:2]

    # remove filled regions
    m = cv2.bitwise_and(mask, cv2.bitwise_not(filled_mask))

    # remove lines by rasterizing them
    # thickness derived from dilation param; small to avoid eating text
    thick = max(1, dc.text_dilate_px + 1)
    line_r = rasterize_lines((h, w), [s for s in segs if seg_len(s[0], s[1]) >= 10], thickness=thick)
    m2 = cv2.bitwise_and(m, cv2.bitwise_not(line_r))

    # dilate to connect character strokes
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (dc.text_dilate_px*2+1, dc.text_dilate_px*2+1))
    m3 = cv2.dilate(m2, k, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m3, connectivity=8)
    boxes: List[Tuple[int,int,int,int]] = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < dc.text_min_area:
            continue
        # reject huge component (the "whole image" bug)
        if (bw * bh) / float(w * h) > dc.text_max_box_cover_ratio:
            continue
        boxes.append((int(x), int(y), int(bw), int(bh)))

    boxes = merge_text_boxes(boxes, gap=dc.text_merge_gap_px)

    out: List[Dict[str, Any]] = []
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        out.append({
            "type": "text_box",
            "id": f"text_{i:04d}",
            "x": int(x),
            "y": int(y),
            "w": int(bw),
            "h": int(bh),
            "text": "",     # to be filled by OCR
            "role": "label",
            "anchor": "middle",
        })
    return out


def make_ocr_collage(gray: np.ndarray, text_boxes: List[Dict[str, Any]], pad_px: int = 2) -> Tuple[bytes, List[str]]:
    # Create a collage image and return PNG bytes + ids in order.
    # Each tile includes "id" label for mapping.
    if not text_boxes:
        return b"", []

    crops: List[Image.Image] = []
    ids: List[str] = []

    # prepare font for labels
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    pil_gray = Image.fromarray(gray)
    for tb in text_boxes:
        x, y, w, h = tb["x"], tb["y"], tb["w"], tb["h"]
        x1 = max(0, x - pad_px)
        y1 = max(0, y - pad_px)
        x2 = min(gray.shape[1], x + w + pad_px)
        y2 = min(gray.shape[0], y + h + pad_px)
        crop = pil_gray.crop((x1, y1, x2, y2)).convert("RGB")
        draw = ImageDraw.Draw(crop)
        draw.rectangle((0, 0, crop.size[0]-1, crop.size[1]-1), outline=(255, 0, 0), width=1)
        draw.text((2, 2), tb["id"], fill=(255, 0, 0), font=font)
        crops.append(crop)
        ids.append(tb["id"])

    # grid layout
    cols = 4 if len(crops) >= 4 else len(crops)
    cols = max(1, cols)
    rows = int(math.ceil(len(crops) / cols))
    tile_w = max(c.size[0] for c in crops)
    tile_h = max(c.size[1] for c in crops)
    gap = 10

    canvas_w = cols * tile_w + (cols + 1) * gap
    canvas_h = rows * tile_h + (rows + 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, crop in enumerate(crops):
        r = idx // cols
        c = idx % cols
        x0 = gap + c * (tile_w + gap)
        y0 = gap + r * (tile_h + gap)
        canvas.paste(crop, (x0, y0))
        # optional: grid guides
        draw.rectangle((x0, y0, x0 + crop.size[0], y0 + crop.size[1]), outline=(220, 220, 220), width=1)

    bio = io.BytesIO()
    canvas.save(bio, format="PNG")
    return bio.getvalue(), ids


def export_svg(
    meta: Dict[str, Any],
    objects: List[Dict[str, Any]],
    style: StyleConfig,
    out_path: Path,
) -> None:
    w_px = int(meta["source_width_px"])
    h_px = int(meta["source_height_px"])

    width_pt = mm_to_pt(style.max_width_mm)
    height_pt = width_pt * (h_px / float(w_px))
    sx = width_pt / float(w_px)
    sy = height_pt / float(h_px)

    # use pt as user units: viewBox is in pt units
    svg_w_mm = style.max_width_mm
    svg_h_mm = style.max_width_mm * (h_px / float(w_px))

    def px_to_pt_x(x: float) -> float:
        return float(x) * sx

    def px_to_pt_y(y: float) -> float:
        return float(y) * sy

    # Build SVG without grouping
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{svg_w_mm:.3f}mm" height="{svg_h_mm:.3f}mm" '
        f'viewBox="0 0 {width_pt:.3f} {height_pt:.3f}">'
    )

    # NOTE: No <g> groups, per requirement.

    for obj in objects:
        t = obj.get("type")
        if t == "line":
            x1 = px_to_pt_x(obj["x1"])
            y1 = px_to_pt_y(obj["y1"])
            x2 = px_to_pt_x(obj["x2"])
            y2 = px_to_pt_y(obj["y2"])

            role = obj.get("role", "main")
            if role == "dash":
                sw = style.stroke_dash_pt
                dash = style.dash_array_default_pt
            elif role == "axis":
                sw = style.stroke_axis_pt
                dash = None
            else:
                sw = style.stroke_main_pt
                dash = None

            if dash:
                lines.append(
                    f'<line id="{obj["id"]}" x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
                    f'stroke="{style.stroke_color}" stroke-width="{sw:.3f}" stroke-linecap="butt" '
                    f'stroke-dasharray="{dash}"/>'
                )
            else:
                lines.append(
                    f'<line id="{obj["id"]}" x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
                    f'stroke="{style.stroke_color}" stroke-width="{sw:.3f}" stroke-linecap="butt"/>'
                )

        elif t == "rect":
            x = px_to_pt_x(obj["x"])
            y = px_to_pt_y(obj["y"])
            ww = px_to_pt_x(obj["w"])
            hh = px_to_pt_y(obj["h"])
            fill = obj.get("fill", "#bfbfbf")
            # bars often have no explicit border in textbook, but OK to keep stroke off or on:
            # We keep stroke OFF by default to avoid double-stroking if bar outline already detected as lines.
            lines.append(
                f'<rect id="{obj["id"]}" x="{x:.3f}" y="{y:.3f}" width="{ww:.3f}" height="{hh:.3f}" '
                f'fill="{fill}" stroke="none"/>'
            )

        elif t == "text_box":
            # place text at center
            tx = px_to_pt_x(obj["x"] + obj["w"] / 2.0)
            ty = px_to_pt_y(obj["y"] + obj["h"] / 2.0)
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            fs = style.label_font_size_pt
            # Keep as middle anchor
            # Use dominant-baseline="middle" for center alignment
            # Escape XML
            text_esc = (
                text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
            )
            lines.append(
                f'<text id="{obj["id"]}" x="{tx:.3f}" y="{ty:.3f}" '
                f'font-family="{style.font_family}" font-size="{fs:.3f}" '
                f'text-anchor="middle" dominant-baseline="middle" fill="{style.stroke_color}">{text_esc}</text>'
            )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def export_debug_png(gray: np.ndarray, objects: List[Dict[str, Any]], out_path: Path) -> None:
    # Draw detection overlay for debugging
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for obj in objects:
        if obj["type"] == "line":
            p1 = (safe_int(obj["x1"]), safe_int(obj["y1"]))
            p2 = (safe_int(obj["x2"]), safe_int(obj["y2"]))
            cv2.line(img, p1, p2, (0, 0, 255), 1)
        elif obj["type"] == "rect":
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 200, 0), 1)
        elif obj["type"] == "text_box":
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
            if obj.get("text"):
                cv2.putText(img, obj["id"], (x, max(10, y-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,0,0), 1)
    cv2.imwrite(str(out_path), img)


def process_one_image(
    img_path: Path,
    cfg: AppConfig,
    job_id: str,
    item_idx0: int,
    out_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # read
    bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to read image")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    meta = {"source_image": img_path.name, "source_width_px": w, "source_height_px": h}

    # step: binarize
    STORE.step_start(job_id, item_idx0, "binarize", "adaptiveThreshold")
    mask = binarize(gray, cfg.detect.adaptive_block_size, cfg.detect.adaptive_C)
    STORE.step_done(job_id, item_idx0, "binarize")

    # step: filled rects
    STORE.step_start(job_id, item_idx0, "detect_filled_rects", "connected components + distTransform")
    rects, filled_mask = detect_filled_rects(gray, mask, cfg.detect)
    STORE.step_done(job_id, item_idx0, "detect_filled_rects", f"rects={len(rects)}")

    # line mask = remove filled regions (bars)
    mask_lines = cv2.bitwise_and(mask, cv2.bitwise_not(filled_mask))
    stroke_width = estimate_stroke_width_px(mask_lines)
    kernel_size = max(1, int(round(stroke_width / 2.0)))
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask_lines = cv2.morphologyEx(mask_lines, cv2.MORPH_OPEN, kernel)
        mask_lines = cv2.morphologyEx(mask_lines, cv2.MORPH_CLOSE, kernel)
        stroke_width = estimate_stroke_width_px(mask_lines)

    # step: lines
    STORE.step_start(job_id, item_idx0, "detect_lines", "skeleton + HoughLinesP + TLS + merge")
    segs = detect_lines_from_skeleton(mask_lines, cfg.detect, stroke_width)
    STORE.step_done(job_id, item_idx0, "detect_lines", f"lines={len(segs)}")

    # step: dashed classify
    STORE.step_start(job_id, item_idx0, "classify_dash", "sample along lines")
    line_objs: List[Dict[str, Any]] = []
    for i, (p1, p2) in enumerate(segs, start=1):
        is_dash = detect_dashed(mask_lines, p1, p2, cfg.detect.dash_noise_run_px, cfg.detect.dash_min_switches)
        role = "dash" if is_dash else "main"
        line_objs.append({
            "type": "line",
            "id": f"line_{i:04d}",
            "x1": float(p1[0]),
            "y1": float(p1[1]),
            "x2": float(p2[0]),
            "y2": float(p2[1]),
            "role": role,
            "dash": cfg.style.dash_array_default_pt if is_dash else None,
        })
    STORE.step_done(job_id, item_idx0, "classify_dash")

    # step: text boxes
    STORE.step_start(job_id, item_idx0, "detect_text_boxes", "remove lines & filled, then CC")
    text_boxes = detect_text_boxes(mask, segs, filled_mask, cfg.detect)
    STORE.step_done(job_id, item_idx0, "detect_text_boxes", f"text_boxes={len(text_boxes)}")

    # OCR
    if cfg.llm.enabled and cfg.llm.use_llm_ocr and text_boxes:
        STORE.step_start(job_id, item_idx0, "llm_ocr", "collage vision OCR (1 call per image)")
        try:
            client = OpenAICompatClient(cfg.llm.base_url, cfg.llm.api_key, timeout=cfg.llm.timeout_sec)
            collage_png, ids = make_ocr_collage(gray, text_boxes, pad_px=cfg.detect.text_pad_px)
            if collage_png:
                prompt = (
                    "你将看到一个拼图，里面有多个小块，每个小块左上角都有红色id（例如 text_0001）。\n"
                    "请逐块识别文字，只输出严格 JSON 对象：\n"
                    "{ \"text_0001\": \"...\", \"text_0002\": \"...\" }\n"
                    "要求：\n"
                    "1) 保持原始字符顺序与内容（中文/英文/数字）。\n"
                    "2) 若某块没有文字，值返回空字符串。\n"
                    "3) 只返回 JSON，不要解释。\n"
                )
                STORE.inc_item_llm_calls(job_id, item_idx0, 1)
                resp = client.vision_chat(cfg.llm.model, prompt, collage_png)
                data = extract_first_json_object(resp) or {}
                # fill text
                for tb in text_boxes:
                    tbid = tb["id"]
                    if tbid in data and isinstance(data[tbid], str):
                        tb["text"] = data[tbid].strip()
            STORE.step_done(job_id, item_idx0, "llm_ocr")
        except Exception as e:
            STORE.step_error(job_id, item_idx0, "llm_ocr", str(e))
            # allow continue with empty texts
            STORE.step_done(job_id, item_idx0, "llm_ocr", "OCR failed, keep empty text")

    # NOTE: user requested NO arrowheads. We do not output arrow markers/shapes at all.

    objects = []
    objects.extend(rects)
    objects.extend(line_objs)
    objects.extend(text_boxes)

    # Export JSON + SVG + debug
    out_json = out_dir / f"{item_idx0+1:03d}.json"
    out_svg = out_dir / f"{item_idx0+1:03d}.svg"
    out_dbg = out_dir / f"{item_idx0+1:03d}_debug.png"

    payload = {"meta": meta, "objects": objects}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    export_svg(meta, objects, cfg.style, out_svg)
    export_debug_png(gray, objects, out_dbg)

    return meta, objects


# -----------------------------
# FastAPI app
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
JOBS_DIR = APP_DIR / "jobs"
ensure_dir(JOBS_DIR)

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    p = STATIC_DIR / "index.html"
    return p.read_text(encoding="utf-8")


@app.post("/api/providers/list_models")
def api_list_models(base_url: str = Form(...), api_key: str = Form(...), timeout_sec: int = Form(30)):
    try:
        client = OpenAICompatClient(base_url, api_key, timeout=int(timeout_sec))
        models = client.list_models()
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/jobs")
async def api_create_job(
    config_json: str = Form(...),
    files: List[UploadFile] = File(...),
):
    try:
        cfg = AppConfig.model_validate_json(config_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config_json: {e}")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # create job directory
    filenames = [f.filename for f in files]
    job = STORE.create_job(cfg, filenames)
    job_dir = JOBS_DIR / job.job_id
    in_dir = job_dir / "input"
    out_dir = job_dir / "output"
    ensure_dir(in_dir)
    ensure_dir(out_dir)

    # save config for traceability
    (job_dir / "config.json").write_text(config_json, encoding="utf-8")

    # save files
    for f in files:
        content = await f.read()
        (in_dir / f.filename).write_bytes(content)

    STORE.add_job_log(job.job_id, f"[{time.strftime('%H:%M:%S')}] Job created. files={len(files)}")

    # background thread processing
    def worker():
        try:
            for idx0, item in enumerate(STORE.get(job.job_id).items):
                STORE.set_item_status(job.job_id, idx0, "processing")
                STORE.add_item_log(job.job_id, idx0, f"[{time.strftime('%H:%M:%S')}] Start {item.filename}")

                img_path = in_dir / item.filename
                try:
                    process_one_image(img_path, cfg, job.job_id, idx0, out_dir)
                    STORE.set_item_status(job.job_id, idx0, "done")
                    STORE.set_item_outputs(job.job_id, idx0, {
                        "svg": f"/api/jobs/{job.job_id}/files/output/{idx0+1:03d}.svg",
                        "json": f"/api/jobs/{job.job_id}/files/output/{idx0+1:03d}.json",
                        "debug": f"/api/jobs/{job.job_id}/files/output/{idx0+1:03d}_debug.png",
                    })
                    STORE.add_item_log(job.job_id, idx0, f"[{time.strftime('%H:%M:%S')}] Done")
                except Exception as e:
                    STORE.set_item_status(job.job_id, idx0, "error", error=str(e))
                    STORE.add_item_log(job.job_id, idx0, f"[{time.strftime('%H:%M:%S')}] ERROR: {e}")
                    # continue to next file (sequential output, but does not stop the whole batch)
                    continue

            # finalize
            st = STORE.get(job.job_id)
            any_err = any(it.status == "error" for it in st.items)
            STORE.finalize_job(job.job_id, "done" if not any_err else "error", error=("Some items failed" if any_err else None))
            STORE.add_job_log(job.job_id, f"[{time.strftime('%H:%M:%S')}] Job finished: {STORE.get(job.job_id).status}")
        except Exception as e:
            STORE.finalize_job(job.job_id, "error", error=str(e))
            STORE.add_job_log(job.job_id, f"[{time.strftime('%H:%M:%S')}] Job crashed: {e}")

    Thread(target=worker, daemon=True).start()

    return {"ok": True, "job_id": job.job_id}


@app.get("/api/jobs/{job_id}")
def api_job_status(job_id: str):
    try:
        st = STORE.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")

    # serialize
    return JSONResponse({
        "job_id": st.job_id,
        "status": st.status,
        "created_at": st.created_at,
        "started_at": st.started_at,
        "finished_at": st.finished_at,
        "config_summary": st.config_summary,
        "logs": st.logs[-400:],
        "error": st.error,
        "items": [
            {
                "index": it.index,
                "filename": it.filename,
                "status": it.status,
                "current_step": it.current_step,
                "steps": it.steps,
                "outputs": it.outputs,
                "error": it.error,
                "logs": it.logs[-400:],
                "llm_calls": it.llm_calls,
            }
            for it in st.items
        ],
    })


@app.get("/api/jobs/{job_id}/files/{relpath:path}")
def api_job_file(job_id: str, relpath: str):
    p = (JOBS_DIR / job_id / relpath).resolve()
    if not str(p).startswith(str((JOBS_DIR / job_id).resolve())):
        raise HTTPException(status_code=400, detail="invalid path")
    if not p.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(p))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
