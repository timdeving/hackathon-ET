"""The demo's results viewer: the annotated video, the events under way, the light as read and
the accident risk, a clickable list of the events, the timeline and the risk curve, all following
the video; any click seeks it. One HTML page with its own script, shown in the Streamlit app
through st.iframe; the website's viewer (website/app.js) for a single clip. The
page is demo/viewer.html; viewer_html() fills in the clip's data.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from src.visualize import EVENT_COLOURS, OTHER_EVENT_COLOUR

PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-basic-dist-min@2.35.2/plotly-basic.min.js"
TEMPLATE = (Path(__file__).resolve().parent / "viewer.html").read_text(encoding="utf-8")
HEIGHT = 1180  # the component's height in the page, in pixels
RISK_STEP_SEC = 0.5  # the risk curve drawn: its highest value in each step
ALARM = 0.5  # the metric's alarm threshold (evaluate.py)


def hex_colour(bgr: tuple[int, int, int]) -> str:
    blue, green, red = bgr
    return f"#{red:02x}{green:02x}{blue:02x}"


def risk_steps(risk: list[list]) -> list[list]:
    """[[t_sec, score], ...] per frame -> the highest score in each RISK_STEP_SEC step."""
    steps: dict[int, float] = {}
    for t_sec, score in risk:
        step = int(t_sec // RISK_STEP_SEC)
        steps[step] = max(steps.get(step, 0.0), score)
    return [[round(step * RISK_STEP_SEC, 2), round(score, 3)]
            for step, score in sorted(steps.items())]


def alarms(risk: list[list]) -> list[list]:
    """Runs of frames at or above the alarm threshold; runs under 2 s apart merge (evaluate.py)."""
    runs: list[list] = []
    start = last = None
    for t_sec, score in risk:
        if score >= ALARM:
            start = t_sec if start is None else start
            last = t_sec
        elif start is not None:
            runs.append([start, last])
            start = None
    if start is not None:
        runs.append([start, last])
    merged: list[list] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] < 2.0:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return merged


def viewer_html(video: bytes | str, events: list[list], risk: list[list], duration: float,
                lights: list[list] | None = None) -> str:
    """The viewer for one clip. video: its annotated version, as a URL or the file's bytes;
    events: [[start, end, label], ...]; risk: [[t_sec, score], ...]; lights: the top road's
    phases, [[start, end, "red" | "amber" | "green" | "unknown"], ...]."""
    source = {"url": video} if isinstance(video, str) else {
        "base64": base64.b64encode(video).decode("ascii")}
    data = {
        "video": source, "events": events, "risk": risk_steps(risk), "duration": duration,
        "lights": lights or [], "alarm": ALARM,
        "colours": {label: hex_colour(bgr) for label, bgr in EVENT_COLOURS.items()},
        "other": hex_colour(OTHER_EVENT_COLOUR),
    }
    # "</" can't appear inside a script element: escape it in the data
    payload = json.dumps(data).replace("</", "<\\/")
    return TEMPLATE.replace("__PLOTLY__", PLOTLY).replace("__DATA__", payload)
