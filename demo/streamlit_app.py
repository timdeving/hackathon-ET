"""The live demo, on Streamlit Community Cloud: upload a clip from the junction camera and get its
traffic events and accident-risk curve back, drawn on the video.

It runs the submission's own code on the CPU with the demo profile (demo/pipeline.py). The
ready-made results for one-minute clips of the sample videos (demo/make_examples.py) are read
from the project's media dataset on Hugging Face, so they show at once.

    streamlit run demo/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

os.environ.setdefault("WIUT_PROFILE", "demo")  # before anything reads the settings
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the repository's root

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from demo.make_examples import EXAMPLES  # noqa: E402
from demo.pipeline import MAX_SECONDS, check, process  # noqa: E402
from src import part_a  # noqa: E402
from src.visualize import EVENT_COLOURS, OTHER_EVENT_COLOUR  # noqa: E402

MEDIA = "https://huggingface.co/datasets/akmaloio/wiut-traffic-media/resolve/main/"
REPO = "https://github.com/timdeving/hackathon-ET"
MAX_SIZE = (1920, 1080)  # 1080p: 4K clips take too long and too much memory on this free CPU
MAX_UPLOAD_MB = 500  # 2 minutes of 1080p at up to about 30 Mb/s
MINUTES_PER_MINUTE = "15-20"  # processing time per minute of 1080p video on 2 CPU cores

INTRO = """
Upload a clip from the hackathon's junction camera. The system finds its traffic events
(pedestrians outside the crossings, vehicles not yielding at a crossing, red-light running,
stopping past the stop line, crossing a solid line) and scores, frame by frame, the risk that an
accident starts within 5 seconds. You get back the events, their timeline, the risk curve and
the video with everything drawn on it.
"""
WARNING = f"""
**This free demo runs on a shared CPU, so it is slow:** about {MINUTES_PER_MINUTE} minutes per
minute of video. Keep this page open and don't click anything while it runs. On a GPU the same
system handles a video faster than it plays: to run it there, follow the
[repository]({REPO})'s README. The ready-made results show at once.

**Accepted:** an `.mp4` from this camera, **1080p (1920 x 1080) or smaller, up to
{MAX_SECONDS / 60:.0f} minutes.** A video from another camera runs too, but the system's map
of this junction won't fit it, so its events will be wrong.
"""


@st.cache_resource
def one_at_a_time() -> threading.Lock:
    """Load the model once per process; the lock lets one clip be processed at a time."""
    part_a.load_models()
    return threading.Lock()


@st.cache_data(show_spinner=False)
def example(folder: str) -> dict:
    """A ready-made result: its predictions and facts, from the media dataset."""
    base = f"{MEDIA}demo_examples/{folder}/"
    with urllib.request.urlopen(base + "predictions.json", timeout=30) as response:
        predictions = response.read()
    with urllib.request.urlopen(base + "meta.json", timeout=30) as response:
        meta = json.loads(response.read())
    entry = next(iter(json.loads(predictions)["videos"].values()))
    return {"video": base + "annotated.mp4", "events": entry["events"], "risk": entry["risk"],
            "duration": meta["duration"], "predictions": predictions, "seconds": None}


def hex_colour(bgr: tuple[int, int, int]) -> str:
    blue, green, red = bgr
    return f"#{red:02x}{green:02x}{blue:02x}"


def timeline_figure(events: list[list], duration: float) -> go.Figure:
    """One row per class, one bar per event."""
    figure = go.Figure()
    for label in sorted({label for _, _, label in events}):
        spans = [(start, end) for start, end, name in events if name == label]
        figure.add_bar(
            y=[label] * len(spans), x=[end - start for start, end in spans],
            base=[start for start, _ in spans], orientation="h", name=label,
            marker_color=hex_colour(EVENT_COLOURS.get(label, OTHER_EVENT_COLOUR)),
            hovertemplate="from %{base:.1f} s, %{x:.1f} s long<extra>" + label + "</extra>",
        )
    figure.update_layout(
        title="Events", xaxis_title="seconds", xaxis_range=[0, duration], showlegend=False,
        height=120 + 40 * max(1, len(figure.data)), margin={"l": 10, "r": 10, "t": 40, "b": 40},
    )
    return figure


def risk_figure(risk: list[list]) -> go.Figure:
    """Part B's score over time, with the alarm threshold."""
    figure = go.Figure(go.Scatter(x=[t for t, _ in risk], y=[s for _, s in risk], mode="lines",
                                  line={"color": "#d62728"}, name="risk"))
    figure.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="alarm")
    figure.update_layout(title="Risk that an accident starts within 5 s", xaxis_title="seconds",
                         yaxis_range=[0, 1], height=260,
                         margin={"l": 10, "r": 10, "t": 40, "b": 40})
    return figure


def show(result: dict, key: str) -> None:
    """The annotated video, a summary, the timeline, the risk curve, the events and a download."""
    events, duration = result["events"], result["duration"]
    st.video(result["video"])
    counts: dict[str, int] = {}
    for _, _, label in events:
        counts[label] = counts.get(label, 0) + 1
    found = ", ".join(f"{n} x `{label}`" for label, n in sorted(counts.items())) or "no events"
    took = f" Processed in {result['seconds'] / 60:.1f} minutes." if result["seconds"] else ""
    st.markdown(f"**{duration:.0f} s of video: {found}.**{took}")
    st.plotly_chart(timeline_figure(events, duration), width="stretch", key=key + "t")
    st.plotly_chart(risk_figure(result["risk"]), width="stretch", key=key + "r")
    st.dataframe(
        [{"event": label, "start (s)": round(start, 1), "end (s)": round(end, 1),
          "length (s)": round(end - start, 1)} for start, end, label in sorted(events)],
        width="stretch", hide_index=True,
    )
    st.download_button("Download the results (predictions.json)", result["predictions"],
                       file_name="predictions.json", mime="application/json", key=key + "d")


def analyse_upload(uploaded) -> dict | None:
    """Check the uploaded clip, process it one clip at a time, and return what show() needs;
    None, with the reason shown, if the clip can't be used or processing fails."""
    with tempfile.TemporaryDirectory() as folder:
        clip = Path(folder) / "clip.mp4"
        clip.write_bytes(uploaded.getbuffer())
        try:
            info = check(clip)
        except ValueError as error:
            st.error(str(error))
            return None
        if info.width > MAX_SIZE[0] or info.height > MAX_SIZE[1]:
            st.error(f"The clip is {info.width} x {info.height}. Please upload 1080p "
                     f"({MAX_SIZE[0]} x {MAX_SIZE[1]}) or smaller.")
            return None
        lock = one_at_a_time()
        if lock.locked():
            st.info("Another visitor's clip is being processed; yours starts when it finishes.")
        with lock:
            bar = st.progress(0.0, text="Starting")
            try:
                result = process(clip, Path(folder) / "work",
                                 lambda fraction, text: bar.progress(min(fraction, 1.0), text=text))
            except Exception as error:  # never crash the page; say what went wrong
                st.error(f"Processing failed: {error}")
                return None
        return {"video": result.video.read_bytes(), "events": result.events,
                "risk": result.risk, "duration": result.info.duration,
                "predictions": result.predictions.read_bytes(), "seconds": result.seconds}


def main() -> None:
    st.set_page_config(page_title="Traffic events: live demo", page_icon="🚦", layout="wide")
    st.title("Traffic events from a junction camera: live demo")
    st.markdown(INTRO)
    st.warning(WARNING)
    one_at_a_time()  # load the model before the first click
    ready, upload = st.tabs(["Ready-made results (instant)", "Upload your clip"])
    with ready:
        titles = {folder: title for folder, _, _, title in EXAMPLES}
        folder = st.selectbox("A one-minute clip of a sample video", list(titles),
                              format_func=titles.get)
        show(example(folder), key="example")
    with upload:
        uploaded = st.file_uploader(f"Your clip: .mp4, 1080p, up to {MAX_SECONDS / 60:.0f} min",
                                    type=["mp4"], max_upload_size=MAX_UPLOAD_MB)
        if uploaded is not None and st.button("Find the events", type="primary"):
            st.session_state["result"] = analyse_upload(uploaded)
        if st.session_state.get("result"):
            show(st.session_state["result"], key="upload")


main()
