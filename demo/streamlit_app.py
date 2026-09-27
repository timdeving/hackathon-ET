"""The live demo, on Streamlit Community Cloud: upload a clip from the junction camera and get its
traffic events and accident risk back, drawn on the video and laid out on a timeline.

It runs the submission's own code on the CPU with the demo profile (demo/pipeline.py); the
results viewer is demo/viewer.py. Ready-made results for one-minute clips of the sample videos
(demo/make_examples.py) come from the project's media dataset on Hugging Face.

    streamlit run demo/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("WIUT_PROFILE", "demo")  # before anything reads the settings
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the repository's root

import streamlit as st  # noqa: E402

from demo.make_examples import EXAMPLES  # noqa: E402
from demo.pipeline import MAX_SECONDS, check, process  # noqa: E402
from demo.viewer import HEIGHT, alarms, hex_colour, viewer_html  # noqa: E402
from src import part_a  # noqa: E402
from src.visualize import EVENT_COLOURS, OTHER_EVENT_COLOUR  # noqa: E402

MEDIA = "https://huggingface.co/datasets/akmaloio/wiut-traffic-media/resolve/main/"
WEBSITE = "https://akmaloio-wiut-traffic.static.hf.space"
REPO = "https://github.com/timdeving/hackathon-ET"
MAX_SIZE = (1920, 1080)  # 1080p: 4K clips take too long and too much memory on this free CPU
MAX_UPLOAD_MB = 500  # 2 minutes of 1080p at up to about 30 Mb/s
MINUTES_PER_MINUTE = (15, 20)  # processing time per minute of video on the demo's 2 CPU cores

STYLE = """<style>
.chips { margin: -4px 0 6px; }
.chip { display: inline-block; padding: 1px 10px; margin: 0 6px 6px 0; border-radius: 999px;
        color: #fff; font-weight: 600; font-size: 0.85rem; }
</style>"""
INTRO = f"""
The system watches a clip from the hackathon's junction camera and reports its traffic events:
pedestrians on the road outside a crossing, vehicles not yielding at a crossing, red-light
running, stopping past the stop line, crossing a solid line. Frame by frame, it also scores the
risk that an accident starts within 5 seconds. **Start with a ready-made result, or upload your
own clip.** More on the [website]({WEBSITE}).
"""
ABOUT = f"""
#### What happens to a clip
1. Every frame is decoded; every 3rd one goes through the YOLO26m detector, and ByteTrack
   follows each road user.
2. The clip's view is aligned onto our map of the junction (the camera's aim differs a little
   between recordings), and the traffic light is read from the picture.
3. One rule per event type turns the tracks into timed events. The accident risk comes from
   the time to collision of every pair of road users with a moving vehicle.
4. The results are drawn on the video: boxes, the events under way, the light as read (top
   right) and the risk (bottom left).

#### Limits of this demo
- **Speed:** it runs on a free, shared CPU: about {MINUTES_PER_MINUTE[0]}-{MINUTES_PER_MINUTE[1]}
  minutes per minute of video, one clip at a time. On a GPU the same system is faster than the
  video plays: see the [repository]({REPO}).
- **Accepted:** `.mp4`, **1080p (1920 x 1080) or smaller, up to {MAX_SECONDS / 60:.0f} minutes**,
  up to {MAX_UPLOAD_MB} MB.
- **This camera only:** the events come from our map of this junction. A video from another
  camera runs, but its events will be wrong.
- Uploaded clips are processed and discarded; nothing is kept.
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
            "duration": meta["duration"], "lights": meta.get("lights", []),
            "predictions": predictions, "seconds": None}


def chip(label: str) -> str:
    colour = hex_colour(EVENT_COLOURS.get(label, OTHER_EVENT_COLOUR))
    return f'<span class="chip" style="background:{colour}">{label}</span>'


def summary(result: dict) -> None:
    """The headline numbers, then one chip per event type with its count."""
    events, risk = result["events"], result["risk"]
    columns = st.columns(4)
    columns[0].metric("Events", len(events), border=True)
    columns[1].metric("Event types", len({label for _, _, label in events}), border=True)
    columns[2].metric("Highest accident risk", f"{max((s for _, s in risk), default=0):.2f}",
                      border=True)
    columns[3].metric("Alarms (risk 0.5 or more)", len(alarms(risk)), border=True)
    counts: dict[str, int] = {}
    for _, _, label in events:
        counts[label] = counts.get(label, 0) + 1
    chips = " ".join(f"{chip(label)}&times;{n}" for label, n in sorted(counts.items()))
    st.markdown(f'<div class="chips">{chips or "No events in this clip."}</div>',
                unsafe_allow_html=True)


def show(result: dict, key: str) -> None:
    """The summary, the interactive viewer, and the results to download."""
    summary(result)
    st.iframe(viewer_html(result["video"], result["events"], result["risk"],
                          result["duration"], result["lights"]), height=HEIGHT)
    took = f"Processed in {result['seconds'] / 60:.1f} minutes. " if result["seconds"] else ""
    st.caption(f"{took}The events and the risk on every frame, in the harness's format:")
    st.download_button("Download predictions.json", result["predictions"],
                       file_name="predictions.json", mime="application/json", key=key)


def saved(uploaded) -> Path:
    """The upload as a file on disk, written once per upload; the previous one is deleted."""
    state = st.session_state
    if state.get("upload_id") != uploaded.file_id:
        if state.get("upload_path"):
            Path(state["upload_path"]).unlink(missing_ok=True)
        handle, path = tempfile.mkstemp(suffix=".mp4")
        with os.fdopen(handle, "wb") as file:
            file.write(uploaded.getbuffer())
        state["upload_id"], state["upload_path"] = uploaded.file_id, path
        state.pop("result", None)
    return Path(state["upload_path"])


def refusal(info) -> str | None:
    """Why the demo can't take this clip, or None."""
    if info.width > MAX_SIZE[0] or info.height > MAX_SIZE[1]:
        return (f"The clip is {info.width} x {info.height}. Please upload 1080p "
                f"({MAX_SIZE[0]} x {MAX_SIZE[1]}) or smaller.")
    return None


def analyse(clip: Path) -> dict | None:
    """Process one clip, one at a time, with a progress bar that estimates the time left."""
    lock = one_at_a_time()
    if lock.locked():
        st.info("Another visitor's clip is being processed; yours starts when it finishes.")
    with lock:
        bar = st.progress(0.0, text="Starting")
        start = time.monotonic()

        def update(fraction: float, text: str) -> None:
            elapsed = time.monotonic() - start
            left = ""
            if fraction > 0.05:
                left = f" · about {elapsed / fraction * (1 - fraction) / 60:.0f} min left"
            bar.progress(min(fraction, 1.0), text=f"{text} · {elapsed / 60:.0f} min so far{left}")

        try:
            with tempfile.TemporaryDirectory() as work:
                result = process(clip, work, update)
                bar.empty()
                return {"video": result.video.read_bytes(), "events": result.events,
                        "risk": result.risk, "duration": result.info.duration,
                        "lights": result.lights, "seconds": result.seconds,
                        "predictions": result.predictions.read_bytes()}
        except Exception as error:  # never crash the page; say what went wrong
            st.error(f"Processing failed: {error}")
            return None


def upload_tab() -> None:
    low, high = MINUTES_PER_MINUTE
    st.markdown(f"**Accepted:** an `.mp4` from this camera, **1080p or smaller, up to "
                f"{MAX_SECONDS / 60:.0f} minutes**. Processing runs on a free CPU: about "
                f"{low}-{high} minutes per minute of video. Keep this tab open while it runs.")
    uploaded = st.file_uploader("Your clip", type=["mp4"], max_upload_size=MAX_UPLOAD_MB)
    if uploaded is None:
        return
    clip = saved(uploaded)
    try:
        info = check(clip)
    except ValueError as error:
        st.error(str(error))
        return
    problem = refusal(info)
    if problem:
        st.error(problem)
        return
    minutes = info.duration / 60
    st.info(f"**{uploaded.name}**: {info.duration:.0f} s, {info.width} x {info.height}, "
            f"{info.fps:.2f} fps, {uploaded.size / 1e6:.0f} MB. Processing will take about "
            f"{max(1, round(minutes * low))}-{max(2, round(minutes * high))} minutes.")
    if st.button("Find the events", type="primary"):
        st.session_state["result"] = analyse(clip)
    if st.session_state.get("result"):
        show(st.session_state["result"], key="upload")


def main() -> None:
    st.set_page_config(page_title="Traffic events: live demo", page_icon="🚦", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("🚦 Traffic events from a junction camera")
    st.caption(f"Live demo · WIUT Hackathon 2026, Computer Vision track · [website]({WEBSITE}) · "
               f"[code]({REPO})")
    st.markdown(INTRO)
    one_at_a_time()  # load the model before the first click
    ready, upload, about = st.tabs(["Ready-made results", "Upload your clip", "About"])
    with ready:
        titles = {folder: title for folder, _, _, title in EXAMPLES}
        folder = st.selectbox("A one-minute clip of a sample video, processed by the system",
                              list(titles), format_func=titles.get)
        show(example(folder), key="example")
    with upload:
        upload_tab()
    with about:
        st.markdown(ABOUT)


main()
