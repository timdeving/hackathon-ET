"""The live demo, a Hugging Face Space: upload a clip from the junction camera and get its traffic
events and accident-risk curve back, drawn on the video.

It runs the submission's own code on the Space's CPU with the demo profile (demo/pipeline.py).
Ready-made results for clips of the sample videos show instantly (demo/examples/, made by
demo/make_examples.py).

    python demo/app.py        # then open http://localhost:7860
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("WIUT_PROFILE", "demo")  # before anything reads the settings
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the repository's root

import gradio as gr  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from demo.pipeline import MAX_SECONDS, process  # noqa: E402
from src import part_a  # noqa: E402
from src.visualize import EVENT_COLOURS, OTHER_EVENT_COLOUR  # noqa: E402

log = logging.getLogger("demo")
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
REPO_URL = "https://github.com/timdeving/hackathon-ET"
MINUTES_PER_MINUTE = 16  # processing time per minute of 4K video on this Space's 2 CPU cores

INTRO = f"""
# Traffic events from a junction camera: live demo

Upload a clip from the hackathon's junction camera. The system finds its traffic events
(pedestrians outside the crossings, vehicles not yielding at a crossing, red-light running,
stopping past the stop line, crossing a solid line) and scores, frame by frame, the risk that an
accident starts within 5 seconds. You get back the events, their timeline, the risk curve and
the video with everything drawn on it.

> **This free demo runs on 2 CPU cores, so it is slow:** about {MINUTES_PER_MINUTE} minutes per
> minute of video, so a 2-minute clip takes about half an hour. Keep this page open while it
> runs. On a GPU the same system handles a video faster than it plays: to run it there, follow
> the [repository]({REPO_URL})'s README. The ready-made results below show instantly.

**Accepted:** an `.mp4` from this camera, up to {MAX_SECONDS / 60:.0f} minutes and 1 GB: 4K
like the original recordings, or scaled down (1080p works). A video from another camera runs
too, but the system's map of this junction won't fit it, so its events will be wrong.
"""


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
            hovertemplate="%{base:.1f}–%{x:.1f} s later<extra>" + label + "</extra>",
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


def events_table(events: list[list]) -> list[list]:
    return [[label, round(start, 1), round(end, 1), round(end - start, 1)]
            for start, end, label in sorted(events)]


def summary(events: list[list], duration: float, seconds: float | None) -> str:
    counts = {}
    for _, _, label in events:
        counts[label] = counts.get(label, 0) + 1
    found = ", ".join(f"{n} × `{label}`" for label, n in sorted(counts.items())) or "no events"
    took = f" Processed in {seconds / 60:.1f} minutes." if seconds is not None else ""
    return f"**{duration:.0f} s of video: {found}.**{took}"


def outputs(video, events, risk, duration, predictions, seconds=None):
    return (str(video), summary(events, duration, seconds), timeline_figure(events, duration),
            risk_figure(risk), events_table(events), str(predictions))


def run_upload(file, progress=gr.Progress()):  # noqa: B008  (how Gradio asks for a bar)
    if file is None:
        raise gr.Error("Choose an .mp4 file first.")
    path = file if isinstance(file, str) else file.name
    workdir = tempfile.mkdtemp(prefix="demo_")
    try:
        result = process(path, workdir, lambda fraction, text: progress(fraction, desc=text))
    except ValueError as error:  # the clip can't be used: say why
        raise gr.Error(str(error)) from error
    except Exception as error:  # never crash the page; the log keeps the details
        log.exception("processing %s failed", path)
        raise gr.Error(f"Processing failed: {error}") from error
    return outputs(result.video, result.events, result.risk, result.info.duration,
                   result.predictions, result.seconds)


def example_names() -> list[str]:
    return sorted(p.name for p in EXAMPLES_DIR.glob("*") if (p / "meta.json").exists())


def show_example(name: str | None):
    if not name:
        raise gr.Error("Choose a sample clip.")
    folder = EXAMPLES_DIR / name
    video = json.loads((folder / "predictions.json").read_text())["videos"]
    entry = next(iter(video.values()))
    duration = json.loads((folder / "meta.json").read_text())["duration"]
    return outputs(folder / "annotated.mp4", entry["events"], entry["risk"], duration,
                   folder / "predictions.json")


def build() -> gr.Blocks:
    with gr.Blocks(title="Traffic events: live demo") as demo:
        gr.Markdown(INTRO)
        with gr.Tabs():
            with gr.Tab("Ready-made results (instant)"):
                names = example_names()
                choice = gr.Dropdown(names, value=names[0] if names else None,
                                     label="A clip of a sample video")
                show = gr.Button("Show the results", variant="primary")
            with gr.Tab("Upload your clip"):
                upload = gr.File(label=f"Your clip (.mp4, up to {MAX_SECONDS / 60:.0f} min)",
                                 file_types=[".mp4"])
                start = gr.Button("Find the events", variant="primary")
        video = gr.Video(label="The clip, annotated", interactive=False)
        text = gr.Markdown()
        timeline = gr.Plot(label="Timeline")
        risk = gr.Plot(label="Accident risk")
        table = gr.Dataframe(headers=["event", "start (s)", "end (s)", "length (s)"],
                             label="Events", interactive=False)
        download = gr.File(label="The results as predictions.json")
        results = [video, text, timeline, risk, table, download]
        show.click(show_example, inputs=choice, outputs=results)
        start.click(run_upload, inputs=upload, outputs=results)
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    part_a.load_models()  # once, before the first visitor
    app = build()
    app.queue(default_concurrency_limit=1, max_size=20)  # one clip at a time on 2 cores
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
               max_file_size="1gb", allowed_paths=[str(EXAMPLES_DIR), tempfile.gettempdir()])
