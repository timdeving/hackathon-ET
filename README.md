# Traffic events from a road camera: WIUT Hackathon 2026, Computer Vision track

A fixed CCTV camera watches a signalised junction in Tashkent. For every video, this system
produces:

- **Part A, event detection:** timed traffic events, `[start_sec, end_sec, label]`: pedestrians
  crossing outside the crossings, vehicles not yielding at a crossing, running a red light,
  stopping past the stop line, crossing a solid line;
- **Part B, accident anticipation:** for every frame, the probability that an accident starts
  within the next 5 seconds, computed only from the frames seen so far.

It runs offline on one NVIDIA GPU, with the organizers' harness (`run_submission.py`,
`evaluate.py`) unchanged. Everything we built is in `src/`, reached through `solution.py`.

- **Website:** https://akmaloio-wiut-traffic.static.hf.space (approach, data analysis, every sample
  video annotated, the report)
- **Live demo:** https://hackathon-et-sbg32s8s84savvbzganrrk.streamlit.app (upload a clip)
- **Annotated videos:** https://huggingface.co/datasets/akmaloio/wiut-traffic-media

## Results on the sample videos

Four samples (18 minutes, 3840×2160, 29.97 fps). We labelled two of them ourselves (C3902 and
C3905: 71 events) and score with the organizers' `evaluate.py`:

| Class | F1 @ tIoU 0.3 | @ 0.5 | @ 0.7 | Mean |
| --- | ---: | ---: | ---: | ---: |
| `failure_to_yield` | 0.526 | 0.395 | 0.395 | 0.439 |
| `jaywalking` | 0.714 | 0.619 | 0.381 | 0.571 |
| `red_light` | 1.000 | 1.000 | 1.000 | 1.000 |
| `solid_line_crossing` | 0.769 | 0.615 | 0.154 | 0.513 |
| `stop_line` | 1.000 | 1.000 | 1.000 | 1.000 |
| **Score A** | | | | **0.705** |

- **Speed:** Part A and Part B together take 0.65–0.71× each video's duration on an RTX A6000;
  the limit is 3×.
- **Repeatability:** two full runs give identical output, events and risk curves alike.
- **Part B:** the samples hold no accident, so it can't be scored on them. On ordinary traffic
  it raises one alarm (score ≥ 0.5) in 18 minutes: a car driving through pedestrians at a
  crossing.

## Quick start

Needs Python 3.10 or newer and an NVIDIA GPU with driver 525 or newer. `requirements.txt`
installs PyTorch's CUDA 12.6 build, which brings its own CUDA libraries.

```bash
git clone https://github.com/timdeving/hackathon-ET.git
cd hackathon-ET
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_submission.py --videos /path/to/videos --out predictions.json
python evaluate.py --pred predictions.json --validate-only                # format check
python evaluate.py --pred predictions.json --gt data/dev_labels.json      # scores on our labels
```

`--videos` takes a folder of `.mp4` files or one file. Per video the harness prints its
duration, the time budget, the events found and the time taken. The model is in `weights/`, so
nothing is downloaded at run time.

### Docker, offline

```bash
docker build -t wiut .
docker run --rm --gpus all --network none \
  -v /path/to/videos:/data/test:ro -v "$PWD/outputs":/app/outputs \
  wiut python run_submission.py --videos /data/test --out outputs/predictions.json
```

`--network none` proves that the run needs no internet.

### On a CPU: the live demo

The judged run uses the GPU. The same code runs on a CPU when the environment variable
`WIUT_PROFILE=demo` is set. It merges `configs/profiles/demo.yaml` over `configs/params.yaml`,
which moves the detector to the CPU (its FP16 weights then run in float32) and lifts the time
limit:

```bash
WIUT_PROFILE=demo python run_submission.py --videos clips --out predictions.json --time-factor 1000
```

On a 20-second clip of a sample, the CPU run finds the same events and the same risk curve as the
GPU run, but much more slowly: the detector takes about 1.2 s per analysed frame on 2 CPU cores.
Without the variable, `params.yaml` alone applies, as in the judged run.

**The live demo** (`demo/`) is a Streamlit app on this profile: upload a clip of up to 2 minutes
at 1080p and get its events, timeline, risk curve and annotated video back. It runs the same
code; to save time on a CPU, Part B there reuses the detections Part A made on the same frames
instead of running the detector again. Ready-made results for one-minute clips of the samples
come from the project's media dataset on Hugging Face.

```bash
pip install -r demo/requirements.txt          # PyTorch's CPU build, Streamlit, Plotly
streamlit run demo/streamlit_app.py
```

## How it works

```text
video ─► decode every frame (PyAV); every 3rd frame ─► 1280-px working image
      ─► YOLO26m detector ─► ByteTrack tracker            (one pass; lights cut out on the way)
      ─► camera alignment ─► track stitching ─► traffic-light phases ─► per-object features
      ─► one rule per class ─► post-processing ─► events
```

### Part A: events

1. **Perception, one pass.** A background thread decodes every frame, while the GPU runs the
   detector on every 3rd frame (10 per second), shrunk to 1280 px wide. The detector is YOLO26m
   (COCO-pretrained, used as released), exported to TorchScript in FP16 and run with plain
   PyTorch: Ultralytics isn't needed at run time. A ByteTrack tracker follows people, bicycles,
   cars, motorcycles, buses and trucks. The same pass cuts a window around each traffic light
   out of the full-resolution frame, and keeps only its lit-pixel counts.
2. **Camera alignment.** The camera's aim differs between recordings (up to about 140 px at 4K,
   1° of rotation). The median of 40 frames collected during the pass gives the empty road. It
   is matched to the reference picture the scene map is drawn on (`configs/scene/reference.jpg`)
   with SIFT features and a RANSAC homography. Track positions are mapped with it; the frames
   aren't warped.
3. **Scene map** (`configs/scene_map.json`): the road, islands, the three pedestrian crossings,
   the junction, lanes with their direction and allowed exits, the stop line, solid lines, the
   traffic lights, the bus stop and the exits. It is drawn once, in LabelMe
   (`configs/scene/labelme.json`), and built with `tools/build_scene_map.py`.
4. **Track stitching.** It joins the fragments of one object, such as a car hidden behind a bus
   for a few seconds, so its path stays whole.
5. **Traffic lights.** Each vehicle signal head's lit lamp (red, amber or green) is read from
   its position in the head, then cleaned over time, giving each arm a phase on every analysed
   frame.
6. **Rules**, one per class, on each object's ground point in reference pixels. Speeds are in
   the object's own box heights per second, so they mean the same near and far.

   | Class | Rule |
   | --- | --- |
   | `jaywalking` | A pedestrian's feet at least 60 px inside the road and outside every crossing, for 2 s or more. Riders and people seen inside vehicles are skipped |
   | `failure_to_yield` | A moving vehicle's footprint passes over a crossing while a pedestrian is on it; the event runs from its front arriving to its rear leaving |
   | `red_light` | The vehicle's front crosses the stop line after the light has been red for at least 1 s; the event runs until it leaves the junction |
   | `stop_line` | A vehicle stops past the stop line on red, without entering the junction; the event runs until the green |
   | `solid_line_crossing` | The vehicle's ground point crosses a solid line once, from 1.5 s before the crossing to 2 s after it |

7. **Post-processing.** Same-class fragments are merged, sub-second blips dropped, and
   same-class segments never overlap.
8. **Time budget.** Before starting, Part A measures how fast this machine decodes this video
   the way the harness will for Part B. It then gives itself a deadline, so Part A and Part B
   together stay inside the 3× limit.

**Classes predicted.** Score A averages F1 over every class in the test set *and* every class
predicted, so predicting a class the test set lacks adds a zero. We predict only the five classes
above, whose rules score well on our labels. Rules for `stopped_vehicle`, `congestion`,
`wrong_way`, `illegal_turn` and `illegal_u_turn` exist, but are switched off: on our labels none
of them finds a true event. For example, vehicles waiting inside the junction for their green are
queued at a signal, which the `stopped_vehicle` definition excludes, and the U-turns here are
legal. `accident`, `near_miss`, `road_obstacle` and `fire_smoke` aren't predicted. The list
lives in `configs/params.yaml` (`rules.enabled`).

### Part B: accident risk

`RiskEstimator.step(frame, t_sec)` sees only the frames given to it, in order. It never opens the
video and never uses Part A's results. Every 3rd frame it runs its own copy of the detector and
its own tracker. Each track's velocity comes from its last 0.6 s of positions. For every pair
that includes a moving vehicle, it computes when and how close the two would come on their
current courses (their time to collision). The riskiest pair sets the frame's raw risk: about
0.5 when they would meet within 1 s. An exponential average of past values smooths it, so one
noisy frame can't raise an alarm. Boxes cut off by the frame's edge, and the far road at the top
of the picture (where perspective squeezes distances), take no part. Its work is capped at its
share of the time budget; past that it stops analysing and keeps its last score.

### Determinism

Seeds are fixed (Python, NumPy and PyTorch, all 0); cuDNN runs in deterministic mode with
benchmarking off, and TF32 is off. Nothing samples at random. Two runs on the same machine write
identical `predictions.json` files (`python -m tools.check_determinism predictions A.json
B.json`).

## Settings

- **`configs/params.yaml`:** every threshold (detector, tracker, stitching, features, each rule,
  post-processing, traffic lights, Part B, time budget). Each has a comment saying where its value
  came from: measured on the samples' tracks, or tuned on our labels.
- **`configs/scene_map.json`, `configs/scene/`:** the scene map, its LabelMe source, the
  reference picture, and an overlay picture for checking it.
- **`configs/profiles/`:** settings profiles, chosen with the environment variable
  `WIUT_PROFILE`: `demo.yaml` for the live demo on a CPU. A profile names only the settings it
  changes, and a name that isn't in `params.yaml` is refused.
- **`weights/`:** the exported detector and how it was made (`weights/README.md`).

## Development

```bash
pip install pytest ruff
pytest -q            # 221 tests, on synthetic data: no video needed
ruff check .
```

Laptops without a GPU can install `requirements-dev.txt` instead; tests that need PyTorch are
then skipped.

Tuning the rules doesn't need the GPU: perception results are cached once, then every rule runs
on the cache and is scored in seconds.

```bash
python -m tools.cache_tracks samples                  # GPU: perception results into cache/
python -m tools.eval_from_cache                       # every rule, scored on data/dev_labels.json
python -m tools.eval_from_cache --rules jaywalking --set rules.jaywalking.min_sec=1.5
```

| Tool | What it does |
| --- | --- |
| `tools/cache_tracks.py` | Runs perception on videos and saves detections, tracks and light readings to `cache/` |
| `tools/eval_from_cache.py` | Rules on cached tracks, scored with the organizers' code; `--set` overrides any setting |
| `tools/check_determinism.py` | Runs perception twice, or compares two predictions files |
| `tools/check_signals.py` | Traffic-light timelines and contact sheets, for checking the phases |
| `tools/build_scene_map.py` | Builds `configs/scene_map.json` from the LabelMe drawing and checks it |
| `tools/labels_to_ground_truth.py` | Turns our label files into `data/dev_labels.json`, in the organizers' format |
| `tools/align_videos.py`, `tools/measure_drift.py` | Camera alignment of the samples, and how the view drifts |
| `tools/check_stitching.py` | Pictures of stitched track fragments, for checking by eye |
| `tools/export_detector.py` | Exports a YOLO model to TorchScript (separate environment: `requirements-export.txt`) |
| `tools/check_detector.py`, `tools/bench_detector.py`, `tools/bench_decode.py` | Detector accuracy against Ultralytics, detector and decoder speed |
| `tools/check_frame_parity.py` | Checks that our decoder numbers frames exactly like the harness |
| `tools/pin_requirements.sh` | Pins `requirements.txt` for Linux and Windows from `requirements.in` |
| `tools/render_samples.py` | Annotated versions of whole videos (tracks, events, the light, the risk), for the website |
| `tools/website_data.py` | The website's data: EDA and results JSON, heat maps, trajectories; `--clips` cuts its clips |
| `demo/make_examples.py` | The live demo's ready-made results, from one-minute clips of the samples |

## Repository layout

```text
solution.py            the interface the harness imports (detect_events, RiskEstimator, CLASSES)
run_submission.py      the organizers' harness, unchanged
evaluate.py            the organizers' scorer, unchanged
src/
  part_a.py            Part A, end to end
  budget.py            the time budget
  video/               decoding (PyAV), frame numbering as the harness counts
  perception/          detector, tracker, the one-pass pipeline, cache, track stitching
  scene/               scene map, geometry, camera alignment, traffic lights
  features/            per-object features: ground points, speeds, zones, lanes
  rules/               one rule per class
  postprocess/         merging and clean-up of event segments
  risk/                Part B: the risk estimator and time to collision
  visualize.py         drawing results on frames (the demo and the website; not the submission)
demo/                  the live demo: a Streamlit app on the CPU profile, its examples, keep_awake.py
website/               the website: a static page (index.html, app.js), its data and pictures
configs/               params.yaml, the scene map and its sources, profiles/ (the demo's)
weights/               the exported detector
data/                  our dev labels (dev_labels.json) and the label files they came from
tools/                 development tools (above)
tests/                 unit tests, on synthetic data
examples/              the organizers' example files
```

## Models, data and licences

- **Detector:** YOLO26m by Ultralytics, pretrained on COCO and used as released, without
  fine-tuning. Ultralytics models are licensed under AGPL-3.0
  (<https://github.com/ultralytics/ultralytics>). It's exported to TorchScript FP16 with a fixed
  736×1280 input; `weights/README.md` has the export command and the file's SHA-256.
- **COCO** (Lin et al., 2014): used only through the pretrained weights. The annotations are
  licensed CC BY 4.0.
- **ByteTrack** (Zhang et al., 2022; MIT licence): its algorithm, reimplemented in NumPy and
  SciPy in `src/perception/tracker.py`.
- **No other datasets** and no other footage of this camera. Our labels were made by hand on the
  organizers' sample videos (`data/`).
- **Runtime libraries:** PyTorch and torchvision (BSD), OpenCV (Apache 2.0), PyAV (BSD; its wheel
  bundles FFmpeg, LGPL), NumPy and SciPy (BSD), PyYAML (MIT).

## Limitations

- **One camera.** The scene map is drawn for this junction's view. Another camera needs its own
  map; the code itself assumes nothing about resolution or frame rate.
- **Small dev set.** Our labels cover two videos (71 events), and the organizers' conventions
  may differ from ours in places.
- **Part B is untested on accidents:** the samples contain none. Its thresholds were set so that
  ordinary traffic rarely raises an alarm.
- **Speed on a CPU.** The judged run needs the GPU to stay within the time limit. The demo
  profile runs the same model on a CPU, at about 1.2 s per analysed frame on 2 cores.
