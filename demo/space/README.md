---
title: WIUT Traffic Events
emoji: 🚦
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.28.0
python_version: "3.11"
app_file: demo/app.py
pinned: false
short_description: Traffic events and accident risk from a junction camera
---

# Traffic events from a junction camera: live demo

The live demo of our WIUT Hackathon 2026 solution (Computer Vision track). Upload a clip from
the hackathon's junction camera, up to 2 minutes; the system finds its traffic events and the
risk of an accident, frame by frame, and draws them on the video.

It runs the same code and model as our submission, on this Space's free CPU, so it is slow:
about 15-20 minutes per minute of video. The ready-made results show instantly. The code, the
model and how to run it on a GPU: https://github.com/timdeving/hackathon-ET
