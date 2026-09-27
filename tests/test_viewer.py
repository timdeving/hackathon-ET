"""The live demo's results viewer (demo/viewer.py): the risk curve it draws, its alarms, and the
page it builds."""
from __future__ import annotations

from demo.viewer import alarms, risk_steps, viewer_html


def test_the_risk_curve_keeps_each_steps_highest_value():
    risk = [[0.0, 0.1], [0.2, 0.4], [0.5, 0.2], [0.9, 0.3], [1.0, 0.0]]
    assert risk_steps(risk) == [[0.0, 0.4], [0.5, 0.3], [1.0, 0.0]]


def test_alarms_are_runs_at_the_threshold_merged_when_under_2_s_apart():
    risk = [[t / 10, 0.6 if 10 <= t < 20 or 30 <= t < 35 or 80 <= t < 90 else 0.1]
            for t in range(100)]
    assert alarms(risk) == [[1.0, 3.4], [8.0, 8.9]]  # 1.9-3.0 s apart merge; 4.6 s don't


def test_the_page_carries_the_clip_and_cannot_be_broken_out_of():
    page = viewer_html("https://example.org/clip.mp4", [[1.0, 2.0, "</script><b>"]],
                       [[0.0, 0.1]], 10.0, [[0.0, 5.0, "red"]])
    assert "https://example.org/clip.mp4" in page and "__DATA__" not in page
    assert "</script><b>" not in page  # a label can't end the page's script early
    assert '"base64"' in viewer_html(b"\x00\x01", [], [], 1.0)  # an upload's bytes travel inline
