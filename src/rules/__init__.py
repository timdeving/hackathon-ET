"""One rule per event class: per-object features and the scene map in, raw segments out.

Each rule module has LABEL, the official class id, and find(context) -> list[Segment].
find_events() runs the enabled ones. Enabling a class is a team decision (params.yaml
`rules.enabled`): predicting a class the test set doesn't contain adds a zero to Score A.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from src.events import Segment
from src.features.tracks import PointMapper, compute_features
from src.perception.pipeline import PerceptionResult
from src.rules import failure_to_yield, jaywalking, solid_line_crossing, wrong_way
from src.rules.common import RuleContext
from src.scene.scene_map import SceneMap

log = logging.getLogger(__name__)

RULES: dict[str, Callable[[RuleContext], list[Segment]]] = {
    module.LABEL: module.find
    for module in (jaywalking, failure_to_yield, wrong_way, solid_line_crossing)
}


def find_events(
    result: PerceptionResult,
    mapper: PointMapper,
    scene: SceneMap,
    params: Mapping[str, Any],
    labels: Sequence[str] | None = None,
    skip_failures: bool = False,
) -> list[Segment]:
    """Raw segments of every rule in `labels` (default: params `rules.enabled`), for one video.

    mapper maps the video's pixels onto the reference picture the scene map is drawn on.
    Segments still need finalize_events(): they may overlap, flicker, or run past the video.
    skip_failures: a rule that raises is logged and skipped, so one rule's bug costs only its
    own class (the submission); otherwise the error propagates (development).
    """
    labels = list(params["rules"]["enabled"] if labels is None else labels)
    missing = sorted(set(labels) - set(RULES))
    if missing:
        raise ValueError(f"no rule for {missing}; rules exist for {sorted(RULES)}")
    if not labels:
        return []
    features = compute_features(result, mapper, scene, params["features"])
    segments = []
    for label in labels:
        context = RuleContext(features, scene, params["rules"][label])
        if not skip_failures:
            segments += RULES[label](context)
            continue
        try:
            segments += RULES[label](context)
        except Exception:  # the submission keeps the other classes' events
            log.exception("the %s rule failed; that class gets no events here", label)
    return segments
