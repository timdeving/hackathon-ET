"""solution.py: the interface the organizers' harness (run_submission.py) imports.

Kept thin on purpose: Part A lives in src/part_a.py and Part B in src/risk/estimator.py.
Names and signatures are exactly the starter kit's.
"""
import logging

from src import part_a
from src.events import CLASSES
from src.part_a import detect_events
from src.risk import estimator
from src.risk.estimator import RiskEstimator

__all__ = ["CLASSES", "RiskEstimator", "detect_events"]

# The harness prints its own progress lines; this shows our pipeline's log lines next to them.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

# The harness imports this module before it starts timing any video, so the models load here:
# Part A's, and Part B's own copy (Part B shares code with Part A, never its state or results).
part_a.load_models()
estimator.load_models()
