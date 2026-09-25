"""solution.py: the interface the organizers' harness (run_submission.py) imports.

Kept thin on purpose: Part A lives in src/part_a.py and Part B in src/risk/estimator.py.
Names and signatures are exactly the starter kit's.
"""
import logging

from src.events import CLASSES
from src.part_a import detect_events
from src.risk.estimator import RiskEstimator

__all__ = ["CLASSES", "RiskEstimator", "detect_events"]

# The harness prints its own progress lines; this shows our pipeline's log lines next to them.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
