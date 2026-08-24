"""KLEOS training-data pipeline: collection, review, promotion, release.

This is the PRIVATE repository. It produces immutable, sanitized, schema-valid
dataset releases consumed by the public ``kleos-models`` repository through one
interface — a directory path::

    python <kleos-models>/scripts/train.py --config <config> --dataset <release-dir>

Nothing here is imported by the public repo, and nothing here imports it at
runtime. The contract the releases must satisfy is mirrored in
:mod:`kleos_training_data.contract` and checked against a pinned public commit by
the differential tests.

The research principle everything else serves: **train a generalizable decision
policy, not private facts about a person.**
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
