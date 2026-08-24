"""Scenario definition, sampling and generation.

A scenario declares a *situation family* and the policy that resolves it. The
training target is computed from that policy, never authored beside the prompt
and never taken from a model's own output — see :mod:`.policies`.
"""

from __future__ import annotations

from kleos_training_data.scenarios.generator import (
    Candidate,
    build_situation,
    generate,
    generation_fingerprint,
)
from kleos_training_data.scenarios.loader import load_catalog, load_scenario
from kleos_training_data.scenarios.models import Scenario
from kleos_training_data.scenarios.policies import POLICIES, Decision, decide, resolve_policy
from kleos_training_data.scenarios.rendering import (
    ANSWER_FORMATS,
    PROMPT_FORMATS,
    render_answer,
    render_prompt,
    render_system_prompt,
)
from kleos_training_data.scenarios.situations import Item, Situation
from kleos_training_data.scenarios.surrogates import SurrogatePool, load_pools

__all__ = [
    "ANSWER_FORMATS",
    "POLICIES",
    "PROMPT_FORMATS",
    "Candidate",
    "Decision",
    "Item",
    "Scenario",
    "Situation",
    "SurrogatePool",
    "build_situation",
    "decide",
    "generate",
    "generation_fingerprint",
    "load_catalog",
    "load_pools",
    "load_scenario",
    "render_answer",
    "render_prompt",
    "render_system_prompt",
    "resolve_policy",
]
