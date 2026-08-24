"""Group-aware splitting, ported line-for-line from the public implementation.

Ported from ``kleos_models.data.splitting``. This is the single highest-value
differential test target in the repository, because a plausible-but-different
splitter produces a *valid-looking* split that disagrees with the public repo's
about the same data — and nothing crashes.

Three internals are easy to lose in a "cleaner" rewrite and each one changes
every assignment:

* Determinism comes from **stable hashing**, not from shuffling. An example's
  split depends only on its group key and the seed, so adding examples never
  reshuffles the existing ones. That is what makes two dataset versions
  comparable at all.
* ``_split_holdout`` renormalizes the inner train/validation fractions by
  ``remainder_fraction`` after carving out the held-out values.
* ``_split_holdout`` then re-routes any group that stable-hashes into "test"
  back into "train", because the inner split has ``test_fraction=0`` but the
  hash can still round into that bucket.

Validation is carved from the *seen*-attribute remainder while test holds the
unseen values. That asymmetry is deliberate: early stopping on OOD data would
leak the very thing the OOD split exists to measure.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from kleos_training_data.contract.schemas import TrainingExample
from kleos_training_data.errors import ContractViolationError
from kleos_training_data.hashing import stable_rank
from kleos_training_data.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SplitConfig:
    """How to partition examples.

    Fractions default to 0.8/0.1/0.1, matching the public ``SplitConfig`` model.
    Note the public *CLI* defaults to 0.7/0.15/0.15 instead — the two disagree,
    so this repository always states fractions explicitly and records them in the
    manifest rather than relying on either default.
    """

    strategy: str = "random"
    seed: int = 42
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    group_key: str | None = None
    holdout_values: list[str] = field(default_factory=list)


@dataclass
class SplitResult:
    """The outcome of a split, with enough detail to reproduce and audit it."""

    train: list[TrainingExample] = field(default_factory=list)
    validation: list[TrainingExample] = field(default_factory=list)
    test: list[TrainingExample] = field(default_factory=list)
    strategy: str = "random"
    seed: int = 42
    holdout_values: list[str] = field(default_factory=list)
    group_key: str | None = None
    notes: list[str] = field(default_factory=list)

    def split(self, name: str) -> list[TrainingExample]:
        return {"train": self.train, "validation": self.validation, "test": self.test}[name]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "counts": self.counts,
            "total": self.total,
            "holdout_values": self.holdout_values,
            "group_key": self.group_key,
            "notes": self.notes,
        }


def _assign_by_fraction(keys: Sequence[str], config: SplitConfig) -> dict[str, str]:
    """Map each key to a split name using stable hashing."""
    assignment: dict[str, str] = {}
    train_cut = config.train_fraction
    val_cut = train_cut + config.validation_fraction
    for key in keys:
        rank = stable_rank(key, config.seed)
        if rank < train_cut:
            assignment[key] = "train"
        elif rank < val_cut:
            assignment[key] = "validation"
        else:
            assignment[key] = "test"
    return assignment


def _group_examples(
    examples: Sequence[TrainingExample], group_key: str | None
) -> dict[str, list[TrainingExample]]:
    """Bucket examples by their resolved group key."""
    groups: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        groups[example.group_key(group_key)].append(example)
    return dict(groups)


def _holdout_attribute(example: TrainingExample, attribute: str) -> str:
    """Read the attribute a ``*_holdout`` strategy partitions on."""
    if attribute == "domain":
        return example.variation_axes.domain
    if attribute == "entities":
        return str(example.variation_axes.entities or "<unspecified>")
    if attribute == "format":
        return str(example.variation_axes.format or "<unspecified>")
    value = example.variation_axes.as_dict().get(attribute)
    return str(value) if value is not None else "<unspecified>"


def _split_random(examples: Sequence[TrainingExample], config: SplitConfig) -> SplitResult:
    """Shuffle and slice. Development only."""
    ordered = sorted(examples, key=lambda e: e.id)
    rng = random.Random(config.seed)
    rng.shuffle(ordered)

    total = len(ordered)
    train_end = int(total * config.train_fraction)
    val_end = train_end + int(total * config.validation_fraction)

    result = SplitResult(
        train=ordered[:train_end],
        validation=ordered[train_end:val_end],
        test=ordered[val_end:],
        strategy="random",
        seed=config.seed,
    )
    result.notes.append(
        "Random split: development use only. Near-duplicate scenarios can land on "
        "both sides, which inflates apparent generalization."
    )
    return result


def _split_grouped(
    examples: Sequence[TrainingExample], config: SplitConfig, *, strategy: str
) -> SplitResult:
    """Assign whole groups to splits so related examples never straddle."""
    key = config.group_key
    if strategy == "scenario_family_holdout" and key is None:
        key = "scenario_family"

    groups = _group_examples(examples, key)
    assignment = _assign_by_fraction(sorted(groups), config)

    result = SplitResult(strategy=strategy, seed=config.seed, group_key=key)
    for group_name, members in sorted(groups.items()):
        target = assignment[group_name]
        result.split(target).extend(sorted(members, key=lambda e: e.id))

    singletons = sum(1 for members in groups.values() if len(members) == 1)
    result.notes.append(
        f"Grouped on {key or 'group_id|scenario_family|id'}: {len(groups)} group(s), "
        f"{singletons} of size 1."
    )
    if singletons == len(groups):
        result.notes.append(
            "WARNING: every group has exactly one example, so this behaves like a "
            "random split. Populate metadata.scenario_family or metadata.group_id "
            "for the grouping to have any effect."
        )
    return result


def _split_holdout(
    examples: Sequence[TrainingExample],
    config: SplitConfig,
    *,
    attribute: str,
    strategy: str,
) -> SplitResult:
    """Hold out whole attribute values for the test split."""
    by_value: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        by_value[_holdout_attribute(example, attribute)].append(example)

    values = sorted(by_value)
    if len(values) < 2:
        raise ContractViolationError(
            f"Cannot apply {strategy!r}: attribute {attribute!r} has only "
            f"{len(values)} distinct value(s).",
            details={"attribute": attribute, "values": ", ".join(values)},
            suggestions=[
                f"A held-out split needs at least two distinct {attribute} values.",
                "Add examples covering another value, or use strategy='group'.",
            ],
        )

    if config.holdout_values:
        holdout = [v for v in config.holdout_values if v in by_value]
        unknown = sorted(set(config.holdout_values) - set(by_value))
        if unknown:
            raise ContractViolationError(
                f"Configured holdout value(s) {unknown} do not occur in the dataset.",
                details={"attribute": attribute, "available_values": ", ".join(values)},
                suggestions=[
                    "A reserved value with no coverage produces a silently empty OOD "
                    "test set, which is worse than no OOD test set.",
                ],
            )
    else:
        # Choose deterministically: take values in stable-hash order until the
        # test split is at least the configured fraction.
        ranked = sorted(values, key=lambda v: stable_rank(v, config.seed))
        target = max(1, int(len(examples) * config.test_fraction))
        holdout = []
        accumulated = 0
        for value in ranked:
            if accumulated >= target and holdout:
                break
            holdout.append(value)
            accumulated += len(by_value[value])

    if len(holdout) >= len(values):
        raise ContractViolationError(
            f"Holding out {holdout} would leave no training data for {attribute!r}.",
            details={"attribute": attribute, "all_values": ", ".join(values)},
            suggestions=["Reduce test_fraction, or hold out fewer values explicitly."],
        )

    test: list[TrainingExample] = []
    remaining: list[TrainingExample] = []
    for value, members in sorted(by_value.items()):
        (test if value in holdout else remaining).extend(members)

    # Split the remainder into train/validation, keeping groups intact. The
    # inner fractions are renormalized by the remainder, which is easy to lose
    # and changes every assignment.
    remainder_fraction = config.train_fraction + config.validation_fraction
    inner = SplitConfig(
        strategy="group",
        seed=config.seed,
        train_fraction=config.train_fraction / remainder_fraction,
        validation_fraction=config.validation_fraction / remainder_fraction,
        test_fraction=0.0,
        group_key=config.group_key,
    )
    groups = _group_examples(remaining, inner.group_key)
    assignment = _assign_by_fraction(sorted(groups), inner)

    result = SplitResult(
        strategy=strategy,
        seed=config.seed,
        holdout_values=sorted(holdout),
        group_key=config.group_key,
        test=sorted(test, key=lambda e: e.id),
    )
    for group_name, members in sorted(groups.items()):
        target_split = assignment[group_name]
        # test_fraction is 0 here, but stable hashing can still round into it.
        if target_split == "test":
            target_split = "train"
        result.split(target_split).extend(sorted(members, key=lambda e: e.id))

    result.notes.append(
        f"Held out {attribute} value(s) {sorted(holdout)} for the test split; "
        f"validation is drawn from seen values so early stopping does not leak OOD signal."
    )
    return result


#: Strategy name -> implementation.
_STRATEGIES = {
    "random": lambda ex, cfg: _split_random(ex, cfg),
    "group": lambda ex, cfg: _split_grouped(ex, cfg, strategy="group"),
    "scenario_family_holdout": lambda ex, cfg: _split_grouped(
        ex, cfg, strategy="scenario_family_holdout"
    ),
    "entity_holdout": lambda ex, cfg: _split_holdout(
        ex, cfg, attribute="entities", strategy="entity_holdout"
    ),
    "domain_holdout": lambda ex, cfg: _split_holdout(
        ex, cfg, attribute="domain", strategy="domain_holdout"
    ),
    "format_holdout": lambda ex, cfg: _split_holdout(
        ex, cfg, attribute="format", strategy="format_holdout"
    ),
}


def split_examples(
    examples: Sequence[TrainingExample], config: SplitConfig, *, verify: bool = True
) -> SplitResult:
    """Partition examples according to the configured strategy."""
    if not examples:
        raise ContractViolationError(
            "Cannot split an empty dataset.",
            suggestions=["Promote some examples first."],
        )

    implementation = _STRATEGIES.get(config.strategy)
    if implementation is None:
        raise ContractViolationError(
            f"Unknown split strategy {config.strategy!r}.",
            details={"valid": ", ".join(sorted(_STRATEGIES))},
        )

    result = implementation(list(examples), config)

    if verify:
        verify_split(result, expected_total=len(examples))

    logger.info(
        "Split %d example(s) with strategy=%s seed=%d -> %s",
        len(examples),
        result.strategy,
        result.seed,
        result.counts,
    )
    return result


def verify_split(result: SplitResult, *, expected_total: int | None = None) -> None:
    """Assert a split is well-formed.

    Raises:
        ContractViolationError: on any overlap, loss, or straddled group.
    """
    ids = {name: {e.id for e in result.split(name)} for name in ("train", "validation", "test")}

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = ids[left] & ids[right]
        if overlap:
            raise ContractViolationError(
                f"Split overlap: {len(overlap)} example(s) appear in both {left} and {right}.",
                details={"overlapping_ids": ", ".join(sorted(overlap)[:20])},
                suggestions=["This is a bug in the split strategy, not a data problem."],
            )

    if expected_total is not None and result.total != expected_total:
        raise ContractViolationError(
            f"Split lost or duplicated examples: expected {expected_total}, got {result.total}.",
            details={k: str(v) for k, v in result.counts.items()},
        )

    if result.group_key or result.strategy in ("group", "scenario_family_holdout"):
        placement: dict[str, str] = {}
        for split_name in ("train", "validation", "test"):
            for example in result.split(split_name):
                group = example.group_key(result.group_key)
                previous = placement.get(group)
                if previous is not None and previous != split_name:
                    raise ContractViolationError(
                        f"Group {group!r} was split across {previous} and {split_name}.",
                        details={"group_key": result.group_key or "(default)"},
                        suggestions=[
                            "A perturbation pair straddling the boundary makes "
                            "consistency testing meaningless.",
                        ],
                    )
                placement[group] = split_name

    if not result.train:
        raise ContractViolationError(
            "Split produced an empty training set.",
            details={k: str(v) for k, v in result.counts.items()},
        )
    if not result.test:
        logger.warning(
            "Split produced an empty test set (%s). No generalization claim can be "
            "made from this split.",
            result.counts,
        )
