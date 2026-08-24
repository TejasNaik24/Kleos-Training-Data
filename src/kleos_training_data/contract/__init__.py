"""Mirror of the public kleos-models dataset contract.

Nothing outside this package should reach for the public repository. Everything
the pipeline needs to *produce* a release — the models, the JSONL bytes, the
vocabulary — is here, so the pipeline runs with kleos-models absent.

See :mod:`.pin` for what the mirror is pinned to and how to update it.
"""

from __future__ import annotations

from kleos_training_data.contract.constants import (
    ALLOWED_METADATA_EXTRAS,
    DATASET_SCHEMA_VERSION,
    MESSAGE_ROLES,
    OOD_SHIFT_KINDS,
    PERTURBATION_KINDS,
    PREPROCESSING_VERSION,
    PROMOTED_QUALITY_STATUS,
    QUALITY_STATUSES,
    REQUIRED_SPLITS,
    REQUIRED_VARIATION_AXES,
    SOURCE_TYPES,
    SPLIT_FILENAMES,
    SPLIT_NAMES,
    SPLIT_STRATEGIES,
    SUPPORTED_TASKS,
    TASK_CODES,
    VARIATION_AXES,
)
from kleos_training_data.contract.pin import (
    CONTRACT_SOURCE_COMMIT,
    CONTRACT_SOURCE_REPO,
)
from kleos_training_data.contract.schemas import (
    DatasetManifest,
    EvaluationExample,
    ExampleMetadata,
    Message,
    SplitCounts,
    TrainingExample,
    VariationAxes,
    contains_reasoning,
    strip_reasoning,
)
from kleos_training_data.contract.writer import (
    dumps_example,
    iter_jsonl,
    read_examples,
    round_trips,
    write_jsonl,
)

__all__ = [
    "ALLOWED_METADATA_EXTRAS",
    "CONTRACT_SOURCE_COMMIT",
    "CONTRACT_SOURCE_REPO",
    "DATASET_SCHEMA_VERSION",
    "MESSAGE_ROLES",
    "OOD_SHIFT_KINDS",
    "PERTURBATION_KINDS",
    "PREPROCESSING_VERSION",
    "PROMOTED_QUALITY_STATUS",
    "QUALITY_STATUSES",
    "REQUIRED_SPLITS",
    "REQUIRED_VARIATION_AXES",
    "SOURCE_TYPES",
    "SPLIT_FILENAMES",
    "SPLIT_NAMES",
    "SPLIT_STRATEGIES",
    "SUPPORTED_TASKS",
    "TASK_CODES",
    "VARIATION_AXES",
    "DatasetManifest",
    "EvaluationExample",
    "ExampleMetadata",
    "Message",
    "SplitCounts",
    "TrainingExample",
    "VariationAxes",
    "contains_reasoning",
    "dumps_example",
    "iter_jsonl",
    "read_examples",
    "round_trips",
    "strip_reasoning",
    "write_jsonl",
]
