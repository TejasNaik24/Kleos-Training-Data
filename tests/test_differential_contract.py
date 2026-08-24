"""The contract mirror is identical to the pinned public implementation.

These are the tests that make the mirror defensible. Without them, "we ported it
faithfully" is an assertion; with them it is a checked property.

They run only where ``kleos-models`` is installed::

    make install-compat

A **skip is a degraded run, not a pass.** CI installs the pinned extra and runs
``check_contract_compat.py --strict``, which treats a skip as failure. Locally
skipping is correct: the offline pipeline must be workable without the public
repo checked out.

What is compared:

* the vocabulary, element-wise **and order-wise**
* the JSONL bytes, over the whole fixture corpus
* accept/reject decisions, over every valid and invalid case
* ``stable_rank`` and ``normalize_text``, the two ported functions whose
  divergence would make the repositories disagree about the same data without
  anything crashing
"""

from __future__ import annotations

import random

import pytest
from tests.contract_cases import INVALID_CASES, VALID_CASES, base

from kleos_training_data.contract import constants as mirror_constants
from kleos_training_data.contract.pin import CONTRACT_SOURCE_COMMIT, MIRRORED_CONSTANTS
from kleos_training_data.contract.schemas import TrainingExample as MirrorExample
from kleos_training_data.contract.writer import dumps_example
from kleos_training_data.hashing import normalize_text, stable_rank

pytestmark = pytest.mark.requires_kleos_models


class TestVocabulary:
    """The early-warning detector.

    A newly registered task or source type fails here long before any byte-level
    test notices, because the byte tests only exercise values we already use.
    """

    @pytest.mark.parametrize("name", MIRRORED_CONSTANTS)
    def test_constant_is_identical(self, name: str) -> None:
        from kleos_models import constants as public

        mine = getattr(mirror_constants, name)
        theirs = getattr(public, name)
        assert mine == theirs, (
            f"{name} has drifted from the public contract at {CONTRACT_SOURCE_COMMIT[:12]}.\n"
            f"  mirror: {mine!r}\n  public: {theirs!r}\n"
            f"Update the mirror, re-run every differential suite, then move the pin."
        )

    @pytest.mark.parametrize("name", MIRRORED_CONSTANTS)
    def test_constant_ordering_is_identical(self, name: str) -> None:
        """Order looks cosmetic and is not.

        It decides how `holdout_values` sorts and how a coverage report
        enumerates cells, so a reordered tuple changes an artifact without
        changing a single value.
        """
        from kleos_models import constants as public

        mine = getattr(mirror_constants, name)
        if not isinstance(mine, tuple):
            pytest.skip(f"{name} is not ordered")
        assert list(mine) == list(getattr(public, name))


class TestWriterBytes:
    @pytest.mark.parametrize("name", sorted(VALID_CASES))
    def test_our_line_matches_the_public_writer(self, name: str, tmp_path) -> None:
        from kleos_models.data.loaders import write_jsonl as public_write
        from kleos_models.data.schemas import TrainingExample as PublicExample

        payload = VALID_CASES[name]
        mine = dumps_example(MirrorExample.model_validate(payload))

        target = tmp_path / f"{name}.jsonl"
        public_write([PublicExample.model_validate(payload)], target)
        theirs = target.read_text(encoding="utf-8")

        assert mine == theirs, (
            f"{name}: our JSONL bytes differ from the public writer's.\n"
            f"  mirror: {mine!r}\n  public: {theirs!r}"
        )

    def test_a_whole_corpus_is_byte_identical(self, tmp_path) -> None:
        from kleos_models.data.loaders import write_jsonl as public_write
        from kleos_models.data.schemas import TrainingExample as PublicExample

        payloads = list(VALID_CASES.values())
        ours = tmp_path / "ours.jsonl"
        ours.write_text(
            "".join(dumps_example(MirrorExample.model_validate(p)) for p in payloads),
            encoding="utf-8",
        )
        theirs = tmp_path / "theirs.jsonl"
        public_write([PublicExample.model_validate(p) for p in payloads], theirs)

        assert ours.read_bytes() == theirs.read_bytes()


class TestValidationDecisions:
    @pytest.mark.parametrize("name", sorted(VALID_CASES))
    def test_both_accept(self, name: str) -> None:
        from kleos_models.data.schemas import TrainingExample as PublicExample

        PublicExample.model_validate(VALID_CASES[name])
        MirrorExample.model_validate(VALID_CASES[name])

    @pytest.mark.parametrize("name", sorted(INVALID_CASES))
    def test_both_reject(self, name: str) -> None:
        from kleos_models.data.schemas import TrainingExample as PublicExample
        from pydantic import ValidationError

        payload, _expected = INVALID_CASES[name]
        with pytest.raises(ValidationError):
            PublicExample.model_validate(payload)
        with pytest.raises(ValidationError):
            MirrorExample.model_validate(payload)

    @pytest.mark.parametrize("name", sorted(INVALID_CASES))
    def test_the_failing_field_paths_match(self, name: str) -> None:
        """Agreeing on *what* is wrong, not merely *that* something is.

        Two validators can both reject a payload for different reasons and still
        diverge on the next payload. Comparing the locations catches that.
        """
        from kleos_models.data.schemas import TrainingExample as PublicExample
        from pydantic import ValidationError

        payload, _expected = INVALID_CASES[name]

        def locations(model) -> set[tuple]:
            try:
                model.model_validate(payload)
            except ValidationError as exc:
                return {tuple(err["loc"]) for err in exc.errors()}
            return set()

        assert locations(MirrorExample) == locations(PublicExample)


class TestDerivedViews:
    @pytest.mark.parametrize("name", sorted(VALID_CASES))
    def test_content_hash_matches(self, name: str) -> None:
        from kleos_models.data.schemas import TrainingExample as PublicExample

        payload = VALID_CASES[name]
        assert (
            MirrorExample.model_validate(payload).content_hash()
            == PublicExample.model_validate(payload).content_hash()
        )

    @pytest.mark.parametrize("name", sorted(VALID_CASES))
    def test_conversation_text_matches(self, name: str) -> None:
        from kleos_models.data.schemas import TrainingExample as PublicExample

        payload = VALID_CASES[name]
        for include in (True, False):
            assert MirrorExample.model_validate(payload).conversation_text(
                include_assistant=include
            ) == PublicExample.model_validate(payload).conversation_text(include_assistant=include)

    @pytest.mark.parametrize("name", sorted(VALID_CASES))
    def test_group_key_matches(self, name: str) -> None:
        from kleos_models.data.schemas import TrainingExample as PublicExample

        payload = VALID_CASES[name]
        assert (
            MirrorExample.model_validate(payload).group_key()
            == PublicExample.model_validate(payload).group_key()
        )

    def test_manifest_content_hash_matches(self) -> None:
        from kleos_models.data.schemas import DatasetManifest as PublicManifest

        from kleos_training_data.contract.schemas import DatasetManifest as MirrorManifest

        hashes = {"train.jsonl": "a" * 64, "test.jsonl": "b" * 64}
        assert (
            MirrorManifest(version="v1", file_hashes=hashes).compute_content_hash()
            == PublicManifest(version="v1", file_hashes=hashes).compute_content_hash()
        )


class TestPortedFunctions:
    """The two functions whose divergence would be completely silent."""

    def test_stable_rank_matches_over_many_keys(self) -> None:
        from kleos_models.data.splitting import _stable_rank

        rng = random.Random(1712)
        alphabet = "abcdefghijklmnop-_.:0123456789 "
        keys = ["".join(rng.choices(alphabet, k=rng.randint(1, 60))) for _ in range(2000)]

        for seed in (0, 1, 42, 1712, 2**31 - 1):
            for key in keys:
                assert stable_rank(key, seed) == _stable_rank(key, seed), (
                    f"stable_rank diverged on {key!r} at seed {seed}. Splits computed "
                    f"here would disagree with the public repo's on the same data."
                )

    @pytest.mark.parametrize(
        "text",
        [
            "Deadline: Mar 3",
            "deadline mar 7",
            "Café RÉSUMÉ — naïve",
            "café",
            "café",
            "  multi   space\t\ttabs  ",
            "",
            "!!!???...",
            "2026-08-22T10:00:00Z",
            "Item A (due 2026-03-03) vs Item B (due 2026-11-19)",
            "日本語のテキスト",
            "mixed 123 numbers 4567 here",
        ],
    )
    def test_normalize_text_matches(self, text: str) -> None:
        from kleos_models.data.leakage import normalize_text as public_normalize

        assert normalize_text(text) == public_normalize(text)

    def test_the_digit_collapse_is_reproduced(self) -> None:
        """Changing only a number does not make a scenario new — on both sides."""
        from kleos_models.data.leakage import normalize_text as public_normalize

        a, b = "Deadline: Mar 3", "Deadline: Mar 7"
        assert normalize_text(a) == normalize_text(b) == public_normalize(a)


class TestPublicLoaderAcceptsOurOutput:
    """The end the whole mirror exists to serve."""

    def test_the_public_loader_reads_a_file_we_wrote(self, tmp_path) -> None:
        from kleos_models.data.loaders import load_examples

        from kleos_training_data.contract.writer import write_jsonl

        examples = []
        for index, payload in enumerate(VALID_CASES.values()):
            copy = dict(payload)
            copy["id"] = f"kx-npr-{index:016d}"
            examples.append(MirrorExample.model_validate(copy))

        path = write_jsonl(examples, tmp_path / "train.jsonl")
        loaded, report = load_examples(path, strict=True)

        assert not report.errors, f"the public loader rejected our output: {report.errors}"
        assert len(loaded) == len(examples)

    def test_the_public_validator_passes_our_output(self, tmp_path) -> None:
        from kleos_models.data.validation import validate_examples

        from kleos_training_data.contract.writer import write_jsonl

        examples = []
        for index, payload in enumerate(VALID_CASES.values()):
            copy = dict(payload)
            copy["id"] = f"kx-npr-{index:016d}"
            copy["metadata"] = {"source": "synthetic", "quality_status": "reviewed"}
            copy["variation_axes"] = {**base()["variation_axes"], **copy.get("variation_axes", {})}
            examples.append(MirrorExample.model_validate(copy))

        write_jsonl(examples, tmp_path / "train.jsonl")

        from kleos_models.data.loaders import load_examples

        loaded, _ = load_examples(tmp_path / "train.jsonl", strict=True)
        report = validate_examples(loaded, split_name="train", require_reviewed=True)

        assert report.ok, f"the public validator found errors:\n{report.render()}"
