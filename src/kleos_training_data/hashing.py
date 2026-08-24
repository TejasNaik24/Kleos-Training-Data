"""Canonicalization and digests.

Two families of function live here and they answer different questions:

**Ours.** :func:`canonical_json`, :func:`canonical_bytes`, :func:`sha256_hex`,
:func:`file_sha256` — used for record integrity, release file hashes and example
identity.

**Ported.** :func:`stable_rank` and :func:`normalize_text` are line-for-line
ports of the public repo's implementations (see
``contract.pin.MIRRORED_BEHAVIOURS``). They exist here so this repository can
split and deduplicate *identically* to the public repo without importing it. If
they drift, the two repositories reach different verdicts about the same data —
the worst failure mode in a two-repository research setup, because nothing
crashes. ``tests/test_differential_*.py`` compare them directly.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

#: Canonical JSON settings. Separators matter: the public manifest hash uses
#: ``(",", ":")``, and a single space would change every content hash.
CANONICAL_JSON_KWARGS: Final[dict[str, Any]] = {
    "sort_keys": True,
    "separators": (",", ":"),
    "ensure_ascii": False,
}

#: Chunk size for streaming file digests. Release files are small, but a
#: whole-file read is a habit that eventually meets a large file.
_CHUNK_BYTES: Final[int] = 1024 * 1024


def canonical_json(obj: Any) -> str:
    """Render an object as canonical JSON: sorted keys, tight separators."""
    return json.dumps(obj, **CANONICAL_JSON_KWARGS)


def canonical_bytes(obj: Any) -> bytes:
    """Canonical JSON as NFC-normalized UTF-8 bytes.

    Unicode normalization is applied to the *serialized form* so that two
    strings which are canonically equivalent but differently composed — "é" as
    one code point versus "e" plus a combining accent — produce one digest. Text
    arrives from YAML files, HTTP responses and human edits on several
    platforms, and those disagree about composition routinely.
    """
    return unicodedata.normalize("NFC", canonical_json(obj)).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    """Hex sha256 of bytes, or of a string's UTF-8 encoding."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_hash(obj: Any) -> str:
    """Hex sha256 over an object's canonical byte form."""
    return sha256_hex(canonical_bytes(obj))


def short_digest(hex_digest: str, chars: int = 16) -> str:
    """First ``chars`` of a hex digest."""
    return hex_digest[:chars]


def file_sha256(path: Path | str) -> str:
    """Hex sha256 of a file's contents, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(directory: Path | str, *, names: Sequence[str]) -> dict[str, str]:
    """Hash each named file that exists in ``directory``.

    Missing names are omitted rather than raising: a release with no validation
    split is legitimate, and its manifest simply has no entry for one.
    """
    root = Path(directory)
    return {name: file_sha256(root / name) for name in names if (root / name).is_file()}


# ---------------------------------------------------------------------------
# Ported from the public repository — see contract/pin.py
# ---------------------------------------------------------------------------


def stable_rank(key: str, seed: int) -> float:
    """Deterministic pseudo-random value in [0, 1) for a key.

    Port of ``kleos_models.data.splitting._stable_rank`` (splitting.py:86-94).

    Hashing rather than shuffling means an example's split assignment depends
    only on its key and the seed — adding new examples never reshuffles the
    existing ones. That property is what makes two dataset versions comparable,
    and it is why group keys must stay stable across versions.
    """
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_DIGIT_RUN = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Aggressively normalize text for duplicate detection.

    Port of ``kleos_models.data.leakage.normalize_text`` (leakage.py:95-108).

    Digit runs collapse to a single ``0``, so "Deadline: Mar 3" and "deadline
    mar 7" normalize to the same string. That is intentional and worth stating
    plainly: **changing only a number does not make a scenario new.** A dataset
    that looks diverse because its dates differ is not diverse.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = stripped.casefold()
    no_digits = _DIGIT_RUN.sub("0", lowered)
    no_punct = _PUNCTUATION.sub(" ", no_digits)
    return _WHITESPACE.sub(" ", no_punct).strip()


def shingles(text: str, size: int = 5) -> set[str]:
    """Character n-grams of ``text``, for Jaccard similarity."""
    if len(text) < size:
        return {text} if text else set()
    return {text[i : i + size] for i in range(len(text) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    """Jaccard similarity of two shingle sets. Empty on both sides is 1.0."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)
