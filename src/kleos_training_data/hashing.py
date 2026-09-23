from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

CANONICAL_JSON_KWARGS: Final[dict[str, Any]] = {
    "sort_keys": True,
    "separators": (",", ":"),
    "ensure_ascii": False,
}

_CHUNK_BYTES: Final[int] = 1024 * 1024


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, **CANONICAL_JSON_KWARGS)


def canonical_bytes(obj: Any) -> bytes:
    return unicodedata.normalize("NFC", canonical_json(obj)).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_hash(obj: Any) -> str:
    return sha256_hex(canonical_bytes(obj))


def short_digest(hex_digest: str, chars: int = 16) -> str:
    return hex_digest[:chars]


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(directory: Path | str, *, names: Sequence[str]) -> dict[str, str]:
    root = Path(directory)
    return {name: file_sha256(root / name) for name in names if (root / name).is_file()}


def stable_rank(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_DIGIT_RUN = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = stripped.casefold()
    no_digits = _DIGIT_RUN.sub("0", lowered)
    no_punct = _PUNCTUATION.sub(" ", no_digits)
    return _WHITESPACE.sub(" ", no_punct).strip()


def shingles(text: str, size: int = 5) -> set[str]:
    if len(text) < size:
        return {text} if text else set()
    return {text[i : i + size] for i in range(len(text) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)
