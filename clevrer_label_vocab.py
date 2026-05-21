"""Map CLEVRER (color, shape, material) tuples to consecutive label ids for nn.Embedding."""

from __future__ import annotations

_COLORS = ("gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow", "gold", "silver")
_SHAPES = ("cube", "sphere", "cylinder")
_MATERIALS = ("metal", "rubber")


def _norm_shape(shape: str) -> str:
    s = str(shape).lower().strip()
    if s == "ball":
        return "sphere"
    return s


def _norm_color(color: str) -> str:
    c = str(color).lower().strip()
    if c == "gold":
        return "yellow"
    if c == "silver":
        return "gray"
    return c


def build_clevrer_attr_vocab() -> dict[tuple[str, str, str], int]:
    d: dict[tuple[str, str, str], int] = {}
    i = 0
    for c in _COLORS:
        for s in _SHAPES:
            for m in _MATERIALS:
                key = (_norm_color(c), _norm_shape(s), m)
                if key not in d:
                    d[key] = i
                    i += 1
    return d


_ATTR_VOCAB = build_clevrer_attr_vocab()
_VOCAB_SIZE = len(_ATTR_VOCAB)


def attr_tuple_to_label_index(color: str, shape: str, material: str, num_embeddings: int) -> int:
    key = (_norm_color(color), _norm_shape(shape), str(material).lower().strip())
    idx = _ATTR_VOCAB.get(key, 0)
    if num_embeddings <= 0:
        return idx
    if idx >= num_embeddings:
        return idx % num_embeddings
    return idx


def clevrer_vocab_size() -> int:
    return _VOCAB_SIZE
