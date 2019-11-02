from typing import List, NamedTuple

import numpy as np
from mypy_extensions import TypedDict


class SRLSpan(NamedTuple):
    predicate: int
    start: int
    end: int
    label: str


class Span(NamedTuple):
    start: int
    end: int
    label: str


class SRLExample(TypedDict):
    text: List[str]
    text_ids: np.ndarray
    srl: List[SRLSpan]
