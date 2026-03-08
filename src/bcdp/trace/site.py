# bcdp/trace/site.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Stream(Enum):
    RESID_PRE = auto()
    RESID_MID = auto()
    RESID_POST = auto()
    ATTN_OUT = auto()
    MLP_OUT = auto()


class Position(Enum):
    SUBJECT = auto()
    ENTITY = auto()
    QA = auto()
    LAST = auto()


@dataclass(frozen=True)
class Site:
    layer: int
    stream: Stream
    position: Position
    index: Optional[int] = None  # required for SUBJECT/ENTITY; must be None for LAST

    def __post_init__(self) -> None:
        if self.position == Position.LAST or self.position == Position.QA:
            if self.index is not None:
                raise ValueError("Site.index must be None when position == LAST")
        else:
            if self.index is None:
                raise ValueError(f"Site.index must be set when position == {self.position}")
            if self.index < 0:
                raise ValueError("Site.index must be >= 0")
