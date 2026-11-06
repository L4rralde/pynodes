from dataclasses import dataclass, field
from typing import Any
import uuid
import time


@dataclass(frozen=True)
class Header:
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0

@dataclass(frozen=True)
class Message:
    topic: str
    payload: Any
    header: Header = field(default_factory=Header)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def create(cls, topic: str, payload: Any, sequence: int):
        return cls(
            topic=topic,
            payload=payload,
            header=Header(sequence=sequence)
        )

