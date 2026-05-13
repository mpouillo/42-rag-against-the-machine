from _typeshed import Incomplete
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing_extensions import TypedDict

__all__ = ['PIIDetectionError', 'PIIMatch', 'RedactionRule', 'ResolvedRedactionRule', 'apply_strategy', 'detect_credit_card', 'detect_email', 'detect_ip', 'detect_mac_address', 'detect_url']

class PIIMatch(TypedDict):
    type: str
    value: str
    start: int
    end: int

class PIIDetectionError(Exception):
    pii_type: Incomplete
    matches: Incomplete
    def __init__(self, pii_type: str, matches: Sequence[PIIMatch]) -> None: ...
Detector = Callable[[str], list[PIIMatch]]

def detect_email(content: str) -> list[PIIMatch]: ...
def detect_credit_card(content: str) -> list[PIIMatch]: ...
def detect_ip(content: str) -> list[PIIMatch]: ...
def detect_mac_address(content: str) -> list[PIIMatch]: ...
def detect_url(content: str) -> list[PIIMatch]: ...
def apply_strategy(content: str, matches: list[PIIMatch], strategy: RedactionStrategy) -> str: ...

@dataclass(frozen=True)
class RedactionRule:
    pii_type: str
    strategy: RedactionStrategy = ...
    detector: Detector | str | None = ...
    def resolve(self) -> ResolvedRedactionRule: ...

@dataclass(frozen=True)
class ResolvedRedactionRule:
    pii_type: str
    strategy: RedactionStrategy
    detector: Detector
    def apply(self, content: str) -> tuple[str, list[PIIMatch]]: ...
