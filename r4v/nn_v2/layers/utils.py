from typing import Sequence, TypeVar

T = TypeVar("T")


def single(s: Sequence[T]) -> T:
    s_len = len(s)
    if s_len < 1:
        raise ValueError("Sequence is empty.")
    if s_len > 1:
        raise ValueError("More than 1 element in sequence.")
    return s[0]
