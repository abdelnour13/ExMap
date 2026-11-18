import math
from typing import Generic, TypeVar, Tuple
from collections.abc import Iterable, Iterator, Sized, Callable
from itertools import product, batched

T = TypeVar('T')
I = TypeVar('I')
O = TypeVar('O')

class Enumerate(Generic[T]):

    def __init__(self, iterable: Iterable[T]) -> None:
        self.iterable: Iterable[T] = iterable

    def __iter__(self) -> Iterator[Tuple[int, T]]:
        return enumerate(self.iterable)

    def __len__(self) -> int:

        if not isinstance(self.iterable, Sized):
            raise TypeError("iterable must implement Sized")

        return len(self.iterable)

class Map(Generic[T]):

    def __init__(self, iterable: Iterable[T], fn : Callable[[I],O]) -> None:
        self.iterable: Iterable[T] = iterable
        self.fn = fn

    def __iter__(self) -> Iterator[O]:
        return map(self.fn, self.iterable)

    def __len__(self) -> int:

        if not isinstance(self.iterable, Sized):
            raise TypeError("iterable must implement Sized")

        return len(self.iterable)

class Batched(Generic[T]):

    def __init__(self, iterable: Iterable[T], n: int) -> None:
        self.iterable: Iterable[T] = iterable
        self.n: int = n

    def __iter__(self) -> Iterator[tuple[T, ...]]:
        return batched(self.iterable, self.n)

    def __len__(self) -> int:
        
        if not isinstance(self.iterable, Sized):
            raise TypeError("iterable must implement Sized")

        return (len(self.iterable) + self.n - 1) // self.n

class Range:

    def __init__(self, 
        stop : int,
        start : int = 0,
        step : int = 1
    ) -> None:
        self.stop = stop
        self.start = start
        self.step = step

    def __iter__(self):
        yield from range(self.start, self.stop, self.step)

    def __len__(self) -> None:
        return math.ceil((self.stop - self.start) / self.step)

class Product(Generic[T]):

    def __init__(
        self,
        *iterables: Iterable[T],
        repeat: int = 1,
    ) -> None:
        self.iterables: tuple[Iterable[T], ...] = iterables
        self.repeat: int = repeat

    def __iter__(self) -> Iterator[tuple[T, ...]]:
        return product(*self.iterables, repeat=self.repeat)

    def __len__(self) -> int:
        lengths: list[int] = []

        for iterable in self.iterables:
            if not isinstance(iterable, Sized):
                raise TypeError("all iterables must implement Sized")

            lengths.append(len(iterable))

        n: int = 1
        for length in lengths:
            n *= length

        return n ** self.repeat