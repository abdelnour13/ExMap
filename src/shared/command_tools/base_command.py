from typing import Any, Generator,Protocol,List
from abc import ABC,abstractmethod
from itertools import chain

class Handler(Protocol):

    def consume(self, event : Any):
        ...

class BaseCommand(ABC):
    """
    The base of all commands, every command should inherit 
    from this class and implement the `__iter__` method.
    """

    def __init__(self):
        super().__init__()
        self.handlers : List[Handler] = []

    def add_handler(self, handler : Handler) -> 'BaseCommand':
        self.handlers.append(handler)
        return self

    @abstractmethod
    def __iter__(self) -> Generator[Any, None, None]:
        ...

    def before_start(self) -> Generator[Any, None, None]:
        yield from []

    def after_end(self) -> Generator[Any, None, None]:
        yield from []

    def run(self) -> None:

        for event in chain(self.before_start(), self, self.after_end()):
            for handler in self.handlers:
                handler.consume(event)