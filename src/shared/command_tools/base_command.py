from src.shared.handlers import Event, EventLevel
from typing import Any, Generator,Protocol,List, TypeVar, Self, Optional, Dict
from abc import ABC, abstractmethod
from itertools import chain

T = TypeVar('T')

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
        self.stop = False

    def add_handler(self, handler : Handler) -> 'BaseCommand':
        self.handlers.append(handler)
        return self

    @abstractmethod
    def __iter__(self) -> Generator[Event, None, None]:
        ...

    def before_start(self) -> Generator[Event, None, None]:
        yield from []

    def after_end(self) -> Generator[Event, None, None]:
        yield from []

    @classmethod
    def from_config(cls, config : T) -> Self:
        raise NotImplementedError

    def run(self) -> None:

        for event in chain(self.before_start(), self, self.after_end()):

            if self.stop:
                break

            self.send_event(event)

    def make_event(self,
        progress : Optional[float],
        level : EventLevel,
        message : str,
        extra : Optional[Dict] = None
    ) -> Event:
        
        return Event(
            progress=progress,
            level=level,
            message=message,
            extra=(extra or {})
        )
    
    def make_debug_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> Event:
        return self.make_event(progress, EventLevel.DEBUG, message, extra)
    
    def make_info_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> Event:
        return self.make_event(progress, EventLevel.INFO, message, extra)
    
    def make_error_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> Event:
        return self.make_event(progress, EventLevel.ERROR, message, extra)
    
    def make_warn_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> Event:
        return self.make_event(progress, EventLevel.WARN, message, extra)
    
    def make_critical_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> Event:
        return self.make_event(progress, EventLevel.CRITICAL, message, extra)
    
    def send_event(self, event : Event) -> None:

        for handler in self.handlers:

            handler.consume(event)

            if event.level == EventLevel.CRITICAL:
                event = Event(**dict(event))
                event.message = "Stopping job due to a critical error, check logs for details."
                handler.consume(event)
                self.stop = True

    def send_debug_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> Event:
        self.send_event(self.make_event(progress, EventLevel.DEBUG, message, extra))
    
    def send_info_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> Event:
        self.send_event(self.make_event(progress, EventLevel.INFO, message, extra))
    
    def send_error_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> Event:
        self.send_event(self.make_event(progress, EventLevel.ERROR, message, extra))
    
    def send_warn_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> Event:
        self.send_event(self.make_event(progress, EventLevel.WARN, message, extra))

    def send_critical_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> Event:
        self.send_event(self.make_event(progress, EventLevel.CRITICAL, message, extra))