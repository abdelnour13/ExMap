import traceback
from src.shared.handlers import Event, EventLevel
from typing import Any, Generator,Protocol,List, TypeVar, Self, Optional, Dict, Iterable, Type
from abc import ABC, abstractmethod
from itertools import chain
from dataclasses import asdict

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
        # self.events : Queue[Event] = Queue()
        self.progresses = { "main" : 0.0 }

    def get_task_progress(self, task : str = "main") -> float | None:
        return self.progresses[task]

    def add_handler(self, handler : Handler) -> 'BaseCommand':
        self.handlers.append(handler)
        return self

    def remove_handlers(self, cls : Type[Handler]) -> int:
        n = len(self.handlers)
        self.handlers = [handler for handler in self.handlers if not isinstance(handler, cls)]
        m = len(self.handlers)
        return n - m

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
            
        for event in self.generator():
            self.send_event(event)

    def generator(self) -> Generator[Event, None, None]:
    
        try:

            yield self.make_event(
                task="main",
                progress=self.get_task_progress('main'),
                level=EventLevel.START,
                message=f"Starting Command {self.__class__.__name__}."
            )

            for event in chain(self.before_start(), self, self.after_end()):

                if self.stop:
                    break

                yield event

            yield self.make_event(
                task="main",
                progress=self.get_task_progress('main'),
                level=EventLevel.END,
                message=f"Command {self.__class__.__name__} has been completed with sucess."
            )

        except Exception as e:

            err = traceback.format_exc()

            yield self.make_critical_event(
                progress=self.get_task_progress(), 
                message=f"Command stopping due to the following error : {err}"
            )

    def cancel(self):

        self.send_event(Event(
            task="main",
            progress=self.get_task_progress(), 
            message="Command stopping by user demand.",
            level=EventLevel.CANCELLED
        ))

        self.stop = True

    def make_event(self,
        progress : Optional[float],
        level : EventLevel,
        message : str,
        task : str = "main",
        extra : Optional[Dict] = None
    ) -> Event:
        
        return Event(
            task=task,
            progress=progress,
            level=level,
            message=message,
            extra=(extra or {})
        )
    
    def make_debug_event(self,
        progress : Optional[float],
        message : str,
        task : str = "main",
        extra : Optional[Dict] = None
    ) -> Event:
        return self.make_event(progress, EventLevel.DEBUG, message, task, extra)
    
    def make_info_event(self,
        progress : Optional[float],
        message : str,
        task : str = "main",
        extra : Optional[Dict] = None
    ) -> Event:
        return self.make_event(progress, EventLevel.INFO, message, task, extra)
    
    def make_error_event(self,
        progress : Optional[float],
        message : str,
        task : str = "main",
        extra : Optional[Dict] = None
    ) -> Event:
        return self.make_event(progress, EventLevel.ERROR, message, task, extra)
    
    def make_warn_event(self,
        progress : Optional[float],
        message : str,
        task : str = "main",
        extra : Optional[Dict] = None
    ) -> Event:
        return self.make_event(progress, EventLevel.WARN, message, task, extra)
    
    def make_critical_event(self,
        progress : Optional[float],
        message : str,
        task : str = "main",
        extra : Optional[Dict] = None
    ) -> Event:
        return self.make_event(progress, EventLevel.CRITICAL, message, task, extra)
    
    def send_event(self, event : Event) -> None:

        self.progresses[event.task] = event.progress

        for handler in self.handlers:
            handler.consume(event)

        if event.level == EventLevel.CRITICAL:

            event = Event(**asdict(event))
            event.message = "Stopping job due to a critical error, check logs for details."

            for handler in self.handlers:
                handler.consume(event)
            
            self.stop = True

    def send_debug_event(self,
        progress : Optional[float],
        message : str,
        task : str = "main",
        extra : Optional[Dict] = None
    ) -> None:
        self.send_event(self.make_event(progress, EventLevel.DEBUG, message, task, extra))
    
    def send_info_event(self,
        progress : Optional[float],
        message : str,
        task : str = "main",
        extra : Optional[Dict] = None
    ) -> None:
        self.send_event(self.make_event(progress, EventLevel.INFO, message, task, extra))
    
    def send_error_event(self,
        progress : Optional[float],
        message : str,
        task : str = "main",
        extra : Optional[Dict] = None
    ) -> None:
        self.send_event(self.make_event(progress, EventLevel.ERROR, message, task, extra))
    
    def send_warn_event(self,
        progress : Optional[float],
        message : str,
        task : str = "main",
        extra : Optional[Dict] = None
    ) -> None:
        self.send_event(self.make_event(progress, EventLevel.WARN, message, task, extra))

    def send_critical_event(self,
        progress : Optional[float],
        message : str,
        task : str = "main",
        extra : Optional[Dict] = None
    ) -> None:
        self.send_event(self.make_event(progress, EventLevel.CRITICAL, message, task, extra))

    def track(self, 
        it : Iterable[T], 
        task : str,
        update_every : int = 1,
        log_message : bool = True
    ) -> Generator[T, None, None]:
        
        total = len(it)

        self.send_info_event(
            progress=0.0,
            task=task,
            message=f"Starting {task.replace('-', ' ')}...",
            extra={ "log_message" : log_message }
        )

        for i, item in enumerate(it):

            yield item

            if (i % update_every == 0) or ((i+1) == total):

                self.send_info_event(
                    progress=(i+1)/total,
                    task=task,
                    message=f"{task.replace('-', ' ')} : {i+1}/{total} elements processed.",
                    extra={ "log_message" : log_message }
                )