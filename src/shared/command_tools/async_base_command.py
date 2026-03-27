from src.shared.handlers import Event, EventLevel
from typing import Any,Protocol,List, TypeVar, Self, Optional, Dict
from abc import ABC,abstractmethod

import asyncio

class Handler(Protocol):

    def consume(self, event : Any):
        ...

T = TypeVar('T')

class BaseCommand(ABC):

    def __init__(self):
        super().__init__()
        self.handlers : List[Handler] = []
        self.events_queue = asyncio.Queue[Event](100)

    def add_handler(self, handler : Handler) -> 'BaseCommand':
        self.handlers.append(handler)
        return self
    
    async def event_consumer(self) -> None:

        while True:

            event = await self.events_queue.get()

            for handler in self.handlers:
                handler.consume(event)

            self.events_queue.task_done()

    async def before_start(self) -> bool:
        return True

    async def after_end(self) -> bool:
        return True

    @abstractmethod
    async def generate(self) -> bool:
        ...

    async def _run(self) -> None:

        consume_task = asyncio.create_task(self.event_consumer())

        try:
            await self.before_start()
            await self.generate()
            await self.after_end()

        except Exception as e:

            await self.emit_warn_event(
                progress=0.0,
                message=f"Shutting down due to an unexpected error waiting for remaining events to be processed. error : {e}."
            )

        finally:

            await self.events_queue.join()
            consume_task.cancel()

    def run(self) -> None:
        asyncio.run(self._run())

    @classmethod
    def from_config(cls, config : T) -> Self:
        raise NotImplementedError
    
    async def emit_event(self,
        progress : Optional[float],
        level : EventLevel,
        message : str,
        extra : Optional[Dict] = None
    ) -> None:
        
        await self.events_queue.put(Event(
            progress=progress,
            level=level,
            message=message,
            extra=(extra or {})
        ))

    async def emit_critical_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> None:
        await self.emit_event(progress, EventLevel.CRITICAL, message, extra)
    
    async def emit_debug_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> None:
        await self.emit_event(progress, EventLevel.DEBUG, message, extra)
    
    async def emit_info_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> None:
        await self.emit_event(progress, EventLevel.INFO, message, extra)
    
    async def emit_error_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> None:
        await self.emit_event(progress, EventLevel.ERROR, message, extra)
    
    async def emit_warn_event(self,
        progress : Optional[float],
        message : str,
        extra : Optional[Dict] = None
    ) -> None:
        await self.emit_event(progress, EventLevel.WARN, message, extra)