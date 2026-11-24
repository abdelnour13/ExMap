import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Optional, Dict

class EventLevel(Enum):
    INFO = logging.INFO
    WARN = logging.WARNING
    ERROR = logging.ERROR
    DEBUG = logging.DEBUG
    CRITICAL = logging.CRITICAL
    START = 60
    END = 70
    CANCELLED = 80

@dataclass
class Event:
    task : str = "main" # The task currently executed by the command
    progress : Optional[float] = None # Estimated Progress if available
    timestamp : str = field(default_factory=lambda : datetime.now().strftime("%Y-%m-%d %H:%M:%S")) # Timestamp
    level : EventLevel = EventLevel.INFO # Level
    message : Optional[str] = None # Message
    extra : Dict = field(default_factory=dict)
    

class Handler(ABC):

    @abstractmethod
    def consume(self, event : Event) -> Event:
        ...