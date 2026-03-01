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

@dataclass
class Event:
    progress : Optional[float] = None # Estimated Progress if available
    timestamp : str = field(default_factory=lambda : datetime.now().strftime("%Y-%m-%d %H:%M:%S")) # Timestamp
    level : EventLevel = EventLevel.INFO # Level
    message : Optional[str] = None # Message
    extra : Dict = field(default_factory=dict)

    # total_processed : int = 0 # Total records already processed
    # total_saved : int = 0 # Total records processed and saved to disk
    # total_skipped : int = 0 # Total records skipped due to an exception
    

class Handler(ABC):

    @abstractmethod
    def consume(self, event : Event) -> Event:
        ...