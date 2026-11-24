import json
from dataclasses import asdict
from .base import Handler, Event

class FileLogger(Handler):

    def __init__(self,
        name : str, 
        log_file : str
    ) -> None:
        super().__init__()
        self.name = name
        self.log_file = log_file
        self.f = open(self.log_file, "a")

    def consume(self, event : Event) -> None:
        raw = asdict(event)
        raw["level"] = event.level.name
        line = json.dumps(raw)
        self.f.write(f'{line}\n')
        self.f.flush()