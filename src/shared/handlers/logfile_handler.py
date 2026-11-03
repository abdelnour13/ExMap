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

# class FileLogger(Handler):

#     def __init__(self,
#         name : str, 
#         log_file : str
#     ) -> None:
#         super().__init__()
#         self.name = name
#         self.log_file = log_file
#         self.f = open(self.log_file, "a")

#     def create_logger(self) -> Logger:

#         logger = getLogger(name="logfile")
#         logger.setLevel(logging.DEBUG)

#         handler = FileHandler(filename=self.log_file)
#         handler.setLevel(logging.DEBUG)
#         formatter = Formatter("%(_name)s,%(_timestamp)s,%(_level)s,%(total_processed)s,%(total_saved)s,%(total_skipped)s,%(message)s,%(progress)s")

#         handler.setFormatter(formatter)
#         logger.addHandler(handler)

#         return logger

#     def consume(self, event : Event) -> None:

#         raw = asdict(event)
#         line = []

#         for key, value in raw.items():

#             if key == 'extra':
                
#                 for k, v in value.items():
#                     line.append(f"{k}={v}")

#                 continue

#             if key == 'task':
#                 value = f'{self.name}::{value}'

#             line.append(f"{key}={value}")

#         line = ",".join(line)

#         self.f.write(f'{line}\n')