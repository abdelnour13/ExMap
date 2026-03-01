import logging
from logging import FileHandler, getLogger, Logger, Formatter
from .base import Handler, Event

class FileLogger(Handler):

    def __init__(self,
        name : str, 
        log_file : str
    ) -> None:
        super().__init__()
        self.name = name
        self.log_file = log_file
        self.logger = self.create_logger()

    def create_logger(self) -> Logger:

        logger = getLogger(name="logfile")
        logger.setLevel(logging.DEBUG)

        handler = FileHandler(filename=self.log_file)
        handler.setLevel(logging.DEBUG)
        formatter = Formatter("%(_name)s,%(_timestamp)s,%(_level)s,%(total_processed)s,%(total_saved)s,%(total_skipped)s,%(message)s,%(progress)s")

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def consume(self, event : Event) -> None:
        
        self.logger.log(
            level=event.level.value,
            msg=event.message,
            extra={
                "_name" : self.name,
                "_timestamp" : event.timestamp,
                "_level" : event.level.name,
                **event.extra,
                "progress" : f"{(100 * event.progress):.2f}" if event.progress else None
            }
        )