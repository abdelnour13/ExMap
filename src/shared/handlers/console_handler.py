import logging
from rich.logging import RichHandler
from rich.progress import Progress
from logging import getLogger, Logger
from .base import Event, Handler

class ConsoleLogger(Handler):

    def __init__(self, name : str) -> None:
        super().__init__()

        self.name = name

        self.logger = self.create_logger()
        self.pbar = Progress()
        
        self.progress_task = self.pbar.add_task(f"[blue]{name}", total=100)
        self.previous = 0.0
        self.pbar.start()

    def create_logger(self) -> Logger:

        ### Create Logger
        logger = getLogger(self.name)
        logger.setLevel(logging.DEBUG)

        ### Terminal Handler
        handler = RichHandler(markup=True)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        return logger

    def consume(self, event : Event) -> None:

        if event.progress is not None:

            fmt = "[bold]{} - " + " - ".join([f"{key} : {{}}" for key in event.extra]) + " - Message : {} - Progress : {}"

            message = fmt.format(
                self.name,
                *event.extra.values(),
                event.message,
                f"{100 * event.progress:.2f}"
            )

        else :

            fmt = "[bold]{} - " + " - ".join([f"{key} : {{}}" for key in event.extra]) + " - Message : {}"

            message = fmt.format(
                self.name,
                *event.extra.values(),
                event.message,
            )            

        self.logger.log(level=event.level.value,msg=message)

        if event.progress is not None:

            progress = 100 * event.progress - self.previous
            self.previous += progress
            self.pbar.update(self.progress_task, advance=progress, refresh=True)