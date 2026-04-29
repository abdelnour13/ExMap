import logging
from rich.logging import RichHandler
from rich.progress import Progress, TaskID
from logging import getLogger, Logger
from .base import Event, Handler

class ConsoleLogger(Handler):

    def __init__(self, name : str) -> None:
        super().__init__()

        self.name = name
        self.tasks : dict[str, TaskID] = {}

        self.logger = self.create_logger()
        self.pbar = Progress()
        self.pbar.start()

    def create_logger(self) -> Logger:

        ### Create Logger
        logger = getLogger(self.name)
        logger.setLevel(logging.DEBUG)

        logger.propagate = False

        ### Terminal Handler
        handler = RichHandler(markup=True)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        return logger

    def consume(self, event : Event) -> None:

        task_id = f'{self.name}::{event.task}'

        if task_id not in self.tasks and event.progress is not None:
            self.tasks[task_id] = self.pbar.add_task(task_id, total=1.0)

        if event.progress is not None:

            fmt = "[bold]{} - " + " - ".join([f"{key} : {{}}" for key in event.extra]) + " - Message : {} - Progress : {}"

            message = fmt.format(
                task_id,
                *event.extra.values(),
                event.message,
                f"{100 * event.progress:.2f}"
            )

        else :

            fmt = "[bold]{} - " + " - ".join([f"{key} : {{}}" for key in event.extra]) + " - Message : {}"

            message = fmt.format(
                task_id,
                *event.extra.values(),
                event.message,
            )            

        self.logger.log(level=event.level.value,msg=message)

        if event.progress is not None:
            task = self.pbar._tasks[self.tasks[task_id]]
            self.pbar.update(self.tasks[task_id], advance=event.progress - task.completed, refresh=True)