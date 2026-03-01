import os
import shutil
from src.helpers.config import makedirs, assert_file_exists
from src.shared.handlers import Event, EventLevel
from typing import Any, Generator,Protocol,List, TypeVar, Self, Optional, Annotated, Dict
from abc import ABC,abstractmethod
from itertools import chain
from pycli import toml_parser
from pydantic import Field, BaseModel, BeforeValidator, computed_field

T = TypeVar('T')

class Connection(BaseModel):
    host : str = Field(description="The host of the socket connection.")
    port : str = Field(description="The port of the socket connection.")

class InputFileOption(BaseModel):

    input_filename : Annotated[str, BeforeValidator(assert_file_exists)] = Field(
        description="The path to the file holding the data."
    )
    
    batch_size : int = Field(default=1,description="How many documents to process on a single pass.")
    buffer_size : int = Field(default=4,description="The size of the buffer used to update the file on the disk.")

class ConnectionOption(BaseModel):

    connection : Optional[Annotated[Connection, BeforeValidator(toml_parser)]] = Field(
        default=None, 
        description="The socket connection used to send updates through."
    )

class OutputDirOption(BaseModel):

    output_dir : Annotated[str, BeforeValidator(makedirs)] = Field(
        description="The path of the directory to save the result to."
    )

    format : str = Field(default="jsonl", description="The format to save result.")

    resume : bool = Field(default=False, description="If true it resumes the job skipping the documents already processed.")

    cache : bool = Field(default=True, description="If true it saves a stage checkpoint that will be used to resume the job later.")

    log : bool = Field(default=True, description="If true it logs to a file.")

    @computed_field
    @property
    def output_filename(self) -> str:
        return os.path.join(self.output_dir, f"data.{self.format}")
    
    @computed_field
    @property
    def log_filename(self) -> str:
        return os.path.join(self.output_dir, "output.log")
    
    @computed_field
    @property
    def checkpoint_filename(self) -> str:
        return os.path.join(self.output_dir, "checkpoint")
    
    def model_post_init(self, context : Any) -> None:
        
        if not self.resume and os.path.exists(self.output_dir):
            
            for file in os.listdir(self.output_dir):

                filename = os.path.join(self.output_dir, file)

                if os.path.isdir(filename):
                    shutil.rmtree(filename)
                else:
                    os.remove(filename)

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

        stop = False

        for event in chain(self.before_start(), self, self.after_end()):

            if stop:
                break

            for handler in self.handlers:

                handler.consume(event)

                if event.level == EventLevel.CRITICAL:
                    event = Event(**dict(event))
                    event.message = "Stopping job due to a critical error, check logs for details."
                    handler.consume(event)
                    stop = True

    def make_event(self,
        progress : Optional[float],
        level : EventLevel,
        message : str,
        extra : Dict
    ) -> Event:
        
        return Event(
            progress=progress,
            level=level,
            message=message,
            extra=extra
        )
    
    def make_debug_event(self,
        progress : Optional[float],
        message : str,
        extra : Dict
    ) -> Event:
        return self.make_event(progress, EventLevel.DEBUG, message, extra)
    
    def make_info_event(self,
        progress : Optional[float],
        message : str,
        extra : Dict
    ) -> Event:
        return self.make_event(progress, EventLevel.INFO, message, extra)
    
    def make_error_event(self,
        progress : Optional[float],
        message : str,
        extra : Dict
    ) -> Event:
        return self.make_event(progress, EventLevel.ERROR, message, extra)
    
    def make_warn_event(self,
        progress : Optional[float],
        message : str,
        extra : Dict
    ) -> Event:
        return self.make_event(progress, EventLevel.WARN, message, extra)