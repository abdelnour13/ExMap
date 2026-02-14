import os
from src.helpers.config import makedirs
from typing import Any, Generator,Protocol,List, TypeVar, Self, Optional, Annotated
from abc import ABC,abstractmethod
from itertools import chain
from pycli import toml_parser
from pydantic import Field, BaseModel, BeforeValidator, computed_field

T = TypeVar('T')

class Connection(BaseModel):
    host : str = Field(description="The host of the socket connection.")
    port : str = Field(description="The port of the socket connection.")

class BaseConfig(BaseModel):

    connection : Optional[Annotated[Connection, BeforeValidator(toml_parser)]] = Field(
        default=None, 
        description="The socket connection used to send updates through."
    )

    output_dir : Annotated[str, BeforeValidator(makedirs)] = Field(
        description="The path of the directory to save the result to."
    )

    format : str = Field(default="jsonl", description="The format to save result.")

    resume : bool = Field(default=False, description="If true it resumes the job skipping the documents already processed.")

    @computed_field
    @property
    def output_filename(self) -> str:
        return os.path.join(self.output_dir, f"data.{self.format}")
    
    @computed_field
    @property
    def log_filename(self) -> str:
        return os.path.join(self.output_dir, f"output.log")
    
    @computed_field
    @property
    def checkpoint_filename(self) -> str:
        return os.path.join(self.output_dir, f"checkpoint")
    
    def model_post_init(self, context : Any) -> None:
        
        if not self.resume and os.path.exists(self.output_dir):
            
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))

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
    def __iter__(self) -> Generator[Any, None, None]:
        ...

    def before_start(self) -> Generator[Any, None, None]:
        yield from []

    def after_end(self) -> Generator[Any, None, None]:
        yield from []

    @classmethod
    def from_config(cls, config : T) -> Self:
        raise NotImplementedError

    def run(self) -> None:

        for event in chain(self.before_start(), self, self.after_end()):
            for handler in self.handlers:
                handler.consume(event)