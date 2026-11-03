import os
import shutil
from functools import wraps
from src.shared.handlers import SocketHandler, ConsoleLogger, FileLogger
from typing import Any, TypeVar, Optional, Annotated
from pycli import json_parser
from pydantic import Field, BaseModel, BeforeValidator, computed_field

T = TypeVar('T')

class Connection(BaseModel):
    host : str = Field(description="The host of the socket connection.")
    port : int = Field(description="The port of the socket connection.")

class InputFileOption(BaseModel):
    # Annotated[str, BeforeValidator(assert_file_exists)]
    input_filename : str = Field(description="The path to the file holding the data.")
    batch_size : int = Field(default=1,description="How many documents to process on a single pass.")
    buffer_size : int = Field(default=4,description="The size of the buffer used to update the file on the disk.")

class ConnectionOption(BaseModel):
    connection : Optional[Annotated[Connection, BeforeValidator(json_parser)]] = Field(
        default=None, 
        description="The socket connection used to send updates through."
    )

class OutputDirOption(BaseModel):
    out : Optional[str] = Field(default=None)
    format : str = Field(default="jsonl", description="The format to save result.")
    resume : bool = Field(default=True, description="If true it resumes the job skipping the documents already processed.")
    cache : bool = Field(default=True, description="If true it saves a stage checkpoint that will be used to resume the job later.")
    log : bool = Field(default=True, description="If true it logs to a file.")

    def output_dir(self) -> str:

        if self.out is None:
            raise NotImplementedError
        
        return self.out

    @computed_field
    @property
    def output_filename(self) -> str:
        return os.path.join(self.output_dir(), f"data.{self.format}")
    
    @computed_field
    @property
    def log_filename(self) -> str:
        return os.path.join(self.output_dir(), "log.jsonl")
    
    @computed_field
    @property
    def checkpoint_filename(self) -> str:
        return os.path.join(self.output_dir(), "checkpoint")
    
    def model_post_init(self, context : Any) -> None:
        
        if not self.resume and os.path.exists(self.output_dir()):
            
            for file in [self.output_filename, self.log_filename, self.checkpoint_filename]:

                if not os.path.exists(file):
                    continue

                if os.path.isdir(file):
                    shutil.rmtree(file)
                else:
                    os.remove(file)

        # os.makedirs(self.output_dir(), exist_ok=True)


class __Config(OutputDirOption, ConnectionOption):
    pass

def add_handlers(cls : T):

    from_config = cls.__dict__['from_config'].__func__    

    @wraps(from_config)
    def new_from_config(cls, config : __Config):

        instance = from_config(cls, config)

        if config.connection is not None:
            instance.add_handler(SocketHandler(host=config.connection.host, port=config.connection.port))

        instance.add_handler(ConsoleLogger(cls.__name__))

        if config.log:
            instance.add_handler(FileLogger(name=cls.__name__, log_file=config.log_filename))

        return instance

    cls.from_config = classmethod(new_from_config)

    return cls