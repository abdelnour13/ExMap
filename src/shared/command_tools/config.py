import os
import shutil
from src.helpers.config import makedirs, assert_file_exists
from typing import Any, TypeVar, Optional, Annotated
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