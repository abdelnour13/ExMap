import os
import torch
from pydantic import BaseModel, BeforeValidator, ValidationError
from functools import cached_property
from typing import List, Optional, Annotated
from definitions import domain_dir
from src import helpers

class Domain(BaseModel):
    sub_domain : str
    definition : Optional[str] = None
    domain : Optional[str] = None
    alt_names : Optional[List[str]] = None

def file_exists(filename : str) -> str:

    if not os.path.exists(filename):
        raise ValidationError({ "attribute" : filename, "error" : f"file {filename} doesn't exists" })

def device(device : Optional[int]) -> int:

    if device is None:
        return 0 if torch.cuda.is_available() else -1

    return device

def DomainConfig(name : str = "domain"):

    class _Domain(BaseModel):

        @cached_property
        def domains(self) -> List[Domain]:
            return [Domain(**domain) for domain in helpers.load_json_file(domain_dir(getattr(self, name)))]

    return _Domain

FileExists = Annotated[str, BeforeValidator(file_exists)]
Device = Annotated[int, BeforeValidator(device)]

if __name__ == '__main__':

    class Config(DomainConfig('domain')):
        domain : str = 'industry_of_future'

    config = Config()

    print(config.domains)