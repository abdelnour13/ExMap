import os
import httpx
import math
import time
import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime
from collections import defaultdict
from itertools import batched
from typing import Optional, Generator, Dict, Union, List, Self
from src.helpers import load_json_file
from src.shared.command_tools import BaseCommand, ConnectionOption, OutputDirOption, add_handlers
from src.shared.handlers import Event, EventLevel
from src.shared.file_tools import BufferedWriter, FileWriter
from returns.result import Result, safe
from returns.pipeline import is_successful
from returns.primitives.exceptions import UnwrapFailedError
from pydantic import BaseModel, Field, computed_field, field_validator
from definitions import CORPUS_DIR, rel_path
from instance import cli

PageResult = Result[List[Dict], httpx.TransportError | httpx.HTTPStatusError]

class BaseAPIConfig(BaseModel):
    teams : List[str]
    format : str
    attributes : List[str]
    from_ : str
    to : str
    languages : List[str]
    sort : str
    page_size : int
    num_pages : Optional[int]

class APIConfig(BaseModel):
    teams : Dict[str, List[str]]
    format : str
    attributes : List[str]
    from_ : str
    to : str
    languages : List[str]
    sort : str
    page_size : int
    num_pages : Optional[int]

    def get_config(self, source  : str) -> BaseAPIConfig:

        dump = self.model_dump()
        teams = dump.pop('teams')
        teams = teams[source]

        return BaseAPIConfig(
            **dump,
            teams=teams
        )

class Config(ConnectionOption, OutputDirOption):

    corpus_id : str = Field(description=f"Name of folder located in {rel_path(CORPUS_DIR)} and contains a `config.json` file.")

    def output_dir(self):
        return os.path.join(CORPUS_DIR, self.corpus_id)
    
    @computed_field
    @property
    def config_path(self) -> str:
        return os.path.join(self.output_dir(), 'config.json')
    
    @field_validator('corpus_id')
    @classmethod
    def validate(cls, value: str) -> str:

        path = os.path.join(CORPUS_DIR, value)

        if not os.path.exists(path):
            raise ValueError(f"Path {path} not found, no such a file or directory.")
        
        config_file = os.path.join(path, "config.json")

        if not os.path.exists(config_file):
            raise ValueError(f"No config file found {path} not found, no such a file or directory.")
        
        return value

    @computed_field
    @property
    def hal_conf(self) -> APIConfig:
        return APIConfig(**load_json_file(self.config_path))

class Api(ABC):

    def __init__(self):
        self._num_pages = None

    @safe(exceptions=(httpx.TransportError,httpx.HTTPStatusError))
    def make_request(self, url : str, max_retries : int = 3) -> httpx.Response:
        
        response = None
        
        for attempt in range(max_retries):
            
            try:

                response = httpx.get(url=url)

                if response.status_code in [
                    httpx.codes.TOO_MANY_REQUESTS,
                    httpx.codes.INTERNAL_SERVER_ERROR,
                    httpx.codes.BAD_GATEWAY,
                    httpx.codes.SERVICE_UNAVAILABLE,
                    httpx.codes.GATEWAY_TIMEOUT
                ]:
                    time.sleep(2 ** (attempt * 0.8))   
                else:
                    break

            except httpx.TransportError as e:
                
                if attempt == (max_retries - 1):
                    raise e
                    
                time.sleep(2 ** (attempt * 0.8)) 

        return response.raise_for_status()

    @abstractmethod
    def url(self, page_id : int) -> str:
        ...

    @abstractmethod
    def get_data(self, response : httpx.Response) -> List[Dict]:
        ...

    @abstractmethod
    def normalize_document(self, document : Dict) -> Dict:
        ...

    def fetch_page(self, page_id : int) -> PageResult:

        url = self.url(page_id=page_id)
        response = self.make_request(url=url)
        data = response.map(self.get_data)
        data = data.map(lambda docs : [self.normalize_document(doc) for doc in docs])

        return data

    @abstractmethod
    def fetch_num_pages(self) -> Result[int, Union[httpx.TransportError, httpx.HTTPStatusError]]:
        ...
    
    def num_pages(self) -> Optional[int]:

        if self._num_pages is not None:
            return self._num_pages

        num_pages = self.fetch_num_pages()

        if is_successful(num_pages):
            self._num_pages = num_pages.unwrap()

        return self._num_pages
