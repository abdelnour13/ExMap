"""
Data download stage.
============================================
Raw data download from HAL API based on a config file found at : /data/corpus/<corpus-id>/config.json.

The config format is a json file with the following schema :

{
    "teams" : [
        "TEAM_1",
        "TEAM_2",
        ...
    ],
    "format" : "jsonl",
    "attributes" : [
        "ATTR_1",
        "ATTR_2",
        ...
    ],
    "from_" : "2020-01-01T00:00:00Z" # (ISO Date or `*`)
    "to" : "*" (ISO Date or `*`),
    "languages" : [
        "LANG_1",
        "LANG_2",
        ...
    ],
    "sort" : "ATTR",
    "page_size" : 32,
    "num_pages" : null # int or null
}
"""

import os
import httpx
import math
import time
import json
from typing import Optional, Generator, Dict, Union, List, Self
from src.helpers import load_json_file
from src.shared.command_tools import BaseCommand, ConnectionOption, OutputDirOption, add_handlers
from src.shared.handlers import ConsoleLogger, Event, EventLevel
from src.shared.file_tools import BufferedWriter, FileWriter
from returns.result import Result, safe
from returns.primitives.exceptions import UnwrapFailedError
from pydantic import BaseModel, Field, computed_field, field_validator
from definitions import CORPUS_DIR, rel_path
from instance import cli

class HALConfig(BaseModel):
    teams : List[str]
    format : str
    attributes : List[str]
    from_ : str
    to : str
    languages : List[str]
    sort : str
    page_size : int
    num_pages : Optional[int]

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
    def hal_conf(self) -> HALConfig:
        return HALConfig(**load_json_file(self.config_path))
    
class HAL_API:

    BASE_URL : str = "https://api.archives-ouvertes.fr/search"

    def __init__(self, config : HALConfig) -> None:
        self.config = config
        self.__num_pages : Optional[Result[int, Union[httpx.TransportError,httpx.HTTPStatusError]]] = None

    def url(self, page_id : int = 0) -> str:

        return ''.join([
            HAL_API.BASE_URL,
            f'?wt={self.config.format}',
            f'&fl={",".join(self.config.attributes)}',
            f'&fq=collCode_s:({" OR ".join(self.config.teams)})'
            f'&fq=producedDate_tdate:[{self.config.from_} TO {self.config.to}]',
            f'&fq=language_s:({" OR ".join(self.config.languages)})',
            f'&sort={self.config.sort}',
            f'&rows={self.config.page_size}',
            f'&start={page_id * self.config.page_size}',
            f'&fq=abstract_s:["" TO *]',
        ])
    
    @safe(exceptions=(httpx.TransportError,httpx.HTTPStatusError))
    def make_request(self, 
        page_id : int,
        max_retries : int = 3
    ) -> httpx.Response:
        
        url = self.url(page_id=page_id)
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
    
    def normalize_document(self, doc : Dict) -> Dict:

        normalized_doc = dict()

        normalized_doc["id"] = doc["docid"]
        normalized_doc["title"] = doc.get("title_s", [None])[0]
        normalized_doc["abstract"] = doc.get("abstract_s", [None])[0]
        normalized_doc["keywords"] = doc.get("keyword_s", [])
        normalized_doc["authors"] = doc.get("authFullName_s", [])
        normalized_doc["venues"] = doc.get("domainAllCode_s", [])

        normalized_doc["teams"] = list(
            filter(
                lambda x : x in self.config.teams,
                doc.get("collCode_s", [])
            )
        )

        normalized_doc["date"] = doc["producedDate_tdate"]

        return normalized_doc
    
    def num_pages(self) -> Result[int, Union[httpx.TransportError, httpx.HTTPStatusError]]:

        compute = self.__num_pages is None

        if not compute:

            try:
                self.__num_pages.unwrap()
            except:
                compute = True
        
        if compute:

            response = self.make_request(0)

            # print(httpx.Response.json(response.unwrap()))

            num_pages = response.map(httpx.Response.json) \
                .map(lambda x : x['response']['numFound']) \
                .map(lambda x : math.ceil(x / self.config.page_size)) \
                .map(lambda x : x if self.config.num_pages is None else min(x, self.config.num_pages))
                        
            self.__num_pages = num_pages

        return self.__num_pages
    
    def fetch_page(self, page_id : int) -> Result[List[Dict], Union[httpx.TransportError, httpx.HTTPStatusError]]:
                
        return self.make_request(page_id) \
            .map(httpx.Response.json) \
            .map(lambda x :  x['response']['docs']) \
            .map(lambda x : list(map(self.normalize_document, x)))


@cli.register_stage(
    name="download",
    description="Download the articles & their abstracts that will be used later to create the graph.",
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class DataDownloader(BaseCommand):

    def __init__(self, 
        api : HAL_API, 
        writer : BufferedWriter, 
    ) -> None:
        super().__init__()
        
        self.api = api
        self.writer = writer

        self._current_page = 0
        self._num_pages = 0
        self._num_processed = 0
        self._num_saved = 0
        self._num_skipped = 0
    
    def _handle_num_pages_response(self) -> Event:

        num_pages = self.api.num_pages()

        try:

            num_pages = num_pages.unwrap()
            self._num_pages = num_pages
            return self._make_event(progress=0,level=EventLevel.INFO,message=f"A Total of {num_pages} pages to download.")
        
        except UnwrapFailedError as e:

            error = num_pages.failure()

            if isinstance(error, httpx.TransportError):
                return self._make_event(progress=None,level=EventLevel.ERROR,message=f"Failed to fetch number of pages due to a connection error.")
            elif isinstance(error, httpx.HTTPStatusError):

                payload = json.dumps({
                    'status' : error.response.status_code,
                    'message' : error.response.text
                })

                return self._make_event(progress=None,level=EventLevel.ERROR,message=f"Failed to fetch number of pages due to an HTTP error : {payload}.")
    
    def _make_event(self, progress : Optional[float], level : EventLevel, message : str) -> Event:

        return Event(
            progress=progress,
            level=level,
            message=message,
            extra={
                'num_processed' : self._num_processed,
                'total_skipped' : self._num_skipped,
                'total_saved' : self._num_saved
            }
        )
            
    def _handle_fetch_page(self) -> Generator[Event, None, None]:

        page_id, num_pages = self._current_page, self._num_pages
        page = self.api.fetch_page(page_id)
        progress = None if num_pages is None else (page_id+1)/num_pages
        self._current_page += 1

        try:

            page = page.unwrap()
            yield self._make_event(progress, EventLevel.INFO,message=f"Downloaded {page_id} has a total of {len(page)} documents.")

            self.writer.write(page)
            self._num_processed += 1
            self._num_saved += 1

            yield self._make_event(progress, EventLevel.INFO,message=f"Page {page_id} with {len(page)} documents was saved to disk.")

        except UnwrapFailedError as e:

            error = page.failure()
            self._num_skipped += 1
            page = None

            if isinstance(error, httpx.TransportError):
                yield self._make_event(progress, EventLevel.ERROR,message=f"Failed to fetch page {page_id} due to a connection error.")
            elif isinstance(error, httpx.HTTPStatusError):

                payload = json.dumps({
                    'status' : error.response.status_code,
                    'message' : error.response.text
                })

                yield self._make_event(progress, EventLevel.ERROR, message=f"Failed to fetch number of pages due to an HTTP error : {payload}.")

        self.writer.flush()

    def __iter__(self) -> Generator[Event, None, None]:

        ### Get the number of pages
        yield self._handle_num_pages_response()
        
        ### Fetch Pages
        for _ in range(self._num_pages):
            yield from self._handle_fetch_page()

        yield self._make_event(progress=1.0,level=EventLevel.INFO,message=f"Download of {self._num_pages} has been completed.")

    @classmethod
    def from_config(cls, config : Config) -> Self:

        downloader = DataDownloader(
            api=HAL_API(config.hal_conf),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.hal_conf.page_size)
        )

        return downloader