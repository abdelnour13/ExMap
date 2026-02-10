import httpx
import math
import time
import json
import tomllib
from typing import Optional, Generator, Dict, Union, List
from src.shared.command_tools import BaseCommand
from src.shared.handlers import ConsoleLogger, Event, EventLevel
from src.shared.file_tools import FileWriter
from returns.result import Result, safe
from returns.primitives.exceptions import UnwrapFailedError
from pydantic import BaseModel, Field
from typing import List
from instance import cli

class Query(BaseModel):
    teams : List[str]
    format : str
    attributes : List[str]
    from_ : str
    to : str
    languages : List[str]
    sort : str

class HALConfig(BaseModel):
    query : Query
    page_size : int
    num_pages : Optional[int] = None

class HAL_API:

    BASE_URL : str = "https://api.archives-ouvertes.fr/search"

    def __init__(self, config : HALConfig) -> None:
        self.config = config
        self.__num_pages : Optional[Result[int, Union[httpx.TransportError,httpx.HTTPStatusError]]] = None

    def url(self, page_id : int = 0) -> str:

        return ''.join([
            HAL_API.BASE_URL,
            f'?wt={self.config.query.format}',
            f'&fl={",".join(self.config.query.attributes)}',
            f'&fq=collCode_s:({" OR ".join(self.config.query.teams)})'
            f'&fq=producedDate_tdate:[{self.config.query.from_} TO {self.config.query.to}]',
            f'&fq=language_s:({" OR ".join(self.config.query.languages)})',
            f'&sort={self.config.query.sort}',
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
                lambda x : x in self.config.query.teams,
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

class DataDownloader(BaseCommand):

    def __init__(self, 
        api : HAL_API, 
        writer : FileWriter, 
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
            total_processed=self._num_processed,
            total_skipped=self._num_skipped,
            total_saved=self._num_saved,
            message=message
        )
            
    def _handle_fetch_page(self) -> Generator[Event, None, None]:

        page_id, num_pages = self._current_page, self._num_pages
        page = self.api.fetch_page(page_id)
        progress = None if num_pages is None else (page_id+1)/num_pages
        self._current_page += 1

        try:

            page = page.unwrap()
            yield self._make_event(progress, EventLevel.INFO,message=f"Downloaded {page_id} has a total of {len(page)} documents.")

            self.writer.write_chunk(page)
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

    def __iter__(self) -> Generator[Event, None, None]:

        ### Get the number of pages
        yield self._handle_num_pages_response()
        
        ### Fetch Pages
        for page_id in range(self._num_pages):
            yield from self._handle_fetch_page()

        yield self._make_event(progress=1.0,level=EventLevel.INFO,message=f"Download of {self._num_pages} has been completed.")

@cli.register_command(
    name="download",
    description="Download the articles & their abstracts that will be used later to create the graph.",
    add_from_conf=False,
    add_save_to=False
)
def download(
    config : str = Field(description="Path to a config.toml file."), 
    output_filename : str = Field(description="The path to the file to save the downloaded data to.")
) -> None:
    
    with open(config, 'rb') as f:
        data = tomllib.load(f)
        config = HALConfig(**data)

    downloader = DataDownloader(
        api=HAL_API(config),
        writer=FileWriter.from_filename(output_filename)
    ).add_handler(ConsoleLogger("DataDownloader"))

    downloader.run()