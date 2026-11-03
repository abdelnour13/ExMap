import os
import httpx
import json
from typing import Optional, Generator, Dict, List, Self, Literal
from src.helpers import load_json_file
from src.shared.command_tools import BaseCommand, ConnectionOption, OutputDirOption, add_handlers
from src.shared.handlers import Event, EventLevel
from src.shared.file_tools import BufferedWriter, FileWriter
from pydantic import BaseModel, Field, computed_field, field_validator
from definitions import CORPUS_DIR, rel_path
from src.commands.data_download.api_list import ApiList, Api
from src.commands.data_download.hal import HAL_API
from src.commands.data_download.pubmedia import PUBMED_API
from returns.pipeline import is_successful
from functools import cached_property
from instance import cli

### ---------- Config ----------

API_TYPE = Literal["hal", "pubmedia"]

class BaseAPIConfig(BaseModel):
    teams : List[str]
    from_ : str = "*"
    to : str = "*"
    page_size : int = 32

class APIConfig(BaseModel):
    teams : Dict[API_TYPE, List[str]]
    from_ : str = "*"
    to : str = "*"
    page_size : int = 32


class Config(ConnectionOption, OutputDirOption):

    corpus_id : str = Field(description=f"Name of folder located in {rel_path(CORPUS_DIR)} and contains a `config.json` file.")

    def output_dir(self):
        return os.path.join(CORPUS_DIR, self.corpus_id)

    @computed_field
    @cached_property
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
    def api_config(self) -> APIConfig:
        return APIConfig(**load_json_file(self.config_path))

    def get_api_config(self, source : str) -> Optional[BaseAPIConfig]:

        if source not in self.api_config.teams:
            return None

        return BaseAPIConfig(
            teams=self.api_config.teams,
            from_=self.api_config.from_,
            to=self.api_config.to,
            page_size=self.api_config.page_size
        )


### ---------- Stage ----------

@cli.register_stage(
    name="download",
    description="Download the articles & their abstracts that will be used later to create the graph.",
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class DataDownloader(BaseCommand):

    def __init__(self,
        api : ApiList,
        writer : BufferedWriter,
    ) -> None:
        super().__init__()

        self.api = api
        self.writer = writer

        self._num_pages : Optional[int] = None
        self._num_processed = 0
        self._num_saved = 0
        self._num_skipped = 0

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

    @staticmethod
    def _describe_error(error : Exception) -> str:

        if isinstance(error, httpx.TransportError):
            return "a connection error"

        if isinstance(error, httpx.HTTPStatusError):

            payload = json.dumps({
                'status' : error.response.status_code,
                'message' : error.response.text
            })

            return f"an HTTP error : {payload}"

        return f"the following error : {error}"

    def _handle_num_pages(self) -> Event:

        self._num_pages = self.api.num_pages()

        if self._num_pages is None:
            return self._make_event(progress=None, level=EventLevel.CRITICAL, message="Failed to fetch the number of pages.")

        return self._make_event(progress=0, level=EventLevel.INFO, message=f"A total of {self._num_pages} pages to download.")

    def _fetch_all(self) -> Generator[Event, None, None]:

        for page_id in range(self._num_pages):

            progress = (page_id + 1) / self._num_pages
            page = self.api.fetch_page(page_id)
            self._num_processed += 1

            if not is_successful(page):

                self._num_skipped += 1
                reason = DataDownloader._describe_error(page.failure())

                yield self._make_event(progress, EventLevel.ERROR, message=f"Failed to fetch page {page_id} due to {reason}.")
                continue

            docs = page.unwrap()
            yield self._make_event(progress, EventLevel.INFO, message=f"Page {page_id} downloaded with {len(docs)} documents.")

            self.writer.write(docs)
            self._num_saved += 1

            yield self._make_event(progress, EventLevel.INFO, message=f"Page {page_id} with {len(docs)} documents was saved to disk.")

        self.writer.flush()

    def __iter__(self) -> Generator[Event, None, None]:

        ### Get the number of pages
        yield self._handle_num_pages()

        if self._num_pages is None:
            return

        ### Fetch Pages
        yield from self._fetch_all()

        ### Final Event
        yield self._make_event(progress=1.0, level=EventLevel.INFO, message=f"Download of {self._num_pages} pages has been completed.")

    @classmethod
    def from_config(cls, config : Config) -> Self:

        conf = config.api_config

        apis = []

        if 'hal' in conf.teams:

            api = HAL_API(
                teams=conf.teams['hal'],
                from_=conf.from_,
                to=conf.to,
                page_size=conf.page_size,
            )

            apis.append(api)

        if 'pubmedia' in conf.teams:

            api = PUBMED_API(
                teams=conf.teams['pubmedia'],
                from_=conf.from_,
                to=conf.to,
                page_size=conf.page_size,
            )

            apis.append(api)

        return cls(
            api=ApiList(apis),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(conf.page_size)
        )