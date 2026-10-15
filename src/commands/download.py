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
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import defaultdict
from itertools import batched
from typing import Optional, Generator, Dict, Union, List, Self
from src.helpers import load_json_file
from src.shared.command_tools import BaseCommand, ConnectionOption, OutputDirOption, add_handlers
from src.shared.handlers import Event, EventLevel
from src.shared.file_tools import BufferedWriter, FileWriter
from returns.result import Result, safe
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
    
class HAL_API:

    BASE_URL : str = "https://api.archives-ouvertes.fr/search"

    def __init__(self, config : BaseAPIConfig) -> None:
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

        if normalized_doc["title"] is not None:
            normalized_doc["title"] = normalized_doc["title"].strip('.').strip('[]')

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
    
    def fetch_page(self) -> Generator[PageResult, None, None]:

        num_pages = self.num_pages().unwrap()

        for page_id in range(num_pages):
                
            yield self.make_request(page_id) \
                .map(httpx.Response.json) \
                .map(lambda x :  x['response']['docs']) \
                .map(lambda x : list(map(self.normalize_document, x)))

def _item_from_list_or_dict(d : dict | list[dict]) -> dict:

    if isinstance(d, list):

        if len(d) > 0:
            return d[0]
        
        return None
    
    return d

def _list_from_list_or_dict(d : dict | list[dict]) -> list[dict]:
    return d if isinstance(d, list) else [d]

class PUBMED_API:

    ESEARCH_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?term={term}&format={format}&retmax={retmax}&retstart={retstart}'
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={ids}"

    def __init__(self, config : BaseAPIConfig) -> None:
        self.config = config
        self.__num_pages : Optional[Result[int, Union[httpx.TransportError,httpx.HTTPStatusError]]] = None
        
    @safe(exceptions=(httpx.TransportError,httpx.HTTPStatusError))
    def make_request(self, 
        url : str,
        max_retries : int = 3,
        **kwargs
    ) -> httpx.Response:
        
        for attempt in range(max_retries):
            
            try:

                response = httpx.get(url=url,**kwargs)

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
    
    def etree_to_dict(self, t : ET.Element):
    
        d = {t.tag: {} if t.attrib else None}

        children = list(t)

        if children:

            dd = defaultdict(list)

            for dc in map(self.etree_to_dict, children):
                for k, v in dc.items():
                    dd[k].append(v)

            d = {t.tag: {k:v[0] if len(v) == 1 else v for k, v in dd.items()}}

        if t.attrib:
            d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
            
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[t.tag]['#text'] = text
            else:
                d[t.tag] = text

        return d
    
    def _get_date(self, d : dict) -> Optional[str]:

        alt_date1 = d['MedlineCitation'].get('DateCompleted', None)
        alt_date1 = f"{alt_date1['Year']}-{alt_date1['Month']}-{alt_date1['Day']}T00:00:00Z" if alt_date1 is not None else None

        alt_date2 = d['MedlineCitation'].get('DateRevised', None)
        alt_date2 = f"{alt_date2['Year']}-{alt_date2['Month']}-{alt_date2['Day']}T00:00:00Z" if alt_date2 is not None else None

        article = d['MedlineCitation']['Article']

        date = article.get('ArticleDate')
        date = f"{date['Year']}-{date['Month']}-{date['Day']}T00:00:00Z" if date is not None else None

        if date is None:
            date = alt_date1
        
        if date is None:
            date = alt_date2

        return date
    
    def _get_title(self, d : dict) -> Optional[str]:

        article = d['MedlineCitation']['Article']
        title = _item_from_list_or_dict(article.get('ArticleTitle', []))

        if title is not None:

            title = title.strip('[]')

            if title.lower() == 'not available':
                title = None

        return title
    
    def _get_abstract(self, d : dict) -> Optional[str]:

        article = d['MedlineCitation']['Article']
        abstract = article.get('Abstract', { 'AbstractText' : None })['AbstractText']

        if isinstance(abstract, list):

            abstract_parts = []

            for item in abstract:
                
                if isinstance(item, str):
                    abstract_parts.append(item)

                elif isinstance(item, dict):

                    if '#text' in item:
                        abstract_parts.append(item['#text'])

            abstract = " ".join(abstract_parts)

        if isinstance(abstract, dict):
            abstract = abstract.get('#text', None)

        return abstract
    
    def _get_authors(self, d : dict) -> List[str]:

        article = d['MedlineCitation']['Article']
        authors_info = article.get('AuthorList', { 'Author' : [] })
        authors = [x['LastName'] + '. ' + x['ForeName'] for x in authors_info['Author'] if ('LastName' in x) and ('ForeName' in x)]

        return authors
    
    def _get_teams(self, d : dict) -> List[str]:

        teams = set()

        article = d['MedlineCitation']['Article']
        authors_info = article.get('AuthorList', { 'Author' : [] })

        for author in authors_info['Author']:
            if isinstance(author, dict):
                for aff in _list_from_list_or_dict(author.get('AffiliationInfo', [])):
                    teams.update([team.strip().lower() for team in aff['Affiliation'].split(',')])

        teams = [team for team in teams if team in list(map(str.lower, self.config.teams))]

        return teams
    
    def _get_keywords(self, d : dict) -> List[str]:

        keywords = _list_from_list_or_dict(d['MedlineCitation'].get('KeywordList', { 'Keyword' : [] })['Keyword'])
        keywords = [k['#text'] for k in keywords]

        return keywords


    def normalize_document(self, d : dict) -> Optional[Dict]:

        record = {
            "id" :  d['MedlineCitation']['PMID']['#text'],
            "title" : self._get_title(d),
            "abstract" : self._get_abstract(d),
            "authors" : self._get_authors(d),
            "teams" : self._get_teams(d),
            "date" : self._get_date(d),
            "keywords" : self._get_keywords(d),
            "venues" : []
        }

        dt = None

        if record['date']:

            dt = datetime.fromisoformat(record['date'].replace("Z", "+00:00"))

        return record

        if dt is None:
            return None

        if self.config.from_ != '*':

            from_ = datetime.fromisoformat(self.config.from_.replace("Z", "+00:00"))

            if dt < from_:
                return None
            
        if self.config.to != '*':

            to = datetime.fromisoformat(self.config.to.replace("Z", "+00:00"))

            if dt > to:
                return None

        if any([
            record['abstract'] is None,
            record['date'] is None,
            len(record['teams']) == 0,
        ]):
            return None

        return record

    def _num_pages(self, team : str) -> Result[int, httpx.TransportError | httpx.HTTPStatusError]:

        response = self.make_request(
            PUBMED_API.ESEARCH_URL.format(term=team,format="json",retmax=1,retstart=0),
            max_retries=3
        )

        content = response.map(httpx.Response.json)

        count = content \
            .map(lambda x : x['esearchresult']['count']) \
            .map(int) \
            .map(lambda x : math.ceil(x / self.config.page_size))        
        
        return count


    def fetch_page(self) -> Generator[PageResult, None, None]:
                
        term = ' OR '.join([f'"{lab}"[Affiliation]' for lab in self.config.teams])

        ids_list = []
                
        ### FETCH the list of ids
        retstart = 0
        K = 4
        num_fetched = 0

        while num_fetched < self.num_pages().unwrap():

            response = self.make_request(
                PUBMED_API.ESEARCH_URL.format(term=term,format="json",retmax=K * self.config.page_size,retstart=retstart),
                max_retries=3
            )

            try:
                response = response.unwrap()
            except Exception as e:
                retstart += K * self.config.page_size
                num_fetched += K
                continue
            
            try:
                content = response.json()
            except Exception as e:
                retstart += K * self.config.page_size
                num_fetched += K
                continue
            
            ids_list.extend(content['esearchresult']['idlist'])
            retstart += K * self.config.page_size
            num_fetched += K
            time.sleep(0.05)

        ids_list = set(ids_list)

        print(f"Found {len(ids_list)} articles.")

        ### Request
        for ids in batched(ids_list, self.config.page_size):

            response = self.make_request(
                PUBMED_API.EFETCH_URL.format(ids=','.join(ids)),
                max_retries=3
            )

            try:
                response = response.unwrap()
            except Exception as e:
                return Result.from_failure(e)

            ### XML to JSON
            content = response.text
            
            for tag in ['i', 'sup', 'b', 'div', 'p', 'sub']:
                content = content.replace(f'<{tag}>', ' ').replace(f'</{tag}>', ' ')

            content = ET.ElementTree(ET.fromstring(content))
            content = self.etree_to_dict(content.getroot())

            ### Process Response
            articles = _list_from_list_or_dict(content['PubmedArticleSet']['PubmedArticle'])
            articles = list(filter(None, map(self.normalize_document, articles)))

            yield Result.from_value(articles)
            time.sleep(0.25)
    
    def num_pages(self) -> Result[int, Union[httpx.TransportError, httpx.HTTPStatusError]]:

        compute = self.__num_pages is None

        if not compute:

            try:
                self.__num_pages.unwrap()
            except:
                compute = True

        if compute:

            term = ' OR '.join([f"\"{lab}\"[Affiliation]" for lab in self.config.teams])

            response = self.make_request(
                PUBMED_API.ESEARCH_URL.format(term=term,format="json",retmax=1,retstart=0),
                max_retries=3
            )

            content = response.map(httpx.Response.json)
                            
            self.__num_pages = content \
                .map(lambda x : x['esearchresult']['count']) \
                .map(int) \
                .map(lambda x : math.ceil(x / self.config.page_size)) 

        return self.__num_pages
    
class ApiList:

    def __init__(self, apis : List[HAL_API | PUBMED_API]):
        self.apis = apis
        self.__num_pages = None

    def num_pages(self) -> Result[int, Union[httpx.TransportError, httpx.HTTPStatusError]]:

        compute = self.__num_pages is None

        if not compute:

            try:
                self.__num_pages.unwrap()
            except:
                compute = True

        if compute:

            num_pages = 0
            
            for api in self.apis:

                result = api.num_pages()
                
                try:
                    num_pages += result.unwrap()
                except:
                    self.__num_pages = result

            self.__num_pages = Result.from_value(num_pages)

        return self.__num_pages
    
    def fetch_page(self) -> Generator[PageResult, None, None]:
        
        for api in self.apis:

            for page in api.fetch_page():

                yield page.map(lambda x : [
                    {**item, 'source' : api.__class__.__name__} 
                    for item in x
                ])
        
    
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
            
    def _fetch_all(self) -> Generator[Event, None, None]:

        for page_id, page in enumerate(self.api.fetch_page()):

            page_id, num_pages = self._current_page, self._num_pages
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

                    yield self._make_event(progress, EventLevel.ERROR,message=f"Failed to fetch page {page_id} due to the following error : {payload}.")

                else:

                    yield self._make_event(progress, EventLevel.ERROR,message=f"Failed to fetch page {page_id} due to the following error : {error}.")

        self.writer.flush()

    def __iter__(self) -> Generator[Event, None, None]:

        ### Get the number of pages
        yield self._handle_num_pages_response()
        
        ### Fetch Pages
        yield from self._fetch_all()

        ### Final Event
        yield self._make_event(progress=1.0,level=EventLevel.INFO,message=f"Download of {self._num_pages} has been completed.")

    @classmethod
    def from_config(cls, config : Config) -> Self:

        hal_api = HAL_API(config.hal_conf.get_config('hal'))
        pubmedia_api = PUBMED_API(config.hal_conf.get_config('pubmedia'))

        downloader = DataDownloader(
            api=ApiList([pubmedia_api,hal_api]),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.hal_conf.page_size)
        )

        return downloader