import httpx
import math
from typing import List, Optional, Dict, Union
from returns.result import Result
from src.commands.data_download.api import Api

class HAL_API(Api):

    BASE_URL : str = "https://api.archives-ouvertes.fr/search"

    FORMAT : str = "json"
    ATTRIBUTES : List[str] = [
        "docid",
        "title_s",
        "keyword_s",
        "abstract_s",
        "authFullName_s",
        "producedDate_tdate",
        "domainAllCode_s",
        "collCode_s",
    ]
    LANGUAGES : List[str] = ["en"]
    SORT : str = "producedDate_tdate desc"
    MAX_PAGES : Optional[int] = None

    def __init__(self,
        teams : List[str],
        from_ : str = "*",
        to : str = "*",
        page_size : int = 32,
    ) -> None:
        super().__init__()
        self.teams = teams
        self.from_ = from_
        self.to = to
        self.page_size = page_size

    def url(self, page_id : int = 0) -> str:

        return ''.join([
            HAL_API.BASE_URL,
            f'?wt={HAL_API.FORMAT}',
            f'&fl={",".join(HAL_API.ATTRIBUTES)}',
            f'&fq=collCode_s:({" OR ".join(self.teams)})',
            f'&fq=producedDate_tdate:[{self.from_} TO {self.to}]',
            f'&fq=language_s:({" OR ".join(HAL_API.LANGUAGES)})',
            f'&sort={HAL_API.SORT}',
            f'&rows={self.page_size}',
            f'&start={page_id * self.page_size}',
            f'&fq=abstract_s:["" TO *]',
        ])

    def get_data(self, response : httpx.Response) -> List[Dict]:
        return response.json()['response']['docs']

    def normalize_document(self, document : Dict) -> Dict:

        normalized_doc = dict()

        normalized_doc["id"] = document["docid"]
        normalized_doc["title"] = document.get("title_s", [None])[0]
        normalized_doc["abstract"] = document.get("abstract_s", [None])[0]
        normalized_doc["keywords"] = document.get("keyword_s", [])
        normalized_doc["authors"] = document.get("authFullName_s", [])
        normalized_doc["venues"] = document.get("domainAllCode_s", [])

        normalized_doc["teams"] = [
            team for team in document.get("collCode_s", [])
            if team in self.teams
        ]

        if normalized_doc["title"] is not None:
            normalized_doc["title"] = normalized_doc["title"].strip('.').strip('[]')

        normalized_doc["date"] = document["producedDate_tdate"]

        return normalized_doc

    def fetch_num_pages(self) -> Result[int, Union[httpx.TransportError, httpx.HTTPStatusError]]:

        return self.make_request(url=self.url(page_id=0)) \
            .map(httpx.Response.json) \
            .map(lambda x : x['response']['numFound']) \
            .map(lambda x : math.ceil(x / self.page_size)) \
            .map(lambda x : x if HAL_API.MAX_PAGES is None else min(x, HAL_API.MAX_PAGES))

if __name__ == '__main__':

    api = HAL_API(
        teams=["LS2N-DUKE"],
        from_="2020-01-01T00:00:00Z",
        to="*",
        page_size=32
    )

    print(api.num_pages())
    print(api.fetch_page(0))