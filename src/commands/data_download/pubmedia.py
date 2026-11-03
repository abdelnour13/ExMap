import httpx
import math
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Union, Optional
from returns.result import Result
from returns.pipeline import is_successful
from src.commands.data_download.api import Api, PageResult
from collections import defaultdict


def _item_from_list_or_dict(d : dict | list[dict]) -> dict | None:

    if isinstance(d, list):
        return d[0] if len(d) > 0 else None

    return d


def _list_from_list_or_dict(d : dict | list[dict]) -> list[dict]:
    return d if isinstance(d, list) else [d]


class PUBMED_API(Api):

    ESEARCH_URL : str = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={term}&format={format}&retmax={retmax}&retstart={retstart}'
    EFETCH_URL : str = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={ids}'

    FORMAT : str = "json"
    ID_BATCH_FACTOR : int = 4          # ids fetched per esearch call = ID_BATCH_FACTOR * page_size
    ESEARCH_DELAY : float = 0.05
    EFETCH_DELAY : float = 0.25
    STRIPPED_TAGS : List[str] = ['i', 'sup', 'b', 'div', 'p', 'sub']

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
        self._ids : Optional[List[str]] = None

    ### ---------- Search term ----------

    @staticmethod
    def _pubmed_date(date : str, default : str) -> str:
        # "2020-01-31T00:00:00Z" -> "2020/01/31"
        return default if date == '*' else date[:10].replace('-', '/')

    def _term(self) -> str:

        teams = ' OR '.join([f'"{team}"[Affiliation]' for team in self.teams])

        if self.from_ == '*' and self.to == '*':
            return teams

        from_ = PUBMED_API._pubmed_date(self.from_, '1000/01/01')
        to = PUBMED_API._pubmed_date(self.to, '3000/12/31')

        return f'({teams}) AND ("{from_}"[dp] : "{to}"[dp])'

    ### ---------- Ids ----------

    def fetch_ids(self) -> Result[List[str], Union[httpx.TransportError, httpx.HTTPStatusError]]:

        if self._ids is not None:
            return Result.from_value(self._ids)

        term = self._term()
        batch_size = PUBMED_API.ID_BATCH_FACTOR * self.page_size

        ids : List[str] = []
        retstart = 0
        count = None

        while count is None or retstart < count:

            response = self.make_request(url=PUBMED_API.ESEARCH_URL.format(
                term=term,
                format=PUBMED_API.FORMAT,
                retmax=batch_size,
                retstart=retstart,
            ))

            if not is_successful(response):
                return response

            result = response.unwrap().json()['esearchresult']
            count = int(result['count'])

            if len(result['idlist']) == 0:
                break

            ids.extend(result['idlist'])
            retstart += batch_size
            time.sleep(PUBMED_API.ESEARCH_DELAY)

        # deduplicate while keeping a stable order, so page_id -> ids is deterministic
        self._ids = list(dict.fromkeys(ids))

        return Result.from_value(self._ids)

    def fetch_num_pages(self) -> Result[int, Union[httpx.TransportError, httpx.HTTPStatusError]]:

        return self.fetch_ids() \
            .map(lambda ids : math.ceil(len(ids) / self.page_size))

    ### ---------- Pages ----------

    def url(self, page_id : int) -> str:

        start = page_id * self.page_size
        ids = self._ids[start : start + self.page_size]

        return PUBMED_API.EFETCH_URL.format(ids=','.join(ids))

    def fetch_page(self, page_id : int) -> PageResult:

        ids = self.fetch_ids()

        if not is_successful(ids):
            return ids

        page = super().fetch_page(page_id)
        time.sleep(PUBMED_API.EFETCH_DELAY)

        return page

    def get_data(self, response : httpx.Response) -> List[Dict]:

        content = response.text

        for tag in PUBMED_API.STRIPPED_TAGS:
            content = content.replace(f'<{tag}>', ' ').replace(f'</{tag}>', ' ')

        root = ET.fromstring(content)
        content = PUBMED_API.etree_to_dict(root)

        return _list_from_list_or_dict(content['PubmedArticleSet']['PubmedArticle'])

    ### ---------- Normalization ----------

    @staticmethod
    def etree_to_dict(t : ET.Element) -> Dict:

        d = {t.tag: {} if t.attrib else None}

        children = list(t)

        if children:

            dd = defaultdict(list)

            for dc in map(PUBMED_API.etree_to_dict, children):
                for k, v in dc.items():
                    dd[k].append(v)

            d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}

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

    @staticmethod
    def _format_date(date : Optional[Dict]) -> Optional[str]:

        if date is None:
            return None

        return f"{date['Year']}-{date['Month']}-{date['Day']}T00:00:00Z"

    def _get_date(self, d : Dict) -> Optional[str]:

        citation = d['MedlineCitation']

        return PUBMED_API._format_date(citation['Article'].get('ArticleDate')) \
            or PUBMED_API._format_date(citation.get('DateCompleted')) \
            or PUBMED_API._format_date(citation.get('DateRevised'))

    def _get_title(self, d : Dict) -> Optional[str]:

        article = d['MedlineCitation']['Article']
        title = _item_from_list_or_dict(article.get('ArticleTitle', []))

        if title is not None:

            title = title.strip('[]')

            if title.lower() == 'not available':
                title = None

        return title

    def _get_abstract(self, d : Dict) -> Optional[str]:

        article = d['MedlineCitation']['Article']
        abstract = article.get('Abstract', {'AbstractText' : None})['AbstractText']

        if isinstance(abstract, list):

            abstract_parts = []

            for item in abstract:

                if isinstance(item, str):
                    abstract_parts.append(item)

                elif isinstance(item, dict) and '#text' in item:
                    abstract_parts.append(item['#text'])

            abstract = " ".join(abstract_parts)

        if isinstance(abstract, dict):
            abstract = abstract.get('#text', None)

        return abstract

    def _get_authors(self, d : Dict) -> List[str]:

        article = d['MedlineCitation']['Article']
        authors_info = article.get('AuthorList', {'Author' : []})

        return [
            x['LastName'] + '. ' + x['ForeName']
            for x in _list_from_list_or_dict(authors_info['Author'])
            if ('LastName' in x) and ('ForeName' in x)
        ]

    def _get_teams(self, d : Dict) -> List[str]:

        wanted = set(map(str.lower, self.teams))
        teams = set()

        article = d['MedlineCitation']['Article']
        authors_info = article.get('AuthorList', {'Author' : []})

        for author in _list_from_list_or_dict(authors_info['Author']):
            if isinstance(author, dict):
                for aff in _list_from_list_or_dict(author.get('AffiliationInfo', [])):
                    teams.update(team.strip().lower() for team in aff['Affiliation'].split(','))

        return [team for team in teams if team in wanted]

    def _get_keywords(self, d : Dict) -> List[str]:

        keywords = d['MedlineCitation'].get('KeywordList', {'Keyword' : []})['Keyword']

        return [k['#text'] for k in _list_from_list_or_dict(keywords)]

    def normalize_document(self, document : Dict) -> Dict:

        return {
            "id" : document['MedlineCitation']['PMID']['#text'],
            "title" : self._get_title(document),
            "abstract" : self._get_abstract(document),
            "authors" : self._get_authors(document),
            "teams" : self._get_teams(document),
            "date" : self._get_date(document),
            "keywords" : self._get_keywords(document),
            "venues" : [],
        }

if __name__ == '__main__':

    from pprint import pprint

    api = PUBMED_API(
        teams=["LS2N"],
        from_="2020-01-01T00:00:00Z",
        to="*",
        page_size=32
    )

    print(api.num_pages())
    pprint(api.fetch_page(0).unwrap()[0])