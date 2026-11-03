from src.commands.data_download.api import Api, PageResult
from typing import List, Tuple, Optional

class ApiList:

    def __init__(self, apis : List[Api]) -> None:
        self.apis = apis
        self._num_pages : Optional[int] = None

    def num_pages(self) -> Optional[int]:

        if self._num_pages is not None:
            return self._num_pages

        counts = [api.num_pages() for api in self.apis]

        # only cache once every api has answered, so a failed api is retried next call
        if all(count is not None for count in counts):
            self._num_pages = sum(counts)

        return self._num_pages

    def _locate(self, page_id : int) -> Tuple[Api, int]:
        """Map a global page_id to (api, page_id local to that api)."""

        local_id = page_id

        for api in self.apis:

            count = api.num_pages()

            if local_id < count:
                return api, local_id

            local_id -= count

        raise IndexError(f"page_id {page_id} out of range (num_pages = {self.num_pages()})")

    def fetch_page(self, page_id : int) -> PageResult:

        # if an api's page count is unavailable, return the underlying failure
        for api in self.apis:
            if api.num_pages() is None:
                return api.fetch_num_pages()

        api, local_id = self._locate(page_id)
        source = api.__class__.__name__

        return api.fetch_page(local_id).map(lambda docs : [
            {**doc, 'source' : source}
            for doc in docs
        ])