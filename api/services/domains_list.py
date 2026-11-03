import os
from definitions import FIELDS_DIR, domain_dir
from typing import List, Optional
from api.models import GetDomainsList, Domain, CreateDomainsList
from src.helpers import load_json_file, save_json_file
from pydantic import BaseModel

class CreateDomainsListResponse(BaseModel):
    created : bool
    details : GetDomainsList

class DomainsListService:

    def __init__(self):
        pass

    def get_all(self) -> List[GetDomainsList]:

        results = []

        for filename in os.listdir(FIELDS_DIR):
            fullpath = domain_dir(filename)
            domains = [Domain(**domain) for domain in load_json_file(fullpath)]
            results.append(GetDomainsList(name=filename, domains=domains))

        return results

    def get_by_id(self, id : str) -> Optional[GetDomainsList]:
    
        return {
            item.name : item
            for item in self.get_all()
        }.get(id)

    def get_by_id_assert(self, id : str) -> GetDomainsList:
        domains_list = self.get_by_id(id=id)
        assert domains_list is not None
        return domains_list

    def create_domains_lists(self, data : CreateDomainsList) -> CreateDomainsListResponse:

        details = self.get_by_id(id=data.name)

        if details is not None:
            return CreateDomainsListResponse(created=False, details=details)

        dirname = os.path.join(FIELDS_DIR, data.name)
        filename = domain_dir(data.name)
        os.mkdir(dirname)

        save_json_file(path=filename, data=[item.model_dump() for item in data.domains])

        return CreateDomainsListResponse(created=True, details=self.get_by_id_assert(id=data.name))