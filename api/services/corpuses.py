import os
import shutil
from definitions import CORPUS_DIR
from typing import List, Optional
from api.services.scheduler import Scheduler
from api.models import CorpusDetails, CreateCorpusBody
from src.helpers import load_json_file, save_json_file
from pydantic import BaseModel

class CreateCorpusResponse(BaseModel):
    created : bool
    details : CorpusDetails

class DeleteCorpusResponse(BaseModel):
    deleted : bool
    details : Optional[CorpusDetails]

class CorpusesService:

    def __init__(self, scheduler : Scheduler):
        self.scheduler = scheduler

    def delete_corpus(self, id : str) -> DeleteCorpusResponse:

        details = self.get_by_id(id=id)

        if details is None:
            return DeleteCorpusResponse(deleted=False, details=None)

        fullpath = os.path.join(CORPUS_DIR, id)
        shutil.rmtree(fullpath)
                
        return DeleteCorpusResponse(deleted=True, details=details)
        
    def create_corpus(self, data : CreateCorpusBody) -> CreateCorpusResponse:

        details = self.get_by_id(id=data.id)

        if details is not None:
            return CreateCorpusResponse(created=False, details=details)

        fullpath = os.path.join(CORPUS_DIR, data.id)
        configpath = os.path.join(fullpath, "config.json")

        os.mkdir(path=fullpath)
        save_json_file(configpath, data=data.config.model_dump())

        return CreateCorpusResponse(created=True, details=self.get_by_id_assert(id=data.id))

    def get_by_id_assert(self, id : str) -> CorpusDetails:
        corpus_details = self.get_by_id(id=id)
        assert corpus_details is not None
        return corpus_details 

    def get_by_id(self, id : str) -> Optional[CorpusDetails]:

        return { 
          item.name : item
          for item in self.get_all() 
        }.get(id)

    def get_all(self) -> List[CorpusDetails]:

        results = []
        jobs = self.scheduler.get_all_jobs(type="DOWNLOAD")
        
        for filename in os.listdir(CORPUS_DIR):

            fullpath = os.path.join(CORPUS_DIR, filename)
            datapath = os.path.join(fullpath, "data.jsonl")
            configpath = os.path.join(fullpath, "config.json")

            size = 0
            num_records = 0
            config = None

            if os.path.exists(configpath):
                config = load_json_file(configpath)
            else:
                continue

            exists = os.path.exists(datapath)

            if exists:

                size = os.path.getsize(datapath)

                with open(datapath, "r") as f:
                    for _ in f.readlines():
                        num_records += 1

            info = CorpusDetails(**{
                "name" : filename,
                "size" : size,
                "exists" : exists,
                "num_records" : num_records,
                "config" : config,
                "jobs" : [job.info for job in jobs if job.config_id == filename]
            })

            results.append(info)

        return results