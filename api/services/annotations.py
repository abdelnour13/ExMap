import os
import shutil
from api.models import AnnotationDetails, CreateAnnotationBody
from api.services.scheduler import Scheduler
from api.services.corpuses import CorpusesService
from api.services.domains_list import DomainsListService
from typing import List, Optional, Dict, Literal
from definitions import ANNOTATION_DIR, CORPUS_DIR
from src.helpers import load_json_file, save_json_file
from pydantic import BaseModel
from returns.result import Result

FAILURE_TYPE = Literal['ANNOTATION_EXISTS', 'CORPUS_NOT_EXISTS', 'DOMAINS_LIST_NOT_EXISTS']

class DeleteAnnotationResponse(BaseModel):
    deleted : bool
    details : AnnotationDetails

class AnnotationFailure(BaseModel):
    kind : FAILURE_TYPE
    details : Dict

class AnnotationService:

    def __init__(self, 
        scheduler : Scheduler, 
        courpses_service : CorpusesService,
        domains_list_service : DomainsListService
    ):
        self.scheduler = scheduler
        self.courpses_service = courpses_service
        self.domains_list_service = domains_list_service

    def delete(self, id : str) -> DeleteAnnotationResponse:
    
        details = self.get_by_id(id=id)

        if details is None:
            return DeleteAnnotationResponse(deleted=False, details=None)

        fullpath = os.path.join(CORPUS_DIR, id)
        shutil.rmtree(fullpath)
                
        return DeleteAnnotationResponse(deleted=True, details=details)

    def create(self, data : CreateAnnotationBody) -> Result[AnnotationDetails, AnnotationFailure]:
    
        details = self.get_by_id(id=data.id)

        if details is not None:

            return Result.from_failure(
                AnnotationFailure(
                    kind='ANNOTATION_EXISTS',
                    details={ "reason" : f"Annotation with id '{details.name}' already exists." }
                )
            )

        if self.courpses_service.get_by_id(data.config.corpus_id) is None:

            return Result.from_failure(
                AnnotationFailure(
                    kind='CORPUS_NOT_EXISTS',
                    details={ "reason" : f"Corpus with id '{data.config.corpus_id}' doesn't exists." }
                )
            )

        if self.domains_list_service.get_by_id(data.config.domain) is None:
        
            return Result.from_failure(
                AnnotationFailure(
                    kind='DOMAINS_LIST_NOT_EXISTS',
                    details={ "reason" : f"Domains List with id '{data.config.domain}' doesn't exists." }
                )
            )
        

        fullpath = os.path.join(ANNOTATION_DIR, data.id)
        configpath = os.path.join(fullpath, "config.json")

        os.mkdir(path=fullpath)
        save_json_file(configpath, data=data.config.model_dump())

        return Result.from_value(self.get_by_id_assert(id=data.id))

    def get_by_id_assert(self, id : str) -> AnnotationDetails:
        annotation_detail = self.get_by_id(id=id)
        assert annotation_detail is not None
        return annotation_detail 

    def get_by_id(self, id : str) -> Optional[AnnotationDetails]:

        return { 
            item.name : item
            for item in self.get_all() 
        }.get(id)

    def get_all(self) -> List[AnnotationDetails]:

        results = []
        jobs = self.scheduler.get_all_jobs(type="ANNOTATION")
            
        for filename in os.listdir(ANNOTATION_DIR):
    
            fullpath = os.path.join(ANNOTATION_DIR, filename)
            configpath = os.path.join(fullpath, "config.json")
            datapath = os.path.join(fullpath, "data.jsonl")

            exists = os.path.exists(datapath)

            print(filename)
            print(configpath)
            print(os.path.exists(configpath))
    
            if os.path.exists(configpath):
                config = load_json_file(configpath)
            else:
                continue
    
            results.append(AnnotationDetails(
                name=filename,
                exists=exists,
                config=config,
                jobs=[job.info for job in jobs if job.config_id == filename]
            ))
    
        return results