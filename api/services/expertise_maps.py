import os
import shutil
import json
from definitions import EMS_DIR
from api.models import CreateExpertiseMapBody, ExpertiseMapDetails, ExpertiseMapConfig
from returns.result import Result
from typing import Literal, Dict, List, Optional
from pydantic import BaseModel, Field
from src.helpers.utils import load_json_file, save_json_file
from api.services.scheduler import Scheduler
from api.services.graph_learning import GraphLearningService
from src.commands.build_expertise_map import BEMConfig

FAILURE_KIND = Literal['EXISTS', 'MODEL_NOT_EXISTS']

class DeleteResponse(BaseModel):
    deleted : bool
    details : Optional[ExpertiseMapDetails]

class CreateFailure(BaseModel):
    kind : FAILURE_KIND
    details : Dict = Field(default_factory=dict)


class ExpertiseMapService:

    def __init__(self, 
        scheduler : Scheduler,
        gl_service : GraphLearningService
    ) -> None:
        self.scheduler = scheduler
        self.gl_service = gl_service

    def delete(self, id : str) -> DeleteResponse:
        
        details = self.get_by_id(id=id)

        if details is None:
            return DeleteResponse(deleted=False, details=None)

        fullpath = os.path.join(EMS_DIR, id)
        shutil.rmtree(fullpath)
                
        return DeleteResponse(deleted=True, details=details)

    def create(self, data : CreateExpertiseMapBody) -> Result[ExpertiseMapDetails, CreateFailure]:

        details = self.get_by_id(data.id)

        if details is not None:
            
            return Result.from_failure(
                CreateFailure(
                    kind='EXISTS',
                    details={ "reason" : f"Expertise Map with id : '{data.id}' already exists." }
                )
            )

        features = self.gl_service.get_by_id(data.config.model)

        if features is None:

            return Result.from_failure(
                CreateFailure(
                    kind='KG_NOT_EXISTS',
                    details={ "reason" : f"Model with id : '{data.config.model}' doesn't exists." }
                )
            )

        fullpath = os.path.join(EMS_DIR, data.id)
        configpath = os.path.join(fullpath, "config.json")

        config = BEMConfig(
            expertise_id=data.id,
            model_id=f"{data.config.model}/baseline",
            llm=data.config.llm,
            name=data.config.name,
            description=data.config.description,
            clustering_alg=data.config.clustering_alg,
            clustering_resolution=data.config.resolution
        )

        os.mkdir(path=fullpath)
        save_json_file(configpath, data=config.model_dump(exclude_computed_fields=True))

        return Result.from_value(self.get_by_id_assert(id=data.id))

    def get_by_id_assert(self, id : str) -> ExpertiseMapDetails:
        kg_details = self.get_by_id(id=id)
        assert kg_details is not None
        return kg_details 

    def get_by_id(self, id : str) -> Optional[ExpertiseMapDetails]:

        return { 
            item.name : item
            for item in self.get_all() 
        }.get(id)

    def get_all(self) -> List[ExpertiseMapDetails]:

        results = []
        jobs = self.scheduler.get_all_jobs(type="MAP_CONSTRUCTION")
            
        for filename in os.listdir(EMS_DIR):
    
            fullpath = os.path.join(EMS_DIR, filename)
            configpath = os.path.join(fullpath, "config.json")
    
            if os.path.exists(configpath):

                inner_config = load_json_file(configpath)
                inner_config = BEMConfig(**inner_config)

                config = ExpertiseMapConfig(
                    model=inner_config.model_id.split('/')[0],
                    llm=inner_config.llm,
                    name=inner_config.name,
                    description=inner_config.description,
                    clustering_alg=inner_config.clustering_alg,
                    resolution=inner_config.clustering_resolution
                )

            else:
                continue
    
            results.append(ExpertiseMapDetails(
                name=filename,
                config=config,
                jobs=[job.info for job in jobs if job.config_id == filename]
            ))
    
        return results 