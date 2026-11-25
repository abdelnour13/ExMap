import os
import shutil
from definitions import FEATURES_DIR
from src.commands.feature_extraction.main import FeatureExtractionConfig, FilterConfig
from api.schemas.models import CreateFeaturesExtractionJobBody, FeatureExtractionJobDetails, FeatureExtractionJobConfig
from returns.result import Result
from typing import Literal, Dict, List, Optional
from pydantic import BaseModel, Field
from src.helpers.utils import load_json_file, save_json_file
from api.services.scheduler import Scheduler
from api.services.knowledge_graph_construction import KnowledgeGraphConstructionService

FAILURE_KIND = Literal['EXISTS', 'KG_NOT_EXISTS']

class DeleteKGResponse(BaseModel):
    deleted : bool
    details : Optional[FeatureExtractionJobDetails]

class CreateFailure(BaseModel):
    kind : FAILURE_KIND
    details : Dict = Field(default_factory=dict)

class FeaturesExtractionService:

    def __init__(self, 
        scheduler : Scheduler,
        kg_service : KnowledgeGraphConstructionService
    ) -> None:
        self.scheduler = scheduler
        self.kg_service = kg_service

    def delete(self, id : str) -> DeleteKGResponse:
        
        details = self.get_by_id(id=id)

        if details is None:
            return DeleteKGResponse(deleted=False, details=None)

        fullpath = os.path.join(FEATURES_DIR, id)
        shutil.rmtree(fullpath)
                
        return DeleteKGResponse(deleted=True, details=details)

    def create(self, data : CreateFeaturesExtractionJobBody) -> Result[FeatureExtractionJobDetails, CreateFailure]:

        details = self.get_by_id(data.id)

        if details is not None:
            
            return Result.from_failure(
                CreateFailure(
                    kind='EXISTS',
                    details={ "reason" : f"Knowledge Graph with id : '{data.id}' already exists." }
                )
            )

        kg = self.kg_service.get_by_id(data.config.kg_id)

        if kg is None:

            return Result.from_failure(
                CreateFailure(
                    kind='KG_NOT_EXISTS',
                    details={ "reason" : f"Knowledge graph with id : '{data.config.kg_id}' doesn't exists." }
                )
            )

        fullpath = os.path.join(FEATURES_DIR, data.id)
        configpath = os.path.join(fullpath, "config.json")

        config = FeatureExtractionConfig(
            kg_id=data.config.kg_id,
            model_name=data.config.embedder,
            batch_size=data.config.batch_size,
            device=-1,
            filter=FilterConfig(
                require_subdomain=True,
                min_entities_per_article=1,
                min_articles_per_entity=1,
                largest_connected_component=True
            )
        )

        os.mkdir(path=fullpath)
        save_json_file(configpath, data=config.model_dump(exclude_computed_fields=True))

        return Result.from_value(self.get_by_id_assert(id=data.id))

    def get_by_id_assert(self, id : str) -> FeatureExtractionJobDetails:
        kg_details = self.get_by_id(id=id)
        assert kg_details is not None
        return kg_details 

    def get_by_id(self, id : str) -> Optional[FeatureExtractionJobDetails]:

        return { 
            item.name : item
            for item in self.get_all() 
        }.get(id)

    def get_all(self) -> List[FeatureExtractionJobDetails]:

        results = []
        jobs = self.scheduler.get_all_jobs(type="FEATURE_EXTRACTION")
            
        for filename in os.listdir(FEATURES_DIR):
    
            fullpath = os.path.join(FEATURES_DIR, filename)
            configpath = os.path.join(fullpath, "config.json")
    
            if os.path.exists(configpath):

                inter_config = FeatureExtractionConfig(**load_json_file(configpath))

                config = FeatureExtractionJobConfig(
                    kg_id=inter_config.kg_id,
                    embedder=inter_config.model_name,
                    batch_size=inter_config.batch_size
                )

            else:
                continue
    
            results.append(FeatureExtractionJobDetails(
                name=filename,
                config=config,
                jobs=[job.info for job in jobs if job.config_id == filename],
                exists=os.path.exists(os.path.join(fullpath, "raw.json"))
            ))
    
        return results 