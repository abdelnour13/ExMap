import os
import shutil
import json
from api.models import CreateKnowledgeGraphBody, KnowledgeGraphDetails, PipelineConfig
from api.services import Scheduler, AnnotationService
from src.helpers import load_json_file, save_json_file
from typing import List, Optional, Literal, Dict
from definitions import KGS_DIR
from pydantic import BaseModel
from returns.result import Result

template = """{{
    "corpus_id" : "{corpus_id}",
    "glob" : {{
        "domain" : "industry_of_future",
        "batch_size" : 1,
        "buffer_size" : 4,
        "resume" : true
    }},
    "stages" : {{
        "corpus_cleaning" : {{
            "stage" : "corpus_cleaning",
            "batch_size" : 32,
            "buffer_size" : 32
        }},
        "coreference_resolution" : {{
            "stage" : "coreference_resolution",
            "model" : "spanbert-large",
            "device" : -1
        }},
        "keywords_extraction" : {{
            "stage" : "keywords_extraction",
            "batch_size" : 1,
            "buffer_size" : 4
        }},
        "sentence_segmentation" : {{
            "stage" : "sentence_segmentation",
            "use_gpu" : false,
            "batch_size" : 128,
            "buffer_size" : 128
        }},
        "acronyms_extraction" : {{
            "stage" : "acronyms_extraction",
            "qa_model" : "timpal0l/mdeberta-v3-base-squad2",
            "threshold" : 0.5
        }},
        "triples_extraction": {{
            "stage" : "triples_extraction",
            "batch_size" : 16,
            "buffer_size" : 16,
            "type" : "two-step-llm/{llm_name}"
        }},
        "entities_normalization" : {{
            "stage" : "entities_normalization",
            "batch_size" : 128,
            "buffer_size" : 128
        }},
        "rule_based_predicate_cleaning" : {{
            "stage" : "rule_based_predicate_cleaning",
            "batch_size" : 128,
            "buffer_size" : 128
        }},
        "llm_based_predicate_cleaning" : {{
            "llm_name" : "{llm_name}",
            "stage" : "llm_based_predicate_cleaning",
            "batch_size" : 128,
            "buffer_size" : 128
        }},
        "llm_based_entities_cleaning" : {{
            "llm_name" : "{llm_name}",
            "stage" : "llm_based_entities_cleaning",
            "batch_size" : 128,
            "buffer_size" : 128,
            "plm_device" : -1
        }}
    }}
}}"""

KIND = Literal['KG_EXISTS', 'ANNOTATION_NOT_EXISTS']

class DeleteKGResponse(BaseModel):
    deleted : bool
    details : Optional[KnowledgeGraphDetails]

class CreateKGFailure(BaseModel):
    kind : KIND
    details : Dict

class KnowledgeGraphConstructionService:

    def __init__(self, scheduler : Scheduler, annotation_service : AnnotationService):
        self.scheduler = scheduler
        self.annotation_service = annotation_service

    def delete(self, id : str) -> DeleteKGResponse:
    
        details = self.get_by_id(id=id)

        if details is None:
            return DeleteKGResponse(deleted=False, details=None)

        fullpath = os.path.join(KGS_DIR, id)
        shutil.rmtree(fullpath)
                
        return DeleteKGResponse(deleted=True, details=details)
        
    def create(self, data : CreateKnowledgeGraphBody) -> Result[KnowledgeGraphDetails, CreateKGFailure]:

        details = self.get_by_id(id=data.id)

        if details is not None:

            return Result.from_failure(
                CreateKGFailure(
                    kind='KG_EXISTS',
                    details={ "reason" : f"Knowledge Graph with id : '{data.id}' already exists." }
                )
            )

        if self.annotation_service.get_by_id(data.config.corpus_id) is None:

            return Result.from_failure(
                CreateKGFailure(
                    kind='ANNOTATION_NOT_EXISTS',
                    details={ "reason" : f"Annotation with id : '{data.config.corpus_id}' doesn't exists." }
                )
            )

        fullpath = os.path.join(KGS_DIR, data.id)
        configpath = os.path.join(fullpath, "config.json")

        config = template.format(
            corpus_id=f"{data.config.corpus_id}/data.jsonl",
            llm_name=data.config.llm_name
        )

        os.mkdir(path=fullpath)
        save_json_file(configpath, data=json.loads(config))

        return Result.from_value(self.get_by_id_assert(id=data.id))

    def get_by_id_assert(self, id : str) -> KnowledgeGraphDetails:
        kg_details = self.get_by_id(id=id)
        assert kg_details is not None
        return kg_details 

    def get_by_id(self, id : str) -> Optional[KnowledgeGraphDetails]:

        return { 
            item.name : item
            for item in self.get_all() 
        }.get(id)

    def get_all(self) -> List[KnowledgeGraphDetails]:

        results = []
        jobs = self.scheduler.get_all_jobs(type="KG_CONSTRUCTION")
            
        for filename in os.listdir(KGS_DIR):
    
            fullpath = os.path.join(KGS_DIR, filename)
            configpath = os.path.join(fullpath, "config.json")
    
            if os.path.exists(configpath):

                config = load_json_file(configpath)

                if 'triples_extraction' not in config['stages']:
                    continue

                config = PipelineConfig(
                    corpus_id=f"{config['corpus_id']}".split('/')[0],
                    llm_name=config['stages']['triples_extraction']['type'].split('/')[-1]
                )

            else:
                continue
    
            results.append(KnowledgeGraphDetails(
                name=filename,
                config=config,
                jobs=[job.info for job in jobs if job.config_id == filename]
            ))
    
        return results        