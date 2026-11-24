import os
import shutil
import json
from definitions import EXPERIMENTS_DIR
from api.schemas.models import CreateGraphLearningJobBody, GraphLearningJobConfig, GraphLearningJobDetails
from returns.result import Result
from typing import Literal, Dict, List, Optional
from pydantic import BaseModel, Field
from src.helpers.utils import load_json_file, save_json_file
from api.services.scheduler import Scheduler
from api.services.feature_extraction import FeaturesExtractionService
from api.constants import SUBDOMAINS_LABEL_KEY

FAILURE_KIND = Literal['EXISTS', 'FEATURES_NOT_EXISTS']

class DeleteResponse(BaseModel):
    deleted : bool
    details : Optional[GraphLearningJobDetails]

class CreateFailure(BaseModel):
    kind : FAILURE_KIND
    details : Dict = Field(default_factory=dict)

template = """{{

    "seed" : 42,
    "num_splits" : 4,
    "split" : "k_fold",

    "graph" : "{graph}",
    "labels" : [
        {{
            "name" : "lab",
            "bounds" : [0.0, 1.0],
            "alpha" : 1.0
        }},
        {{
            "name" : "{label}",
            "bounds" : [0.0, 1.0],
            "alpha" : 1.0
        }}
    ],

    "batch_size" : {batch_size},
    "num_articles_neighbors" : [8, 4],
    "domain" : null,
    "full_batch_training": false,

    "model_name" : "ArticleClassifier",
    "model_params" : {{
        "encoder_type" : "gnn",
        "hidden_channels" : [256, 256],
        "out_channels" : 256,
        "variant" : "conv",
        "activation" : "relu",
        "norm" : true,
        "shortcut" : true,
        "shortcut_dim" : null,
        "jk" : true,
        "readout" : "mean"
    }},

    "lr" : {lr},
    "num_epochs" : {num_epochs},
    "accumelation_steps" : 1,
    "weight_decay" : 1e-4,

    "device" : "{device}",

    "log_every" : 1,

    "losses" : {{
        "bec_loss_{label}" : {{
            "weight" : 0.5,
            "params" : {{}}
        }},
        "bec_loss_lab" : {{
            "weight" : 0.5,
            "params" : {{}}
        }}
    }},

    "hparams" : {{
        "baseline" : {{}}
    }}
}}
"""


class GraphLearningService:

    def __init__(self, 
        scheduler : Scheduler,
        features_service : FeaturesExtractionService
    ) -> None:
        self.scheduler = scheduler
        self.features_service = features_service

    def delete(self, id : str) -> DeleteResponse:
        
        details = self.get_by_id(id=id)

        if details is None:
            return DeleteResponse(deleted=False, details=None)

        fullpath = os.path.join(EXPERIMENTS_DIR, id)
        shutil.rmtree(fullpath)
                
        return DeleteResponse(deleted=True, details=details)

    def create(self, data : CreateGraphLearningJobBody) -> Result[GraphLearningJobDetails, CreateFailure]:

        details = self.get_by_id(data.id)

        if details is not None:
            
            return Result.from_failure(
                CreateFailure(
                    kind='EXISTS',
                    details={ "reason" : f"Experiment with id : '{data.id}' already exists." }
                )
            )

        features = self.features_service.get_by_id(data.config.features_id)

        if features is None:

            return Result.from_failure(
                CreateFailure(
                    kind='KG_NOT_EXISTS',
                    details={ "reason" : f"Knowledge graph with id : '{data.config.features_id}' doesn't exists." }
                )
            )

        fullpath = os.path.join(EXPERIMENTS_DIR, data.id)
        configpath = os.path.join(fullpath, "config.json")

        config = json.loads(template.format(
            graph=data.config.features_id,
            label=SUBDOMAINS_LABEL_KEY,
            lr=data.config.learning_rate,
            num_epochs=data.config.num_epochs,
            batch_size=data.config.batch_size,
            device=data.config.device
        ))

        os.mkdir(path=fullpath)
        save_json_file(configpath, data=config)

        return Result.from_value(self.get_by_id_assert(id=data.id))

    def get_by_id_assert(self, id : str) -> GraphLearningJobDetails:
        kg_details = self.get_by_id(id=id)
        assert kg_details is not None
        return kg_details 

    def get_by_id(self, id : str) -> Optional[GraphLearningJobDetails]:

        return { 
            item.name : item
            for item in self.get_all() 
        }.get(id)

    def get_all(self) -> List[GraphLearningJobDetails]:

        results = []
        jobs = self.scheduler.get_all_jobs(type="GRAPH_LEARNING")
            
        for filename in os.listdir(EXPERIMENTS_DIR):
    
            fullpath = os.path.join(EXPERIMENTS_DIR, filename)
            configpath = os.path.join(fullpath, "config.json")
    
            if os.path.exists(configpath):

                inner_config = load_json_file(configpath)
                config = GraphLearningJobConfig(
                    features_id=inner_config['graph'],
                    num_epochs=inner_config['num_epochs'],
                    batch_size=inner_config['batch_size'],
                    learning_rate=inner_config['lr'],
                    device=inner_config['device']
                )

            else:
                continue
    
            results.append(GraphLearningJobDetails(
                name=filename,
                config=config,
                jobs=[job.info for job in jobs if job.config_id == filename],
                exists=os.path.exists(os.path.join(fullpath, "models", "model_baseline_final.pt"))
            ))
    
        return results 