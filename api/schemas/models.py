from pydantic import BaseModel, Field, BeforeValidator
from src.commands.data_download.main import APIConfig
from src.commands.helpers import Domain
from typing import List, Annotated
from api.wrappers.annotation import CorpusAnnotatorConfig
from api.services.scheduler import JobInfo
from typing import Literal
from src.helpers.torch_utils import get_all_devices

class ExpertiseMapConfig(BaseModel):
    model : str
    llm : str
    name : str
    description : str
    clustering_alg : Literal["leiden", "louvain"] = "leiden"
    resolution : int = 15

class CreateExpertiseMapBody(BaseModel):
    id : str
    config : ExpertiseMapConfig

class ExpertiseMapDetails(BaseModel):
    name : str
    config : ExpertiseMapConfig
    jobs : List[JobInfo]
    exists : bool
    uploaded : bool


def validate_device(device : str) -> str:

    devices = get_all_devices()

    if device not in devices:
        raise ValueError(f"Device {device} is not available. Available devices are : {'|'.join(devices)}.")

    return device

Device = Annotated[str, BeforeValidator(validate_device)]

class GraphLearningJobConfig(BaseModel):
    features_id : str
    batch_size : int = 16
    learning_rate : float = 0.0025
    num_epochs : int = 10
    device : Device = "cpu" # -1 for cpu else gpu index, for example "cuda:0" ...etc / find available devices at "/devices"

class CreateGraphLearningJobBody(BaseModel):
    id : str
    config : GraphLearningJobConfig

class GraphLearningJobDetails(BaseModel):
    name : str
    config : GraphLearningJobConfig
    jobs : List[JobInfo]
    exists : bool

class FeatureExtractionJobConfig(BaseModel):
    kg_id : str
    embedder : str
    batch_size : int

class CreateFeaturesExtractionJobBody(BaseModel):
    id : str
    config : FeatureExtractionJobConfig

class FeatureExtractionJobDetails(BaseModel):
    name : str
    config : FeatureExtractionJobConfig
    jobs : List[JobInfo] = Field(default=list)
    exists : bool

class PipelineConfig(BaseModel):
    corpus_id : str
    llm_name : str

class CreateKnowledgeGraphBody(BaseModel):
    id : str
    config : PipelineConfig

class KnowledgeGraphDetails(BaseModel):
    name : str
    config : PipelineConfig
    jobs : List[JobInfo] = Field(default=list)
    exists : bool

class CreateCorpusBody(BaseModel):
    id : str
    config : APIConfig

class GetDomainsList(BaseModel):
    name : str
    domains : List[Domain]

class CreateDomainsList(BaseModel):
    name : str
    domains : List[Domain]

class CreateAnnotationBody(BaseModel):
    id : str
    config : CorpusAnnotatorConfig

class AnnotationDetails(BaseModel):
    name : str
    exists : bool
    config : CorpusAnnotatorConfig
    jobs : List[JobInfo] = Field(default=list)    # All Associated Jobs
    
class DeleteResponse(BaseModel):
    succeed : bool

class CorpusDetails(BaseModel):
    name : str                                    # Name of the corpus
    size : int                                    # Size in Bytes
    num_records : int                             # Number of records
    exists : bool                                 # whether $DIR/data.jsonl actually exists or not
    config : APIConfig                            # The config used to download the data
    jobs : List[JobInfo] = Field(default=list)    # All jobs with same type and config_id
 
class StartJobResponse(BaseModel):
    job_id : str