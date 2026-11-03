from pydantic import BaseModel, Field
from src.commands.data_download.main import APIConfig, API_TYPE
from src.commands.helpers import Domain
from typing import List
from api.wrappers.annotation import CorpusAnnotatorConfig
from api.services.scheduler import JobInfo

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