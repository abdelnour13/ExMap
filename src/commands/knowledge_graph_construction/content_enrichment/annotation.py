import torch
from typing import Dict, List, Any
from src.shared.command_tools.vasync import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.methods.zs_classifiers import ZeroShotClassifier, ZeroShotClassifierRegistry, Label
from src.commands.helpers import DomainConfig
from pycli import Field
from instance import cli
from returns.pipeline import is_successful
from pydantic import BaseModel

class ClassifierConfig(BaseModel):
    name : str
    args : Dict[str, Any] = Field(default_factory=dict)

class Config(StageConfig, DomainConfig('domain')):

    models : List[ClassifierConfig] = Field(
        default_factory=lambda : [ClassifierConfig(name="llm:deepseek-chat:binary")],
        description="The model(s) to be used for triple extraction."
    )

    device : str = Field(
        default=('cuda' if torch.cuda.is_available() else 'cpu'),
        description=f"The device to run inference on."
    )

    domain : str = Field(description="The json file that holds the labels and their definitions.")

@cli.register_stage(
    name="corpus_annotation",
    description=(
        "Classify using the provided model(s) each document into one of the domains defined in the json file passed."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class CorpusAnnotatator(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        models : Dict[str, ZeroShotClassifier],
        labels : List[Label]
    ) -> None:
        super().__init__(reader, writer)

        self.models = models
        self.labels = labels

    def encode_document(self, doc: Dict) -> str:

        encoded_doc = [
            f"Title : {doc['title']}",
            f"Abstract : {doc['abstract']}",
            f"Keywords : {' | '.join(doc['keywords'])}",
        ]

        return "\n".join(encoded_doc)

    def transform(self, doc: Dict) -> Dict:
        _, results, _ =  self.transform_chunk([doc])
        return results[0]

    def transform_chunk(self, batch : List[Dict]) -> List[Dict]:

        results = [doc for doc in batch]
        errors = []
        docs = []
        ignored = set()

        for classifier_name, classifier in self.models.items():

            inputs = [
                {
                    "id" : i,
                    "content" : self.encode_document(doc)
                }
                for i,doc in enumerate(batch) if i not in ignored
            ]

            for input_, result, doc in zip(
                inputs, 
                classifier.classify(inputs, self.labels), 
                batch
            ):

                if is_successful(result):
                    result = result.unwrap()
                    docs.append(doc)
                else:
                    results[input_['id']] = None
                    ignored.add(input_['id'])
                    errors.append(result.failure())
                    continue

                results[input_['id']]['pseudo_labels'] = results[input_['id']].get('pseudo_labels', {})
                results[input_['id']]['pseudo_labels'][classifier_name] = result['labels']

        results = list(filter(None,results))

        return docs, results, errors
    
    @classmethod
    def create_classifiers(cls, config : Config) -> Dict[str, ZeroShotClassifier]:

        models = {}

        for model in config.models:

            models[model.name] = ZeroShotClassifierRegistry.create(
                name=model.name,
                device=config.device,
                batch_size=config.batch_size,
                **model.args
            )

        return models
    
    @classmethod
    def from_config(cls, config : Config) -> 'CorpusAnnotatator':

        cache = TextfileCache(config.checkpoint_filename)
        
        cmd = CorpusAnnotatator(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            models=cls.create_classifiers(config),
            labels=[Label(
                name=d.sub_domain,
                description=d.definition,
                alt_names=d.alt_names
            ) for d in config.domains]
        )

        return cmd
