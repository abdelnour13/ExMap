import torch
from typing import Dict, List, Any
from src.shared.command_tools.vasync import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.helpers import load_json_file
from src.shared.zs_classifiers import ZeroShotClassifier
from src.commands.helpers import DomainConfig, Domain
from pycli import Field
from definitions import domain_dir
from instance import cli
from returns.pipeline import is_successful
from pydantic import BaseModel

class ClassifierConfig(BaseModel):
    name : str
    args : Dict[str, Any] = Field(default_factory=dict)

class Config(StageConfig, DomainConfig('domain')):

    models : List[ClassifierConfig] = Field(
        default_factory=lambda : [ClassifierConfig(name="llm:deepseek-chat", args={ "prompt_name" : "binary" })],
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
    ) -> None:
        super().__init__(reader, writer)

        self.models = models

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

            for input_, result, doc in zip(inputs, classifier.classify(inputs), batch):

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

            models[model.name] = ZeroShotClassifier.from_name(
                name=model.name,
                labels=[
                    {
                        "name" : domain.sub_domain,
                        "description" : domain.definition,
                        "alt_names" : domain.alt_names
                    }
                    for domain in config.domains
                ],
                device=config.device,
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
        )

        return cmd
    
if __name__ == '__main__':
    
    example = {
        "id": "4175668", 
        "title": "Incremental discovery of new defects: application to screwing process monitoring", 
        "abstract": "Defect detection by in-process monitoring plays a key role in the traceability and optimization of production. Many fault detection algorithms are trained on known faults. However, industrial data is generally unlabeled and certain faults are unknown or missing in the training dataset. This paper presents an original approach for the incremental discovery of new manufacturing defects, by Bayes rule and distance rejection. Rejects are analyzed periodically to determine the possible appearance of new defect cluster among them. Visualization then supports the cluster interpretation by a manufacturing expert. The approach was successfully applied to a screwing database from automotive industry.", 
        "keywords": ["Monitoring", "Machine learning", "User interaction"], 
        "authors": ["Mahmoud Ferhat", "Mathieu Ritou", "Philippe Leray", "Nicolas Le Du"], 
        "venues": ["info.info-lg", "spi.meca.geme"], 
        "teams": ["LS2N-DUKE"], 
        "date": "2021-06-01T00:00:00Z"
    }


    domains = [Domain(**info) for info in load_json_file(domain_dir('industry_of_future'))]

    annotator = CorpusAnnotatator(
        None, 
        None, 
        {
            "llm" : ZeroShotClassifier.from_name(
                name="llm:deepseek-chat",
                device="cpu",
                labels=[
                    {
                        "name" : domain.sub_domain,
                        "description" : domain.definition
                    }
                    for domain in domains
                ],
                prompt_name="binary",
                mode="seperate"
            )
        }, 
        domains=domains
    )

    result = annotator.transform(example)

    from pprint import pprint

    pprint(result)
