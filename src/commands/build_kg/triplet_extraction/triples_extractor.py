import dotenv
from typing import List, Dict, TypedDict
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache, Cache
from src.env import env, EntityType
from returns.pipeline import is_successful
from src.shared.triples_extractors.two_step_llm import TwoStepLLMTriplesExtractorWithOntology
from pycli import Field
from definitions import ENV
from instance import cli

class Entity(TypedDict):
    name : str
    types : List[str]

dotenv.load_dotenv(ENV)

class Config(StageConfig):

    type : str = Field(
        default='two-step',
        description=f"The type of the extractor used.",
    )

    llm : str = Field(default="deepseek-chat",description="The LLM to be used for triple extraction.")
    ontology : str = Field(default="industry_of_future", description="The ontology used to type the entities.")

@cli.register_stage(
    name="triples_extraction",
    description=(
        "Extract triples from sentences json file using an LLM-Based triples extractor componenet."
    ),
    add_from_conf=False,
    add_save_to=False
) 
@add_handlers
class TriplesExtractor(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        cache : Cache,
        triple_extractor : TwoStepLLMTriplesExtractorWithOntology,
        ontology : List[EntityType]
    ) -> None:
        super().__init__(reader, writer, cache)
        self.triple_extractor = triple_extractor
        self.ontology = ontology

    def transform(self, doc : Dict) -> Dict:
        return self.transform_chunk([doc])
    
    def transform_chunk(self, chunk : List[Dict]) -> List[Dict]:

        responses = self.triple_extractor.extract(
            [doc['sentence'] for doc in chunk],
            self.ontology
        )

        errors = []
        records = []
        docs = []

        for doc,response in zip(chunk, responses):

            if is_successful(response):

                triples = response.unwrap()
                docs.append(doc)
                
                for i,triple in enumerate(triples):

                    records.append({
                        "id" : f"{doc['id']}_{i}",
                        "head" : triple["head"],
                        "predicate" : triple["predicate"],
                        "tail" : triple["tail"],
                        "sentence" : doc
                    })
                    
            else:
                errors.append(response.failure())

        return docs, records, errors
    
    @classmethod
    def from_config(cls, config : Config) -> 'TriplesExtractor':

        cache = TextfileCache(config.checkpoint_filename)
        
        return TriplesExtractor(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size),
            cache=cache,
            triple_extractor=TwoStepLLMTriplesExtractorWithOntology(
                client=env.llms[config.llm].async_client,
                model=config.llm
            ),
            ontology=env.fields[config.ontology].ontology
        )
    
if __name__ == '__main__':

    from pprint import pprint

    example = {
        "id": "5471943_0", 
        "sentence": "Accurately estimating the latency and power consumption of software applications deployed on multicore systems remains a major challenge for early-stage optimization , as existing methods typically rely on slow and resource-intensive simulations .", 
        "document": {"id": "5471943", "title": "Assessing Graph Neural Networks for latency and power consumption prediction in application mappings on multicore architectures", "abstract": "Accurately estimating the latency and power consumption of software applications deployed on multicore systems remains a major challenge for early-stage optimization , as existing methods typically rely on slow and resource-intensive simulations . This paper explores modeling application-to-architecture mappings as heterogeneous graphs and investigates Graph Neural Networks ( GNNs ) for predicting application-to-architecture mappings performance . Four GNN models are evaluated across eleven datasets , considering five Neural Network-based software applications . The two best models achieve mean absolute percentage errors of about 2 % for power prediction and 15 % for latency , with prediction times of only a few tens of milliseconds . These results indicate the potential of GNN-based prediction as an efficient alternative to simulation-driven estimation , paving the way for early-stage AI-assisted mapping optimization .", "keywords": [{"keyword": "multicore", "source": "keyphrase_transformer"}, {"keyword": "power consumption", "source": "keyphrase_transformer"}, {"keyword": "latency estimation", "source": "keyphrase_transformer"}, {"keyword": "application-to-architecture mapping", "source": "keyphrase_transformer"}, {"keyword": "graph neural networks", "source": "keyphrase_transformer"}, {"keyword": "heterogeneous graphs", "source": "keyphrase_transformer"}, {"keyword": "gnn model", "source": "keyphrase_transformer"}, {"keyword": "neural network", "source": "keyphrase_transformer"}, {"keyword": "power prediction", "source": "keyphrase_transformer"}, {"keyword": "latency prediction", "source": "keyphrase_transformer"}, {"keyword": "gnn-based prediction", "source": "keyphrase_transformer"}, {"keyword": "simulation-driven estimation", "source": "keyphrase_transformer"}, {"keyword": "mapping", "source": "keyphrase_transformer"}], "authors": ["Oscar Roussel", "Zainab Ghrayeb", "S\u00e9bastien Le Nours", "Christine Sinoquet"], "venues": ["spi", "spi.other", "spi.tron"], "teams": ["LS2N-DUKE"], "date": "2026-04-22T00:00:00Z", "pseudo_labels": [{"model": "deepseek-chat", "sub_domains": {"Robotics": 3, "Processes and Assembly": 3, "Additive Manufacturing": 3, "Monitoring and Analysis": 2, "Eco-design": 2, "Reduction and Treatment of Effluents and Waste": 3, "Recycling / Reuse": 3, "Eco-responsible Processes": 2, "Decarbonization of Mobility": 3, "Decarbonization of Energy": 3, "Composites and New Materials": 3, "Industrial Digitalization": 2, "Digital Engineering of Products and Processes": 1, "Industrial Organization Performance": 2, "Preparation for New Jobs in the Future Industry": 3, "Change Management": 3, "User-Centered Design": 3, "Augmentation of Human Capabilities in Manufacturing": 3}, "domains": {"Advanced Production Technologies": 2, "Sustainable Industry": 2, "Industrial Digitalization": 1, "Industry 5.0": 3}}, {"model": "gemini-3.5-flash", "sub_domains": {"Robotics": 3, "Processes and Assembly": 3, "Additive Manufacturing": 3, "Monitoring and Analysis": 3, "Eco-design": 2, "Reduction and Treatment of Effluents and Waste": 3, "Recycling / Reuse": 3, "Eco-responsible Processes": 3, "Decarbonization of Mobility": 3, "Decarbonization of Energy": 3, "Composites and New Materials": 3, "Industrial Digitalization": 3, "Digital Engineering of Products and Processes": 1, "Industrial Organization Performance": 3, "Preparation for New Jobs in the Future Industry": 3, "Change Management": 3, "User-Centered Design": 3, "Augmentation of Human Capabilities in Manufacturing": 3}, "domains": {"Advanced Production Technologies": 3, "Sustainable Industry": 2, "Industrial Digitalization": 1, "Industry 5.0": 3}}]}
    }

    llm = 'deepseek-chat'

    extractor = TwoStepLLMTriplesExtractorWithOntology(
        client=env.llms[llm].async_client,
        model=llm
    )

    ontology = env.fields['industry_of_future'].ontology

    results = extractor.extract([example["sentence"]], ontology)

    pprint(results[0].unwrap())