from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.handlers import Event, EventLevel
from src.methods.coref_resolvers import CorefResolver, CorefResolverRegistry
from typing import Dict
from pycli import Field
from instance import cli

class Config(StageConfig):

    model : str = Field(
        default="spanbert-large",
        description="The name of the allennlp model used for coreference resolution", 
    )

    device : int = Field(
        default=-1,
        description="The device id (CPU/GPU) used to preform the coreference resolution (if applicable)."
    )

@cli.register_stage(
    name="coreference_resolution",
    description="Resolve coreferences from abstracts as a preprocessing step.",
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class CoreferenceResolver(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        coref_resolver : CorefResolver
    ) -> None:
        super().__init__(reader, writer)
        self.coref_resolver = coref_resolver

    def before_start(self):
        
        yield Event(
            progress=0.0,
            level=EventLevel.INFO,
            extra={
                'total_processed' : 0,
                'total_saved' : 0,
                'total_skipped' : 0
            },
            message=f"Loading {self.coref_resolver.__class__.__name__}..."
        )


        yield Event(
            progress=0.0,
            level=EventLevel.DEBUG,
            extra={
                'total_processed' : 0,
                'total_saved' : 0,
                'total_skipped' : 0
            },
            message=f"Supports GPU: {self.coref_resolver.supports_gpu()}"
        )

        self.coref_resolver.load()

        yield Event(
            progress=0.0,
            level=EventLevel.INFO,
            extra={
                'total_processed' : 0,
                'total_saved' : 0,
                'total_skipped' : 0
            },
            message=f"{self.coref_resolver.__class__.__name__} Loaded."
        )

    def transform(self, doc : Dict) -> Dict:

        text = doc["abstract"]
        resolved_text = self.coref_resolver.resolve(text)        
        doc["abstract"] = resolved_text

        return doc

    @classmethod
    def from_config(cls, config : Config) -> Stage:

        cache = TextfileCache(config.checkpoint_filename)

        return CoreferenceResolver(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            coref_resolver=CorefResolverRegistry.create(config.model, device=config.device)
        )
    
if __name__ == '__main__':

    config = Config(
        input_filename="/home/fellah-a/Documents/Text2KG/data/corpus/ls2n-duke/data.jsonl",
        out="/home/fellah-a/Documents/Text2KG/data/corpus/ls2n-duke/out",
    )

    coreference_resolver = CoreferenceResolver.from_config(config)

    result = coreference_resolver.transform(
        { 
            "abstract" : """
            The main objective of this thesis is to combine the advantages of probabilistic graphical model learning and formal verification 
            in order to build a novel strategy for security assessments. The second objective is to assess the security of a given system 
            by veri- fying whether it satisfies given properties and, if not, how far is it from satisfying them. We are interested in 
            performing formal verification of this system based on event sequences col- lected from its execution. Consequently, 
            we propose a model-based approach where a Re- cursive Timescale Graphical Event Model (RT- GEM), learned from the event streams, 
            is con- sidered to be representative of the underlying system. This model is then used to check a se- curity property. 
            If the property is not verified, we propose a search methodology to find an- other close model that satisfies it.
            """
        }
    )

    print(result)