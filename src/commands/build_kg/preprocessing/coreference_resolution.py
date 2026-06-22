import re
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.handlers import Event, EventLevel
from src.rpc.interface import allennlp_predict, load_predictor_from_path
from typing import Dict
from pycli import Field, Model
from instance import cli

MODELS = {
    "spanbert-large" : "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
}

class Config(StageConfig):

    model : str = Field(
        default="spanbert-large",
        description="The name of the allennlp model used for coreference resolution", 
    )

    device : int = Field(
        default=-1,
        description="The device id (CPU/GPU) used to preform the job."
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
        model : str,
        device : int
    ) -> None:
        super().__init__(reader, writer)
        self.model = model
        self.device = device

    def before_start(self):
        
        yield Event(
            progress=0.0,
            level=EventLevel.INFO,
            extra={
                'total_processed' : 0,
                'total_saved' : 0,
                'total_skipped' : 0
            },
            message=f"Loading AllenNLP-Coreference resolver..."
        )

        load_predictor_from_path(MODELS[self.model], self.device)

        yield Event(
            progress=0.0,
            level=EventLevel.INFO,
            extra={
                'total_processed' : 0,
                'total_saved' : 0,
                'total_skipped' : 0
            },
            message=f"AllenNLP-Coreference Loaded."
        )

    def transform(self, doc : Dict) -> Dict:

        text = doc["abstract"]

        ### Use RPC-Server to do the AllenNLP Prediction
        result : Dict = allennlp_predict(MODELS[self.model], self.device, text)

        ### Resolve The Text
        tokens = result['document']
        clusters = result["clusters"]

        ### Build Replacements Dictionary
        replacements = {}

        for cluster in clusters:

            main_start, main_end = cluster[0]
            main_mention = " ".join(tokens[main_start: main_end + 1])

            for start, end in cluster[1:]:
                replacements[start] = (main_mention, end)

        ### Build New Lists Of Tokens
        new_tokens = []
        i = 0

        while i < len(tokens):
            if i in replacements:
                main_mention, end = replacements[i]
                new_tokens.append(main_mention)
                i = end + 1   # skip the whole span
            else:
                new_tokens.append(tokens[i])
                i += 1

        ### Join & Clean Tokenizations Side Effects
        resolved_text = " ".join(new_tokens)

        # Fix whitespace/tokenization issues
        resolved_text = re.sub(r"\s+", " ", resolved_text)
        resolved_text = re.sub(r"\s-", "-", resolved_text)
        resolved_text = re.sub(r"-\s", "-", resolved_text)

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
            model=config.model,
            device=config.device
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