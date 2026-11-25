from src.commands.knowledge_graph_construction.triplet_extraction import triples_extractor
from src.methods.triple_extractors import TwoStepLLMTripleExtractor, LLM
from src.shared.file_tools import TextfileCache, FileReader, FileWriter
from src.shared.command_tools import add_handlers

class Config(triples_extractor.Config):
    api_key : str

@add_handlers
class TriplesExtractor(triples_extractor.TriplesExtractor):

    @classmethod
    def from_config(cls, config : Config) -> 'TriplesExtractor':

        cache = TextfileCache(config.checkpoint_filename)

        model = config.extractor.split(':')[1]
        llm = LLM(name=model, api_key=config.api_key)
        
        return TriplesExtractor(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size),
            cache=cache,
            triple_extractor=TwoStepLLMTripleExtractor(llm=llm),
        )