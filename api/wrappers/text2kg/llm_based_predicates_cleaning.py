from src.commands.knowledge_graph_construction.graph_cleaning import llm_based_predicates_cleaning
from src.shared.command_tools import Stage, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.methods.embedders import EmbedderRegistry
from src.shared.platforms import LLM
from src.methods.nn import NNRetrieverRegistry

class Config(llm_based_predicates_cleaning.Config):
    api_key : str

@add_handlers
class IterativePredicatesCleaning(llm_based_predicates_cleaning.IterativePredicatesCleaning):

    @classmethod
    def from_config(cls, config: Config) -> Stage:

        cache = TextfileCache(config.checkpoint_filename)

        return IterativePredicatesCleaning(

            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),

            embedder=EmbedderRegistry.create(
                name=config.embedder,
                device=config.device
            ),
            plm_batch_size=config.plm_batch_size,

            llm=LLM(name=config.llm, api_key=config.api_key),
            llm_batch_size=config.llm_batch_size,
            
            nn=NNRetrieverRegistry.create(
                name="faiss",
                k=config.search_k,
                search_m=config.search_m,
                search_nbits=config.search_nbits,
                search_nlist=config.search_nlist,
                search_nprobe=config.search_nprobe
            ),
            num_iterations=config.num_iterations,
            mapping_dir=config.mapping_dir
        )
