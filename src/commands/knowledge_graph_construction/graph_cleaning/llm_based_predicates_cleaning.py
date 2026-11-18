import numpy as np
import pandas as pd
import os
from pycli import Field
from instance import cli
from returns.pipeline import is_successful
from typing import List, Tuple, Dict, Any, Optional
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import BatchedReader, BufferedWriter
from src.shared.file_tools import Batched, Map, FileReader, FileWriter, TextfileCache, Enumerate
from src import helpers
from src.methods.embedders import EmbedderRegistry, Embedder
from src.methods.nn import NNRetriever, NNRetrieverRegistry
from src.shared.platforms import LLM, RequestReturnType

class Config(StageConfig):

    ### PLM Config
    embedder : str = Field(default="sbert:sentence-transformers/all-MiniLM-L6-v2", description="The language model used to compute the similarity.")
    device : int = Field(default=-1, description="The device used to perform the computation '-1' for cpu, '0' for gpu.")
    plm_batch_size : int = 32

    ### LLM
    llm_batch_size : int = 32
    llm : str = 'deepseek-v4-flash'

    ### Search
    search_k : int = Field(default=16, description="The number of neighbors to consider.")
    search_m : int = Field(default=48, description="Number of subvectors")
    search_nbits : int = Field(default=8,description="Bits per subvector")
    search_nlist : Optional[int] = Field(default=None,description="Number of clusters, If None then nlist <- sqrt(num_datapoints)")
    search_nprobe : int = Field(default=10,description="Number of clusters to search in.")

    # Num iterations
    num_iterations : int = Field(default=2, description="The number of times the operation should be repreated.")

    @property
    def mapping_dir(self) -> str:
        return os.path.join(self.output_dir(), '_mappings')

@cli.register_stage(
    name="llm_based_predicate_cleaning",
    description=(
        "Perform predicates cleaning using Wikidata aliases attributes combined with Semantic similarity using a language model."
    ),
    add_from_conf=False,
    add_save_to=False
) 
@add_handlers
class IterativePredicatesCleaning(Stage):

    def __init__(self, 
        reader : BatchedReader, 
        writer : BufferedWriter, 
        embedder : Embedder,
        plm_batch_size : int,
        llm : LLM,
        llm_batch_size : int,
        nn : NNRetriever,
        num_iterations : int,
        mapping_dir : str
    ) -> None:
        super().__init__(reader, writer)

        self.nn = nn
        self.embedder = embedder
        self.plm_batch_size = plm_batch_size
        self.llm = llm
        self.llm_batch_size = llm_batch_size
        self.num_iterations = num_iterations
        self.mapping = None
        self.mapping_dir = mapping_dir

    def get_predicates_df(self) -> pd.DataFrame:

        predicates = set()

        for records, _ in self.reader:

            for record in records:
                predicates.add(record['predicate'])

        return pd.DataFrame(data = {
            "predicate" : list(predicates)
        }).drop_duplicates()

    def embeddings(self, sentences : List[str]) -> np.ndarray:
    
        iterator = Batched(iterable=sentences, n=self.plm_batch_size)
        iterator = self.track(it=iterator, task="feature-extraction", update_every=1)

        M = self.embedder.extract_query(data=iterator)
        M = M.numpy()

        return M

    def k_neighbors(self, X : np.ndarray) -> NNRetriever:
        return self.nn.fit(X)

    def most_similar(self,
        idx : int,
        nn : NNRetriever, 
        df : pd.Series,
    ) -> Tuple[List[str], List[int]]:
        
        distances, indices = nn.nn(nn.X[[idx],:])

        distances = distances.flatten()
        indices = indices.flatten()
        indices = indices[1:,]

        return df.iloc[indices].tolist(), indices.tolist()

    def query_equivelant_predicates(self, inputs : List[Dict]) -> List[RequestReturnType[Dict]]:

        result = self.llm.batch_make_async_request(
            prompt_name='EquivelantPredicatesQuerying',
            inputs=inputs,
            lang='en',
            validate_input=True,
            validate_output=True
        )

        return result

    def _task_info(self, 
        target_id : int, 
        target : str, 
        nn : NNRetriever, 
        candidates : pd.Series,
    ) -> Dict[str, Any]:

        top_k_tokens, top_k_indices = self.most_similar(target_id, nn, candidates)

        return {
            "target_id" : target_id,
            "target" : target,
            "candidates" : top_k_tokens,
            "ids" : top_k_indices
        }

    def get_mapping(self, df : pd.DataFrame) -> Dict[str, str]:

        X = self.embeddings(df['predicate'].tolist())
        nn = self.k_neighbors(X)

        tasks_info = Batched(
            iterable=Map(
                iterable=Enumerate(df['predicate'].tolist()),
                fn=lambda t : self._task_info(target_id=t[0], target=t[1], nn=nn, candidates=df['predicate'])
            ),
            n=self.llm_batch_size
        )

        ### Query For Equivelant Sets
        visited = set()
        mapping = {}

        for batch in self.track(tasks_info, task="equivalent-subset-querying"):

            batch = [
                task_info
                for task_info in batch if task_info['target'] not in visited
            ]

            results = self.query_equivelant_predicates([
                {
                    "target" : task_info['target'],
                    "candidates" : task_info['candidates']
                }
                for task_info in batch
            ])

            for result, task_info in zip(results, batch):

                if task_info['target'] in visited:
                    continue

                if not is_successful(result):
                    continue

                result = result.unwrap()

                if len(result['selected']) == 0:
                    continue

                target = task_info['target']
                mapping[target] = result['representative']

                for dup in result['selected']:

                    try:
                        mapping[dup] = result['representative']
                    except:
                        pass

                visited.update([task_info['target'], *result['selected']])

        return mapping

    def mapping_path(self, i : int) -> str:
        os.makedirs(self.mapping_dir, exist_ok=True)
        return os.path.join(self.mapping_dir, f'mapping_{i}.json')

    def get_predicates_mapping(self) -> Dict[str, str]:

        predicates_df = self.get_predicates_df()
        predicates_df['predicate_original'] = predicates_df['predicate']

        mappings : List[Dict[str, str]] = []

        for i in range(self.num_iterations):

            if not os.path.exists(self.mapping_path(i)):
                mapping = self.get_mapping(df=predicates_df)
                helpers.save_json_file(path=self.mapping_path(i), data=mapping)
            else:
                mapping = helpers.load_json_file(self.mapping_path(i))

            predicates_df['predicate'] = predicates_df['predicate'].apply(lambda x : mapping.get(x, x))
            mappings.append(mapping)

        final_mapping = mappings[0]

        for mapping in mappings[1:]:

            for k, v in final_mapping.items():
                v = v.strip()
                final_mapping[k] = mapping.get(v, v)

        return final_mapping

    def transform(self, doc : Dict) -> Dict:

        predicate = self.mapping.get(doc['predicate'], doc['predicate'])

        if predicate is not None and predicate != "":
            doc['predicate'] = predicate

        return doc

    def before_start(self):

        yield self.make_debug_event(
            progress=0.0,
            message="Faiss Index"
        )

        self.mapping = self.get_predicates_mapping()
        yield from []

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

            llm=LLM(name=config.llm),
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
