import numpy as np
import pandas as pd
import os
import math
import asyncio
import faiss
import json
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any, Optional
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import BatchedReader, BufferedWriter
from definitions import hf_model_dir
from pycli import Field
from instance import cli
from src.shared.env import env, LLM
from src.shared.prompts import PromptsManager
from returns.result import ResultE
from returns.pipeline import is_successful
from src.shared.file_tools import Batched, Map, FileReader, FileWriter, TextfileCache, Enumerate
from src import helpers
from src.shared.embedders import Embedder

class Config(StageConfig):

    ### pre-trained language model params
    pretrained_language_model : str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        description="The language model used to compute the similarity."
    )

    plm_device : int = Field(
        default=-1, 
        description="The device used to perform the computation '-1' for cpu, '0' for gpu."
    )

    plm_batch_size : int = 128

    ### Large Language Model
    llm_batch_size : int = 32
    llm_name : str = 'deepseek-v4-flash'

    ### Search
    search_k : int = Field(default=16, description="The number of neighbors to consider.")
    search_m : int = Field(default=48, description="Number of subvectors")
    search_nbits : int = Field(default=8,description="Bits per subvector")
    search_nlist : Optional[int] = Field(default=None,description="Number of clusters, If None then nlist <- sqrt(num_datapoints)")
    search_nprobe : int = Field(default=10,description="Number of clusters to search in.")

    num_iterations : int = Field(default=1, description="Number of entities steps.")

    @property
    def llm(self) -> LLM:
        return env.llms[self.llm_name]

    @property
    def mapping_dir(self) -> str:
        path = os.path.join(self.output_dir(), '_mappings')
        os.makedirs(path, exist_ok=True)
        return path


@cli.register_stage(
    name="llm_based_entities_cleaning",
    description=(
        "Perform predicates cleaning using Wikidata aliases attributes combined with Semantic similarity using a language model."
    ),
    add_from_conf=False,
    add_save_to=False
) 
@add_handlers
class IterativeLLMBasedEntitiesCleaning(Stage):

    def __init__(self, 
                 
        reader : BatchedReader, 
        writer : BufferedWriter, 

        pretrained_language_model : str,
        plm_device : int,
        plm_batch_size : int,

        llm : LLM,
        llm_batch_size : int,

        search_k : int,
        search_m : int,
        search_nbits : int,
        search_nlist : int,
        search_nprobe : int,

        num_iterations : int,
        mapping_dir : str
        
    ) -> None:
        
        super().__init__(reader, writer)

        self.pretrained_language_model = pretrained_language_model
        self.plm_device = 'cpu' if plm_device == -1 else 'gpu'
        self.plm_batch_size = plm_batch_size

        self.llm = llm
        self.llm_batch_size = llm_batch_size

        self.search_k = search_k
        self.search_m = search_m
        self.search_nbits = search_nbits
        self.search_nlist = search_nlist
        self.search_nprobe = search_nprobe

        self.num_iterations = num_iterations
        self.mapping = None

        self.mapping_dir = mapping_dir

        self.embedder = Embedder(
            model=SentenceTransformer(
                hf_model_dir(pretrained_language_model),
                local_files_only=True,
                device=self.plm_device
            )
        )

    def get_entities_df(self) -> pd.DataFrame:
    
        entities = set()

        for records, _ in self.reader:

            for record in records:
                entities.add(record['head'])
                entities.add(record['tail'])

        return pd.DataFrame(data = {
            "entity" : list(entities)
        }).drop_duplicates()

    def embeddings(self, sentences : List[str]) -> np.ndarray:

        iterator = Batched(iterable=sentences, n=self.plm_batch_size)
        iterator = self.track(it=iterator, task="feature-extraction", update_every=1)

        M = self.embedder.extract_query(data=iterator)
        M = M.numpy()

        return M

    def make_faiss_index(self, X : np.ndarray) -> faiss.IndexIVFPQ:

        N, D = X.shape
        search_nlist = self.search_nlist or int(math.sqrt(N))

        print(search_nlist)

        index = faiss.IndexIVFPQ(
            faiss.IndexFlatIP(D),
            D,
            search_nlist,
            self.search_m,
            self.search_nbits,
            faiss.METRIC_INNER_PRODUCT,
        )

        index.train(X)
        index.add(X)
        index.nprobe = self.search_nprobe

        return index

    def most_similar(self,
        idx : int,
        index : faiss.IndexIVFPQ, 
        df : pd.Series,
        X : np.ndarray
    ) -> Tuple[List[str], List[int]]:

        Q = X[[idx],:]
        _, indices = index.search(x=Q, k=self.search_k)

        indices = indices[0]
        indices = indices[indices >= 0]
        indices = indices[indices != idx]

        return df.iloc[indices].tolist(), indices.tolist()

    async def query_representative_entity(self, predicates : List[str]) -> ResultE[str]:
    
        result = await PromptsManager.make_request(
            prompt_name='RepresentativeEntityQuerying',
            client=self.llm.async_client,
            model_name=self.llm.info.name,
            input={
                "predicates" : predicates,
            },
            lang='en',
            validate_input=True,
            validate_output=True
        )

        result = result.map(lambda x : x['representative'])

        return result

    async def _batch_query_representative_entity(
        self,
        input : List[List[str]],
    ) -> List[ResultE[str]]:
        
        r = await asyncio.gather(*[
            self.query_representative_entity(predicates=predicates)
            for predicates in input
        ])

        return r

    def batch_query_representative_entity(
        self,
        input : List[List[str]],
    ) -> List[ResultE[str]]:
        r = asyncio.run(self._batch_query_representative_entity(input=input))

    async def query_equivelant(
        self,
        target : str,
        candidates : List[str],
    ) -> ResultE[Dict[str, str | List[str]]]:

        if len(candidates) == 0:

            return ResultE.from_value({
                "selected" : [],
                "representative" : None
            })

        result = await PromptsManager.make_request(
            prompt_name='EquivalentEntitiesQuerying',
            client=self.llm.async_client,
            model_name=self.llm.info.name,
            input={
                "target" : target,
                "candidates" : candidates
            },
            lang='en',
            validate_input=True,
            validate_output=True
        )

        return result

    async def _batch_query_equivelant(
        self,
        input : List[Tuple[str, List[str]]],
    ) -> List[ResultE[List[str]]]:
        
        r = await asyncio.gather(*[
            self.query_equivelant(target=target, candidates=candidates)
            for target, candidates in input
        ])

        return r

    def batch_query_equivelant(
        self,
        input : List[Tuple[str, List[str]]],
    ) -> List[ResultE[List[str]]]:
        r = asyncio.run(self._batch_query_equivelant(input=input))
        return r

    def _task_info(self, 
        target_id : int, 
        target : str, 
        index : faiss.IndexIVFPQ, 
        candidates : pd.Series,
        X : np.ndarray
    ) -> Dict[str, Any]:

        top_k_tokens, top_k_indices = self.most_similar(target_id, index, candidates, X)

        return {
            "target_id" : target_id,
            "target" : target,
            "candidates" : top_k_tokens,
            "ids" : top_k_indices
        }

    def get_mapping(self, df : pd.DataFrame) -> Dict[str, str]:

        col = 'entity'
        X = self.embeddings(sentences=df[col].tolist())
        index = self.make_faiss_index(X)

        tasks_info = Batched(
            iterable=Map(
                iterable=Enumerate(df[col].tolist()),
                fn=lambda t : self._task_info(target_id=t[0], target=t[1], index=index, candidates=df[col], X=X)
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

            results = self.batch_query_equivelant([
                (task_info['target'], task_info['candidates'])
                for task_info in batch
            ])

            for result, task_info in zip(results, batch):

                if task_info['target'] in visited:
                    continue

                if not is_successful(result):
                    continue

                result = result.unwrap()
                result['candidates'] = task_info['candidates']

                with open("/home/fellah-a/Documents/Text2KG/data/knowledge_graphs/industry_of_future_v2/llm_based_entities_cleaning/debug.jsonl", "a") as f:
                    f.write(json.dumps(result) + "\n")

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
        return os.path.join(self.mapping_dir, f'mapping_{i}.json')
    
    def get_entities_mapping(self) -> Dict[str, str]:

        entities_df = self.get_entities_df()
        entities_df['entity_original'] = entities_df['entity']

        mappings : List[Dict[str, str]] = []

        for i in range(self.num_iterations):

            if not os.path.exists(self.mapping_path(i)):
                mapping = self.get_mapping(df=entities_df)
                entities_df['entity'] = entities_df['entity'].apply(lambda x : mapping.get(x, x))
                helpers.save_json_file(path=self.mapping_path(i), data=mapping)
            else:
                mapping = helpers.load_json_file(self.mapping_path(i))

            mappings.append(mapping)

        final_mapping = mappings[0]

        for mapping in mappings[1:]:

            for k, v in final_mapping.items():
                v = v.strip()
                final_mapping[k] = mapping.get(v, v)

        return final_mapping

    def transform(self, doc : Dict) -> Dict:
    
        head = self.mapping.get(doc['head'], doc['head']).strip()
        tail = self.mapping.get(doc['tail'], doc['tail']).strip()

        if head is not None and head != "":
            doc['head'] = head

        if tail is not None and tail != "":
            doc['tail'] = tail

        return doc

    def before_start(self):
        self.mapping = self.get_entities_mapping()
        yield from []

    @classmethod
    def from_config(cls, config: Config) -> Stage:

        cache = TextfileCache(config.checkpoint_filename)

        return IterativeLLMBasedEntitiesCleaning(

            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),

            pretrained_language_model=config.pretrained_language_model,
            plm_batch_size=config.plm_batch_size,
            plm_device=config.plm_device,

            llm=config.llm,
            llm_batch_size=config.llm_batch_size,

            search_k=config.search_k,
            search_m=config.search_m,
            search_nbits=config.search_nbits,
            search_nlist=config.search_nlist,
            search_nprobe=config.search_nprobe,

            num_iterations=config.num_iterations,
            mapping_dir=config.mapping_dir
        )