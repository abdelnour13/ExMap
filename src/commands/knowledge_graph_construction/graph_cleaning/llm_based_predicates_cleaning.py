import numpy as np
import pandas as pd
import os
import asyncio
from nltk import pos_tag, word_tokenize
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Any, Literal, Optional
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import BatchedReader, BufferedWriter
from definitions import hf_model_dir
from pycli import Field
from instance import cli
from sklearn.neighbors import NearestNeighbors
from src.shared.env import env, LLM
from src.shared.prompts import PromptsManager
from returns.result import ResultE
from returns.pipeline import is_successful
from src.shared.file_tools import Batched, Map, FileReader, FileWriter, TextfileCache, Enumerate
from src import helpers

class Config(StageConfig):

    language_model : str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        description="The language model used to compute the similarity."
    )

    device : int = Field(
        default=-1, 
        description="The device used to perform the computation '-1' for cpu, '0' for gpu."
    )

    k : int = Field(
        default=16,
        description="The number of neighbors to consider."
    )

    llm_batch_size : int = 32

    llm_name : str = 'deepseek-v4-flash'

    max_distance : float = Field(
        default=0.7,
        description="The maximum distance for top-k neighbors retrieval."
    )

    num_iterations : int = Field(
        default=1,
        description="The number of times the operation should be repreated."
    )

    @property
    def llm(self) -> LLM:
        return env.llms[self.llm_name]

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
        language_model : str,
        llm : LLM,
        device : int,
        batch_size : int,
        llm_batch_size : int,
        k : int,
        max_distance : float,
        num_iterations : int,
        mapping_dir : str
    ) -> None:
        super().__init__(reader, writer)

        self.device = "cpu" if device == -1 else "cuda"
        self.batch_size = batch_size
        self.k = k
        self.llm = llm
        self.max_distance = max_distance
        self.llm_batch_size = llm_batch_size
        self.num_iterations = num_iterations
        self.mapping = None
        self.mapping_dir = mapping_dir

        self.language_model = SentenceTransformer(
            hf_model_dir(language_model),
            local_files_only=True,
            device=self.device
        )

    def get_predicates_df(self) -> pd.DataFrame:

        predicates = set()

        for records, _ in self.reader:

            for record in records:
                predicates.add(record['predicate'])

        return pd.DataFrame(data = {
            "predicate" : list(predicates)
        }).drop_duplicates()

    def tokenize_predicates(self, predicates_df : pd.DataFrame) -> pd.DataFrame:

        predicates_df = predicates_df.copy()

        predicates_df['tokens'] = predicates_df['predicate'].apply(word_tokenize)
        predicates_df['tags'] = predicates_df['tokens'].apply(pos_tag)

        return predicates_df

    def get_tokens_df(self, predicates_df : pd.Series) -> pd.Series:

        tokens_df = []

        for tokens_tags in predicates_df['tags']: 

            for token, tag in tokens_tags:

                tokens_df.append({
                    "token" : token,
                    "tag" : tag,
                })

        tokens_df = pd.DataFrame(tokens_df)
        mask = ~tokens_df['tag'].isin(['TO', 'IN'])
        tokens_df = tokens_df.drop_duplicates(subset='token')
        tokens_df = tokens_df.iloc[mask]

        return tokens_df

    def embeddings(self, sentences : List[str]) -> np.ndarray:

        X = self.language_model.encode(
            sentences=sentences,
            batch_size=self.llm_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return X

    def k_neighbors(self, X : np.ndarray) -> NearestNeighbors:
        return NearestNeighbors(n_neighbors=self.k, metric='cosine').fit(X)

    def most_similar(self,
        idx : int,
        nn : NearestNeighbors, 
        tokens : pd.Series,
    ) -> Tuple[List[str], List[int]]:
        
        distances, indices = nn.kneighbors(
            nn._fit_X[[idx],:], 
            n_neighbors=self.k, 
            return_distance=True
        )

        distances = distances.flatten()
        indices = indices.flatten()

        indices = indices[distances <= self.max_distance]

        if len(indices) <= 1:
            return [], []

        indices = indices[1:,]

        return tokens.iloc[indices].tolist(), indices.tolist()

    async def query_equivelant(
        self,
        target : str,
        candidates : List[str],
        kind : Literal['token', 'predicate']
    ) -> ResultE[Dict[str, str | List[str]]]:

        if len(candidates) == 0:
            return ResultE.from_value([])

        result = await PromptsManager.make_request(
            prompt_name='EquivelantPredicateTokensQuerying' if kind == 'token' else 'EquivelantPredicatesQuerying',
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
        kind : Literal['token', 'predicate']
    ) -> List[ResultE[List[str]]]:
        
        r = await asyncio.gather(*[
            self.query_equivelant(target=target, candidates=candidates, kind=kind)
            for target, candidates in input
        ])

        return r

    def batch_query_equivelant(
        self,
        input : List[Tuple[str, List[str]]],
        kind : Literal['token', 'predicate']
    ) -> List[ResultE[List[str]]]:
        r = asyncio.run(self._batch_query_equivelant(input=input, kind=kind))
        return r

    def _task_info(self, 
        target_id : int, 
        target : str, 
        nn : NearestNeighbors, 
        candidates : pd.Series,
    ) -> Dict[str, Any]:

        top_k_tokens, top_k_indices = self.most_similar(target_id, nn, candidates)

        return {
            "target_id" : target_id,
            "target" : target,
            "candidates" : top_k_tokens,
            "ids" : top_k_indices
        }

    async def query_representative_predicate(self, predicates : List[str]) -> ResultE[str]:

        result = await PromptsManager.make_request(
            prompt_name='RepresentativePredicateQuerying',
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

    async def _batch_query_representative_predicate(
        self,
        input : List[List[str]],
    ) -> List[ResultE[str]]:
        
        r = await asyncio.gather(*[
            self.query_representative_predicate(predicates=predicates)
            for predicates in input
        ])

        return r

    def batch_query_representative_predicate(
        self,
        input : List[List[str]],
    ) -> List[ResultE[str]]:
        r = asyncio.run(self._batch_query_representative_predicate(input=input))
        return r

    def get_mapping(self, tokens_df : pd.DataFrame, col : str) -> Dict[str, str]:

        X = self.embeddings(tokens_df[col].tolist())
        nn = self.k_neighbors(X)

        tasks_info = Batched(
            iterable=Map(
                iterable=Enumerate(tokens_df[col].tolist()),
                fn=lambda t : self._task_info(target_id=t[0], target=t[1], nn=nn, candidates=tokens_df[col])
            ),
            n=self.llm_batch_size
        )

        ### Query For Equivelant Sets
        visited = set()
        mapping = {}

        for batch in self.track(tasks_info, task="equivalent-subset-querying"):

            results = self.batch_query_equivelant([
                (task_info['target'], task_info['candidates'])
                for task_info in batch if task_info['target'] not in visited
            ], kind=col)

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
        return os.path.join(self.mapping_dir, f'mapping_{i}.json')

    def get_predicates_mapping(self) -> Dict[str, str]:

        predicates_df = self.get_predicates_df()
        predicates_df['predicate_original'] = predicates_df['predicate']

        mappings : List[Dict[str, str]] = []

        for i in range(self.num_iterations):

            if not os.path.exists(self.mapping_path(i)):
                mapping = self.get_mapping(tokens_df=predicates_df, col='predicate')
                predicates_df['predicate'] = predicates_df['predicate'].apply(lambda x : mapping.get(x, x))
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

        predicate = self.mapping.get(doc['predicate'], doc['predicate'])

        if predicate is not None and predicate != "":
            doc['predicate'] = predicate

        return doc

    def before_start(self):
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
            language_model=config.language_model,
            llm=config.llm,
            device=config.device,
            batch_size=config.batch_size,
            k=config.k,
            max_distance=config.max_distance,
            llm_batch_size=config.llm_batch_size,
            num_iterations=config.num_iterations,
            mapping_dir=config.mapping_dir
        )