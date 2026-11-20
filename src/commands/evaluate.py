import os
import numpy as np
import pandas as pd
from dataclasses import asdict
from functools import cached_property
from typing import Any, Dict, List
from pydantic import BaseModel
from returns.pipeline import is_successful
from definitions import EMS_DIR, EVAL_DIR
from instance import cli
from src.helpers import utils
from src.commands.feature_extraction.graph_utils import Graph, EdgeType
from src.shared.command_tools import BaseCommand
from src.shared.file_tools import FileWriter, FileReader, Batched, Product
from src.shared.handlers import ConsoleLogger
from src.shared.platforms import LLM


ALIGN_PROMPT = 'ExpertiseDomainAlignment'
COVER_PROMPT = 'ExpertiseEntitiesCoverage'


def to_dict(n) -> dict:
    d = asdict(n)
    p = d.pop('properties', {})
    return {**d, **p}


class EvaluationConfig(BaseModel):

    expertise_id: str
    llm: str
    batch_size: int
    lang: str = 'en'

    @property
    def eval_dir(self) -> str:
        dir = os.path.join(EVAL_DIR, self.expertise_id)
        os.makedirs(dir, exist_ok=True)
        return dir

    @property
    def expertise_dir(self) -> str:
        return os.path.join(EMS_DIR, self.expertise_id)

    @cached_property
    def graph(self) -> Graph:
        return Graph.load(self.expertise_dir)

    @cached_property
    def cache_path(self) -> str:

        path = os.path.join(self.eval_dir, 'cache.jsonl')

        with open(path, "w"):
            pass

        return path

    @cached_property
    def file_writer(self) -> FileWriter:
        return FileWriter.from_filename(self.cache_path)

    @cached_property
    def file_reader(self) -> FileReader:
        return FileReader.from_filename(self.cache_path)

    @cached_property
    def graph_dir(self) -> str:
        dir = os.path.join(EVAL_DIR, self.expertise_id, 'graph')
        os.makedirs(dir, exist_ok=True)
        return dir

class Config(BaseModel):
    evaluation_id : str

    @property
    def evaluation_config(self) -> EvaluationConfig:
        path = os.path.join(EVAL_DIR, self.evaluation_id, "config.json")
        raw_data = utils.load_json_file(path)
        return EvaluationConfig(**raw_data)

@cli.register_stage(
    name="evaluate",
    description=(
        "Runs the pipeline responsible for LLM-based expertise map evaluation."
    ),
    add_from_conf=False,
    add_save_to=False,
)
class ExpertiseMapEvaluator(BaseCommand):

    def __init__(self,
        llm: LLM,
        graph: Graph,
        batch_size: int,
        writer: FileWriter,
        reader: FileReader,
        graph_dir: str,
        lang: str = 'en'
    ) -> None:
        super().__init__()

        self.llm = llm
        self.graph = graph
        self.batch_size = batch_size
        self.writer = writer
        self.reader = reader
        self.graph_dir = graph_dir
        self.lang = lang

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #

    def dataframe(self) -> pd.DataFrame:

        graph = self.graph

        df = pd.DataFrame([
            to_dict(e)
            for e in graph._edges[EdgeType(src="Entity", dst="Expertise", name="BELONGSTO")]._edges
        ])

        entities = pd.DataFrame([to_dict(e) for e in graph._nodes['Entity']._nodes])
        expertises = pd.DataFrame([to_dict(e) for e in graph._nodes['Expertise']._nodes])

        entities.columns = [f'{col}_entity' for col in entities.columns]
        expertises.columns = [f'{col}_expertise' for col in expertises.columns]

        df = df.merge(right=expertises[['id_expertise', 'llm_label_expertise']], left_on='dst', right_on='id_expertise')
        df = df.merge(right=entities, left_on='src', right_on='id_entity')
        df = df.drop(columns=['src', 'dst', 'id'])

        df = df.groupby('id_expertise').agg({
            'llm_label_expertise': 'first',
            'id_entity': list,
            'name_entity': list
        }).reset_index()

        return df

    # ------------------------------------------------------------------ #
    # LLM requests
    # ------------------------------------------------------------------ #

    def _run_prompt(self, prompt_name: str, inputs: List[Dict[str, Any]]) -> List[Dict]:

        """Runs a batch of requests and wraps each output in the cache format:
        {"result": {...}} on success, {"error": {...}} on failure."""

        outputs = self.llm.batch_make_async_request(
            prompt_name=prompt_name,
            inputs=inputs,
            lang=self.lang,
        )

        results = []

        for out in outputs:

            if is_successful(out):
                results.append({"result": out.unwrap()})
            else:
                failure = out.failure()
                results.append({
                    "error": {
                        "name": failure.__class__.__name__,
                        "message": str(failure),
                    }
                })

        return results

    def align_batch(self, batch: List[Dict]) -> List[Dict]:

        return self._run_prompt(
            prompt_name=ALIGN_PROMPT,
            inputs=[
                {"expertise_name": task['expertise_name'], "domain": task['domain']}
                for task in batch
            ],
        )

    def cover_batch(self, batch: List[Dict]) -> List[Dict]:
        
        return self._run_prompt(
            prompt_name=COVER_PROMPT,
            inputs=[
                {"expertise_name": task['expertise_name'], "entities": ' | '.join(task['entities'])}
                for task in batch
            ],
        )

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    @staticmethod
    def align_score(row: pd.Series) -> float | None:

        scores = [
            score
            for domain, score in zip(row['domain_id'], row['score'])
            if domain in row['dst']
        ]

        return sum(scores) / len(scores) if scores else None

    def compute_scores(self) -> pd.DataFrame:

        df = pd.read_json(self.reader.filename, lines=True)

        # Failed tasks are re-run and appended, so keep only the latest
        # entry per task and drop the ones that are still failing.
        df = df.drop_duplicates(subset='task_id', keep='last')
        df = df[df['result'].apply(lambda x: 'result' in x)]
        df['score'] = df['result'].apply(lambda x: x['result']['score'])

        ### Expertise -> SubDomain
        expertise2subdomain = pd.DataFrame([
            to_dict(e)
            for e in self.graph._edges[EdgeType(src="Expertise", dst="SubDomain", name="BELONGSTO")]._edges
        ])
        expertise2subdomain = expertise2subdomain[['src', 'dst']].groupby('src').agg(list).reset_index()

        ### Align Score (DAS)
        df_align = df.loc[df['task'] == 'align', ['expertise_id', 'domain_id', 'score']]
        df_align = df_align.groupby(by='expertise_id').agg(list).reset_index()
        df_align = df_align.merge(right=expertise2subdomain, left_on='expertise_id', right_on='src')
        df_align['DAS'] = df_align.apply(self.align_score, axis=1)
        df_align = df_align[['expertise_id', 'DAS']]

        ### Cover Score (CCS)
        df_cover = df.loc[df['task'] == 'cover', ['expertise_id', 'score']]
        df_cover = df_cover.rename(columns={'score': 'CCS'})

        ### CWAS
        df_scores = df_align.merge(df_cover, on='expertise_id')

        CCS = df_scores['CCS'].to_numpy(dtype=float)
        DAS = df_scores['DAS'].to_numpy(dtype=float)
        df_scores['CWAS'] = np.sqrt(CCS * DAS)

        return df_scores

    def update_graph(self, data: pd.DataFrame) -> None:

        for _, row in data.iterrows():
            
            self.graph.add_node(
                type='Expertise',
                id=row['expertise_id'],
                properties={
                    "DAS": row['DAS'],
                    "CCS": row['CCS'],
                    "CWAS": row['CWAS']
                }
            )

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #

    def _load_cache(self) -> Dict[str, bool]:
        """task_id -> True if the cached result is an error (must be re-run)."""
        return {
            doc['task_id']: "error" in doc['result']
            for doc, _ in self.reader.generate()
        }

    @staticmethod
    def _needs_run(task: Dict, cache: Dict[str, bool]) -> bool:
        return cache.get(task['task_id'], True)

    def _process(self, tasks, task_name: str, build, run) -> None:

        cache = self._load_cache()

        for batch in self.track(it=tasks, task=task_name):

            batch = [build(item) for item in batch]
            batch = [task for task in batch if self._needs_run(task, cache)]

            if not batch:
                continue

            for task, result in zip(batch, run(batch)):
                task['result'] = result

            self.writer.write(batch)

    def __iter__(self):

        df = self.dataframe()
        records = df.to_dict(orient='records')
        domains = list(map(to_dict, self.graph._nodes['SubDomain']._nodes))

        ### Alignment Score
        yield self.make_info_event(progress=0.0, message="Starting Alignment Score calculation")

        def build_align(item) -> Dict:
            expertise, domain = item
            return {
                "task_id": f"{expertise['id_expertise']}_{domain['id']}_align",
                "expertise_id": expertise['id_expertise'],
                "expertise_name": expertise['llm_label_expertise'],
                "domain": domain['name'],
                "domain_id": domain['id'],
                "entities": expertise['name_entity'],
                "task": "align",
            }

        self._process(
            tasks=Batched(Product(records, domains), self.batch_size),
            task_name="alignment-score",
            build=build_align,
            run=self.align_batch,
        )

        ### Cover Score
        yield self.make_info_event(progress=0.5, message="Starting Cover Score calculation")

        def build_cover(expertise) -> Dict:
            return {
                "task_id": f"{expertise['id_expertise']}_cover",
                "expertise_id": expertise['id_expertise'],
                "expertise_name": expertise['llm_label_expertise'],
                "entities": expertise['name_entity'],
                "task": "cover",
            }

        self._process(
            tasks=Batched(records, self.batch_size),
            task_name="cover-score",
            build=build_cover,
            run=self.cover_batch,
        )

        ### Compute Scores
        yield self.make_info_event(progress=0.8, message="Computing Scores")
        scores_df = self.compute_scores()

        ### Update Graph
        yield self.make_info_event(progress=0.9, message="Updating Graph")
        self.update_graph(scores_df)
        self.graph.save(self.graph_dir)

        yield self.make_info_event(progress=1.0, message="Graph Update Completed")

    @classmethod
    def from_config(cls, config: Config) -> 'ExpertiseMapEvaluator':

        evaluation_config = config.evaluation_config

        evaluator = cls(
            graph=evaluation_config.graph,
            llm=LLM(evaluation_config.llm),
            batch_size=evaluation_config.batch_size,
            writer=evaluation_config.file_writer,
            reader=evaluation_config.file_reader,
            graph_dir=evaluation_config.graph_dir,
            lang=evaluation_config.lang,
        )

        evaluator.add_handler(ConsoleLogger("ExpertiseMapEvaluator"))
        return evaluator