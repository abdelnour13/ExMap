import os
import asyncio
import json
import pandas as pd
import numpy as np
from pydantic import BaseModel
from functools import cached_property
from itertools import batched, product
from src.commands.feature_extraction_old import Graph, EdgeType
from src.shared.command_tools import BaseCommand
from src.shared.handlers import ConsoleLogger
from src.shared.env import env, LLM
from src import helpers
from dataclasses import asdict
from definitions import EMS_DIR, EVAL_DIR
from src.shared.file_tools import FileWriter, FileReader, Batched, Product
from typing import TypedDict, Dict, List, TypeVar
from returns.pipeline import is_successful
from instance import cli


T = TypeVar("T")

class AlignmentOutput(TypedDict):
    score : float

def to_dict(n) -> dict:

    d = asdict(n)
    p = d.pop('properties', {})
    d = { **d, **p }

    return d

class Config(BaseModel):

    expertise_id: str 
    llm : str
    batch_size : int

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

    @property
    def cache_path(self) -> str:
        return os.path.join(self.eval_dir, 'cache.jsonl')

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
        llm : LLM, 
        graph : Graph,
        batch_size : int,
        writer : FileWriter,
        reader : FileReader,
        graph_dir : str
    ) -> None:
        super().__init__()

        self.llm = llm
        self.graph = graph
        self.batch_size = batch_size
        self.writer = writer
        self.reader = reader
        self.graph_dir = graph_dir

    def dataframe(self) -> pd.DataFrame:

        graph = self.graph 

        df = pd.DataFrame([
            to_dict(e) 
            for e in graph._edges[EdgeType(src="Entity", dst="Expertise", name="BELONGSTO")]._edges
        ])

        entities = pd.DataFrame([
            to_dict(e)
            for e in graph._nodes['Entity']._nodes
        ])

        expertises = pd.DataFrame([
            to_dict(e)
            for e in graph._nodes['Expertise']._nodes
        ])

        entities.columns = [f'{col}_entity' for col in entities.columns]
        expertises.columns = [f'{col}_expertise' for col in expertises.columns]

        df = df.merge(right=expertises[['id_expertise', 'llm_label_expertise']], left_on='dst', right_on='id_expertise')
        df = df.merge(right=entities, left_on='src', right_on='id_entity')
        df = df.drop(columns=['src', 'dst', 'id'])

        df = df.groupby('id_expertise').agg({
            'llm_label_expertise' : 'first',
            'id_entity' : list,
            'name_entity' : list
        }).reset_index()

        return df

    async def align(self, 
        expertise_name : str,
        domain : str
    ) -> Dict:
        
        result = await helpers.make_openai_request(
            client=self.llm.async_client,
            model=self.llm.info.name,
            messages=[
                {
                    "role" : "system",
                    "content" : """

                        You are an expert in scientific classification and domain relevance assessment. 
                        Your task is to evaluate the degree of alignment between a research expertise and a target scientific domain. 
                        The research expertise is defined by its title. 
                        The domain is defined by its name. 

                        Analyze the expertise' main objectives and application area to dermine the degree of relevance to
                        the scientific domain.

                        ### **Output Format** :
                            - Score between 0 (not alignment) and 1 (perfect alignment).
                            - The return should be in json format and should contain only one field named "score" of type "float" :
                            {
                                "score" : {score}
                            }
                            - Important : Do not provide explanations, reasoning or additional text.
                        
                    """,
                },
                {
                    "role" : "user",
                    "content" : """
                        - Expertise name: {expertise_name}
                        - Label / Domain: {domain}
                    """.format(expertise_name=expertise_name, domain=domain)
                }
            ]
        )

        content = result.map(lambda x : x.choices[0].message.content)
        result = content.map(json.loads)

        if is_successful(result):
            result = result.unwrap()
            return { "result" : result }
        else:
            return { 
                "error" : {
                    "name" : result.failure().__class__.__name__,
                    "raw_content" : content.unwrap() if is_successful(content) else None
                } 
            }

    async def cover(self, 
        expertise_name : str,
        entities : List[str]
    ) -> Dict:
        
        result = await helpers.make_openai_request(
            client=self.llm.async_client,
            model=self.llm.info.name,
            messages=[
                {
                    "role" : "system",
                    "content" : """

                        You are an expert in scientific text analysis. 
                        Your task is to evaluate the degree of coverage between a research expertise and a setf of entities. 
                        The research expertise is defined by its title. 
                        The entities are defined by their names. 

                        Analyze the expertise' main objectives and application area to dermine the degree of relevance to
                        the entities.

                        ### **Output Format** :
                            - Score between 0 (not alignment) and 1 (perfect alignment).
                            - The return should be in json format and should contain only one field named "score" of type "float" :
                            {
                                "score" : {score}
                            }
                            - Important : Do not provide explanations, reasoning or additional text.
                        
                    """,
                },
                {
                    "role" : "user",
                    "content" : """
                        - Expertise name: {expertise_name}
                        - Entities: {entities}
                    """.format(expertise_name=expertise_name, entities=' | '.join(entities))
                }
            ]
        )

        content = result.map(lambda x : x.choices[0].message.content)
        result = content.map(json.loads)

        if is_successful(result):
            result = result.unwrap()
            return { "result" : result }
        else:
            return { 
                "error" : {
                    "name" : result.failure().__class__.__name__,
                    "raw_content" : content.unwrap() if is_successful(content) else None
                } 
            }

    async def _align_batch(self, batch : List[Dict]) -> List[Dict]:

        results = await asyncio.gather(*[
            self.align(expertise_name=task_info['expertise_name'], domain=task_info['domain'])
            for task_info in batch
        ])

        return results

    def align_batch(self, batch : List[Dict]) -> List[Dict]:
        return asyncio.run(self._align_batch(batch))

    async def _cover_batch(self, batch : List[Dict]) -> List[Dict]:
    
        results = await asyncio.gather(*[
            self.cover(expertise_name=task_info['expertise_name'], entities=task_info['entities'])
            for task_info in batch
        ])

        return results

    def cover_batch(self, batch : List[Dict]) -> List[Dict]:
        return asyncio.run(self._cover_batch(batch))

    def align_score(self, row : pd.Series) -> float | None:

        scores = []

        for domain, score in zip(row['domain_id'], row['score']):

            if domain in row['dst']:
                scores.append(score)

        return sum(scores) / len(scores) if len(scores) > 0 else None

    def compute_scores(self) -> pd.DataFrame:

        df = pd.read_json(self.reader.filename, lines=True)

        ### Expertise 2 Domain
        expertise2subdomain = pd.DataFrame([
            to_dict(e) 
            for e in self.graph._edges[EdgeType(src="Expertise", dst="SubDomain", name="BELONGSTO")]._edges
        ])
        expertise2subdomain = expertise2subdomain[['src', 'dst']]
        expertise2subdomain = expertise2subdomain.groupby('src').agg(list).reset_index()

        ### Align Score
        df_align = df[df['task'] == 'align']
        df_align['score'] = df_align['result'].apply(lambda x : x['result']['score'])
        df_align = df_align[['expertise_id', 'domain_id', 'score']]
        df_align = df_align.groupby(by='expertise_id').agg(list).reset_index()
        df_align = df_align.merge(right=expertise2subdomain, left_on='expertise_id', right_on='src')
        df_align['DAS'] = df_align.apply(self.align_score, axis=1)
        df_align = df_align[['expertise_id', 'DAS']]

        ### Cover Score
        df_cover = df[df['task'] == 'cover']
        df_cover['CCS'] = df_cover['result'].apply(lambda x : x['result']['score'])
        df_cover = df_cover[['expertise_id', 'CCS']]

        ### CWAS
        df_scores = df_align.merge(df_cover, right_on='expertise_id', left_on='expertise_id')

        CCS = df_scores['CCS'].to_numpy() 
        DAS = df_scores['DAS'].to_numpy() 
        df_scores['CWAS'] = np.sqrt(CCS * DAS)

        return df_scores

    def update_graph(self, data : pd.DataFrame) -> None:

        for idx, row in data.iterrows():

            self.graph.add_node(
                type='Expertise',
                id=row['expertise_id'],
                properties={
                    "DAS" : row['DAS'],
                    "CCS" : row['CCS'],
                    "CWAS" : row['CWAS']
                }
            )

    def __iter__(self):

        df = self.dataframe()
        domains = list(map(to_dict, self.graph._nodes['SubDomain']._nodes))

        cache = {
            doc['task_id'] : "error" in doc['result'] 
            for doc, _ in self.reader.generate() 
        }

        ### Alignment Score
        yield self.make_info_event(
            progress=0.0,
            message="Starting Alignment Score calculation"
        )

        tasks = Batched(
            Product(*[
                df.to_dict(orient='records'),
                domains
            ]),
            self.batch_size
        )

        for batch in self.track(
            it=tasks,
            task="alignment-score"
        ):
            
            batch = [
                {
                    "task_id" : f'{expertise['id_expertise']}_{domain['id']}_align',
                    "expertise_id" : expertise['id_expertise'],
                    "expertise_name" : expertise['llm_label_expertise'],
                    "domain" : domain['name'],
                    "domain_id" : domain['id'],
                    "entities" : expertise['name_entity'],
                    "task" : "align"
                }
                for expertise, domain in batch
            ]

            batch = [task for task in batch if task['task_id'] not in cache or cache[task['task_id']]]

            if len(batch) == 0:
                continue

            results = self.align_batch(batch)

            for task, result in zip(batch, results):
                task['result'] = result

            self.writer.write(batch)

        ### Cover Score
        yield self.make_info_event(
            progress=0.5,
            message="Starting Cover Score calculation"
        )

        tasks = Batched(
            df.to_dict(orient='records'),
            self.batch_size
        )

        for batch in self.track(
            it=tasks,
            task="cover-score"
        ):
            
            batch = [
                {
                    "task_id" : f'{expertise['id_expertise']}_cover',
                    "expertise_id" : expertise['id_expertise'],
                    "expertise_name" : expertise['llm_label_expertise'],
                    "entities" : expertise['name_entity'],
                    "task" : "cover"
                }
                for expertise in batch
            ]

            batch = [task for task in batch if task['task_id'] not in cache or cache[task['task_id']]]
            
            if len(batch) == 0:
                continue

            results = self.cover_batch(batch)

            for task, result in zip(batch, results):
                task['result'] = result

            self.writer.write(batch)

        ### Compute Scores
        yield self.make_info_event(
            progress=0.8,
            message="Computing Scores"
        )

        scores_df = self.compute_scores()

        yield self.make_info_event(
            progress=0.9,
            message="Updating Graph"
        )

        ### Update Graph
        self.update_graph(scores_df)
        self.graph.save(self.graph_dir)

        yield self.make_info_event(
            progress=1.0,
            message="Graph Update Completed"
        )

        yield from []

    @classmethod
    def from_config(cls, config: Config) -> 'ExpertiseMapEvaluator':

        evaluator = ExpertiseMapEvaluator(
            graph=config.graph,
            llm=env.llms[config.llm],
            batch_size=config.batch_size,
            writer=config.file_writer,
            reader=config.file_reader,
            graph_dir=config.graph_dir
        )

        evaluator.add_handler(ConsoleLogger("ExpertiseMapEvaluator"))
        return evaluator