import os
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch_sparse import SparseTensor
from src.shared.file_tools import FileReader
from src.shared.command_tools import BaseCommand
from src.shared.handlers import ConsoleLogger
from src import helpers
from definitions import DATA_DIR, hf_model_dir
from typing import Tuple, Self, Dict, List
from pydantic import BaseModel
from instance import cli
from collections import defaultdict
from functools import partial
from rich.progress import Progress
from dataclasses import dataclass

class Config(BaseModel):
    corpus : str = 'ls2n-duke'
    model_name : str = 'sentence-transformers/all-MiniLM-L6-v2'
    batch_size : int = 128
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu'

def hashabledict(d : dict, key : str) -> dict:

    class _hashabledict(dict):

        def __eq__(self, value):
            return self[key] == value[key]
        
        def __hash__(self):
            return hash(self[key])
    
    return _hashabledict(d)

@dataclass
class Graph:
    nodes : Dict[str, List[Dict]]
    relations : Dict[str, List[Dict]]
    predicates : Dict[str, int]
    node2id : Dict[str, List[str]]
    id2incid : Dict[str, Dict[str, int]]

@cli.register_stage(
    name="feature_extraction",
    description=(
        "Transofrms the graph nodes (entities & articles) into a tensor representation ready to be fed to "
        "gnn models using the provided `model_name` to map text to vectors."
    ),
    add_from_conf=False,
    add_save_to=False
)
class FeatureExtractor(BaseCommand):

    def __init__(self, config : Config) -> None:
        super().__init__()

        self.config = config

    @staticmethod
    def create_adjency_matrix(
        edges : Tensor,
        num_nodes : Tuple[int, int]
    ) -> SparseTensor:
        
        return SparseTensor(
            row = edges[:,0],
            col = edges[:,1],
            sparse_sizes = num_nodes
        )
    
    def process_triples(self) -> Graph:

        data_path = os.path.join(DATA_DIR, self.config.corpus, "entities_mapping", "data.jsonl")

        self.send_info_event(
            progress=None,
            message=f"Loading data file : {data_path}"
        )

        reader = FileReader.from_filename(data_path)

        nodes = defaultdict(set)
        relations = defaultdict(set)
        predicates = {}

        Record = partial(hashabledict, key='id')

        pbar = Progress()
        task1 = pbar.add_task("Triples Processing", total=100)
        pbar.start()

        previous = 0.0
        
        for record, prog in reader:

            document = record['sentence']['document']
            
            nodes['Article'].add(Record({
                "id" : document['id'],
                "title" : document['title'],
                "abstract" : document['abstract'],
                "date" : document['date'],
                "type" : "Article"
            }))

            for author in document['authors']:

                author_id = author.replace(' ','_')

                nodes['Author'].add(Record({
                    "id" : author_id,
                    "name" : author,
                    "type" : "Author"
                }))

                relations['Article_WrittenBy_Author'].add(Record({
                    "id"  : f"{document['id']}-{author_id}",
                    "src" : document['id'],
                    "dst" : author_id,
                }))

            for venue in document['venues']:

                venue_id = venue

                nodes['Venue'].add(Record({
                    "id" : venue_id,
                    "name" : venue,
                    "type" : "Venue"
                }))

                relations['Article_BelongsTo_Venue'].add(Record({
                    "id"  : f"{document['id']}-{venue_id}",
                    "src" : document['id'],
                    "dst" : venue_id
                }))

            for team in document['teams']:

                lab_id = team

                nodes['Lab'].add(Record({
                    "id" : lab_id,
                    "name" : team,
                    "type" : "Lab"
                }))

                relations['Article_BelongsTo_Lab'].add(Record({
                    "id"  : f"{document['id']}-{lab_id}",
                    "src" : document['id'],
                    "dst" : lab_id
                }))

            for label in document['labels']:

                domain_id = label.replace('', '_')

                nodes['SubDomain'].add(Record({
                    "id" : domain_id,
                    "name" : label,
                    "type" : "SubDomain"
                }))

                relations['Article_BelongsTo_SubDomain'].add(Record({
                    "id"  : f"{document['id']}-{domain_id}",
                    "src" : document['id'],
                    "dst" : domain_id
                }))

            for keyword in document['keywords']:

                keyword_id = keyword.replace('', '_')

                nodes['Keyword'].add(Record({
                    "id" : keyword_id,
                    "name" : keyword,
                    "type" : "Keyword"
                }))

                relations['Article_HasKeyword_Keyword'].add(Record({
                    "id"  : f"{document['id']}-{keyword_id}",
                    "src" : document['id'],
                    "dst" : keyword_id
                }))

            for entity in [record['head'], record['tail']]:

                entity_id = entity.replace(' ','_')

                nodes['Entity'].add(Record({
                    "id" : entity_id,
                    "name" : entity,
                    "type" : "Entity"
                }))

                relations['Entity_AppearsIn_Article'].add(Record({
                    "id"  : f"{entity_id}-{document['id']}",
                    "src" : entity_id,
                    "dst" : document['id']
                }))

            head_id = record['head'].replace(' ','_')
            tail_id = record['tail'].replace(' ','_')

            if record['predicate'] not in predicates:
                predicates[record['predicate']] = len(predicates)

            relations['Entity_HasRelationWith_Entity'].add(Record({
                "id" : f"{head_id}-{record['predicate'].replace(' ','_')}-{tail_id}",
                "src" : head_id,
                "dst" : tail_id,
                "properties" : {
                    "predicate" : record['predicate'],
                    "sentence" : record['sentence']
                }
            }))
            
            pbar.update(task1, advance=100*prog-previous, refresh=True, visible=True)
            previous = 100 * prog

        for node_type, nodes_list in nodes.items():

            self.send_info_event(
                progress=None,
                message=f"Found {len(nodes_list)} nodes of type : {node_type}."
            )

        for rel_type, rel_list in relations.items():

            self.send_info_event(
                progress=None,
                message=f"Found {len(rel_list)} edges of type : {rel_type}."
            )
        
        ### Feature Extraction
        nodes = {
            key : list(value)
            for key, value in nodes.items()
        }

        relations = {
            key : list(value)
            for key, value in relations.items()
        }

        node2id = {
            key : [item['id'] for item in value]
            for key, value in nodes.items()
        }

        id2incid = {
            key : { item : i for i, item in enumerate(value) }
            for key, value in node2id.items()
        }

        return Graph(
            nodes=nodes,
            relations=relations,
            predicates=predicates,
            node2id=node2id,
            id2incid=id2incid
        )
    
    def extract_features(self, graph : Graph) -> Dict:

        nodes = graph.nodes
        node2id = graph.node2id
        predicates = graph.predicates
        relations = graph.relations
        id2incid = graph.id2incid

        ### Load model locally
        model = SentenceTransformer(
            hf_model_dir(self.config.model_name), 
            self.config.device
        )

        G = {
            "nodes" : defaultdict(dict),
            "relations" : defaultdict(dict),
        }

        entities = (
            [f"query: {e['name']}" for e in nodes['Entity']] 
            if self.config.model_name == 'intfloat/e5-base-v2' else 
            [e['name'] for e in nodes['Entity']] 
        )

        articles = (
            [f"query: {e['abstract']}" for e in nodes['Article']] 
            if self.config.model_name == 'intfloat/e5-base-v2' else 
            [e['abstract'] for e in nodes['Article']] 
        )

        predicate_list = (
            [f'query: {predicate}' for predicate in predicates.keys()] 
            if self.config.model_name == 'intfloat/e5-base-v2' else 
            list(predicates.keys())
        )

        G['nodes']['Entity'] = {
            "id": node2id['Entity'],
            "embeddings": model.encode(
                sentences = entities,
                batch_size = self.config.batch_size,
                device = self.config.device,
                show_progress_bar = True,
                convert_to_tensor = True,
                normalize_embeddings = True
            ).cpu(),
        }

        G['nodes']['Article'] = {
            "id": node2id['Article'],
            "embeddings" : model.encode(
                sentences = articles,
                batch_size = self.config.batch_size,
                device = self.config.device,
                show_progress_bar = True,
                convert_to_tensor = True,
                normalize_embeddings = True
            ).cpu()
        }

        passage_embeddings = model.encode(
            sentences = [f"passage: {e['abstract']}" for e in nodes['Article']],
            batch_size = self.config.batch_size,
            device = self.config.device,
            show_progress_bar = True,
            convert_to_tensor = True,
            normalize_embeddings = True
        ).cpu() if self.config.model_name == 'intfloat/e5-base-v2' else G['nodes']['Article']['embeddings']
        

        G['predicates'] = {
            "id": [f'query: {predicate}' for predicate in predicates.values()],
            "predicates": model.encode(
                sentences = predicate_list,
                batch_size = self.config.batch_size,
                device = self.config.device,
                show_progress_bar = True,
                convert_to_tensor = True,
                normalize_embeddings = True
            ).cpu()
        }

        for relation_name, rels in relations.items():

            src_node, rel_name, dst_node = relation_name.split('_')

            G['relations'][(src_node,rel_name,dst_node)] = []

            for r in rels:
                    
                if 'properties' not in r:

                    G['relations'][(src_node,rel_name,dst_node)].append([
                        id2incid[src_node][r['src']],
                        id2incid[dst_node][r['dst']],
                    ])

                else:

                    G['relations'][(src_node,rel_name,dst_node)].append([
                        id2incid[src_node][r['src']],
                        id2incid[dst_node][r['dst']],
                        predicates[r['properties']['predicate']]
                    ])     

            G['relations'][(src_node,rel_name,dst_node)] = torch.tensor(G['relations'][(src_node,rel_name,dst_node)])  

        ### Generality Scores
        pbar = Progress()
        task2 = pbar.add_task("Computing Generality Score", total=len(nodes['Entity']))
        task3 = pbar.add_task("Computing Relevancy Score", total=len(nodes['Entity']))
        pbar.start()

        scores = []

        for entity_idx in range(len(nodes['Entity'])):
            entities_ids, articles_ids = G['relations'][('Entity','AppearsIn','Article')].t()
            articles_ids = torch.unique(articles_ids[entities_ids == entity_idx])
            articles_embeddings = G['nodes']['Article']['embeddings'][articles_ids]
            score = 1 - (articles_embeddings @ articles_embeddings.T).mean().item()
            scores.append(score)
            pbar.update(task_id=task2, advance=1, refresh=True)

        scores = torch.tensor(scores)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        G['nodes']['Entity']['generality_score'] = scores

        ### Relevancy Scores
        scores = []

        for entity_idx, entity_embedding in enumerate(G['nodes']['Entity']['embeddings']):
            entities_ids, articles_ids = G['relations'][('Entity','AppearsIn','Article')].t()
            articles_ids = torch.unique(articles_ids[entities_ids == entity_idx])
            articles_embeddings = passage_embeddings[articles_ids]
            score = 1 - (entity_embedding.reshape(1, -1) @ articles_embeddings.T).mean().item()
            scores.append(score)
            pbar.update(task_id=task3, advance=1, refresh=True)

        scores = torch.tensor(scores)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        G['nodes']['Entity']['relevancy_score'] = scores

        return G


    def __iter__(self):

        graph = self.process_triples()
        G = self.extract_features(graph)        

        ### Save to disk
        GRAPHS_DIR = os.path.join(DATA_DIR, self.config.corpus, 'graphs')
        graph_id = self.config.model_name.split('/')[-1]
        GRAPH_DIR  = os.path.join(GRAPHS_DIR, graph_id)

        os.makedirs(GRAPH_DIR)

        EMBEDDING_FILENAME = os.path.join(GRAPH_DIR, 'graph.pt')
        ENTITIES_FILENAME = os.path.join(GRAPH_DIR, 'nodes.json')
        RELATIONS_FILENAME = os.path.join(GRAPH_DIR, 'relations.json')

        helpers.save_json_file(ENTITIES_FILENAME, graph.nodes)
        helpers.save_json_file(RELATIONS_FILENAME, graph.relations)
        torch.save(G, EMBEDDING_FILENAME)

        yield from []

    @classmethod
    def from_config(cls, config : Config) -> Self:

        downloader = FeatureExtractor(config=config)
        
        downloader \
            .add_handler(ConsoleLogger("FeatureExtraction"))

        return downloader