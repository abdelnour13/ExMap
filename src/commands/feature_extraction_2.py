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
from typing import Tuple, Self
from pydantic import BaseModel
from instance import cli
from collections import defaultdict

class Config(BaseModel):
    corpus : str = 'ls2n-duke'
    model_name : str = 'intfloat/e5-base'
    batch_size : int = 128
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu'

def hashabledict(key : str, d : dict) -> dict:

    class _hashabledict(dict):

        def __eq__(self, value):
            return self[key] == value[key]
        
        def __hash__(self):
            return hash(self[key])
    
    return _hashabledict(d)

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

    def __iter__(self):

        data_path = os.path.join(DATA_DIR, self.config.corpus, "entities_mapping", "data.jsonl")

        yield self.make_info_event(
            progress=None,
            message=f"Loading data file : {data_path}"
        )

        reader = FileReader.from_filename(data_path)

        nodes = defaultdict(set)
        relations = defaultdict(set)
        predicates = {}

        for record, _ in reader:

            document = record['sentence']['document']
            
            nodes['Article'].add(hashabledict('id', {
                "id" : document['id'],
                "title" : document['title'],
                "abstract" : document['title'],
                "date" : document['date'],
                "type" : "Article"
            }))

            for author in document['authors']:

                author_id = author.replace(' ','_')

                nodes['Author'].add(hashabledict('id', {
                    "id" : author_id,
                    "name" : author,
                    "type" : "Author"
                }))

                relations['Article_WrittenBy_Author'].add(hashabledict('id', {
                    "id"  : f"{document['id']}-{author_id}",
                    "src" : document['id'],
                    "dst" : author_id,
                }))

            for venue in document['venues']:

                venue_id = venue

                nodes['Venue'].add(hashabledict('id', {
                    "id" : venue_id,
                    "name" : venue,
                    "type" : "Venue"
                }))

                relations['Article_BelongsTo_Venue'].add(hashabledict('id', {
                    "id"  : f"{document['id']}-{venue_id}",
                    "src" : document['id'],
                    "dst" : venue_id
                }))

            for team in document['teams']:

                lab_id = team

                nodes['Lab'].add(hashabledict('id', {
                    "id" : lab_id,
                    "name" : team,
                    "type" : "Lab"
                }))

                relations['Article_BelongsTo_Lab'].add(hashabledict('id', {
                    "id"  : f"{document['id']}-{lab_id}",
                    "src" : document['id'],
                    "dst" : lab_id
                }))

            for label in document['labels']:

                domain_id = label.replace('', '_')

                nodes['SubDomain'].add(hashabledict('id', {
                    "id" : domain_id,
                    "name" : label,
                    "type" : "SubDomain"
                }))

                relations['Article_BelongsTo_SubDomain'].add(hashabledict('id', {
                    "id"  : f"{document['id']}-{domain_id}",
                    "src" : document['id'],
                    "dst" : domain_id
                }))

            for entity in [record['head'], record['tail']]:

                entity_id = entity.replace(' ','_')

                nodes['Entity'].add(hashabledict('id', {
                    "id" : entity_id,
                    "name" : entity,
                    "type" : "Entity"
                }))

                relations['Entity_AppearsIn_Article'].add(hashabledict('id', {
                    "id"  : f"{entity_id}-{document['id']}",
                    "src" : entity_id,
                    "dst" : document['id']
                }))

            head_id = record['head'].replace(' ','_')
            tail_id = record['tail'].replace(' ','_')

            if record['predicate'] not in predicates:
                predicates[record['predicate']] = len(predicates)

            relations['Entity_HasRelationWith_Entity'].add(hashabledict('id', {
                "id" : f"{head_id}-{record['predicate'].replace(' ','_')}-{tail_id}",
                "src" : head_id,
                "dst" : tail_id,
                "properties" : {
                    "predicate" : record['predicate'],
                    "sentence" : record['sentence']
                }
            }))
        
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

        model = SentenceTransformer(hf_model_dir(self.config.model_name), self.config.device)

        G = {
            "nodes" : defaultdict(dict),
            "relations" : defaultdict(dict),
        }

        G['nodes']['Entity'] = {
            "id": node2id['Entity'],
            "embeddings": model.encode(
                sentences = [f'query: {e['name']}' for e in nodes['Entity']],
                batch_size = self.config.batch_size,
                device = self.config.device,
                show_progress_bar = True,
                convert_to_tensor = True,
                normalize_embeddings = True
            ).cpu()
        }

        G['nodes']['Article']['query'] = {
            "id": node2id['Article'],
            "embeddings" : model.encode(
                sentences = [f'query: {e['abstract']}' for e in nodes['Article']],
                batch_size = self.config.batch_size,
                device = self.config.device,
                show_progress_bar = True,
                convert_to_tensor = True,
                normalize_embeddings = True
            ).cpu()
        }

        G['nodes']['Article']['passage'] = {
            "id": node2id['Article'],
            "embeddings" : model.encode(
                sentences = [f'passage: {e['abstract']}' for e in nodes['Article']],
                batch_size = self.config.batch_size,
                device = self.config.device,
                show_progress_bar = True,
                convert_to_tensor = True,
                normalize_embeddings = True
            ).cpu()
        }

        G['predicates'] = {
            "id": list(predicates.values()),
            "predicates": model.encode(
                sentences = list(predicates.keys()),
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

        ### Save to disk
        GRAPHS_DIR = os.path.join(DATA_DIR, self.config.corpus, 'graphs')
        graph_id = self.config.model_name.split('/')[-1]
        GRAPH_DIR  = os.path.join(GRAPHS_DIR, graph_id)

        os.makedirs(GRAPH_DIR)

        EMBEDDING_FILENAME = os.path.join(GRAPH_DIR, 'graph.pt')
        ENTITIES_FILENAME = os.path.join(GRAPH_DIR, 'nodes.json')
        RELATIONS_FILENAME = os.path.join(GRAPH_DIR, 'relations.json')

        helpers.save_json_file(ENTITIES_FILENAME, nodes)
        helpers.save_json_file(RELATIONS_FILENAME, relations)
        torch.save(G, EMBEDDING_FILENAME)

    @classmethod
    def from_config(cls, config : Config) -> Self:

        downloader = FeatureExtractor(config=config)
        
        downloader \
            .add_handler(ConsoleLogger("FeatureExtraction"))

        return downloader