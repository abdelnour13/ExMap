import os
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from src.shared.file_tools import FileReader
from src.shared.command_tools import BaseCommand
from src.shared.handlers import ConsoleLogger
from definitions import DATA_DIR, hf_model_dir
from typing import Tuple, Self
from pydantic import BaseModel
from instance import cli

class Config(BaseModel):
    corpus : str = 'ls2n-duke'
    model_name : str = 'sentence-transformers/all-MiniLM-L6-v2'
    batch_size : int = 128
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        
        # edges = gnn_utils.coalesce(edges, sort_by_row = False)

        return SparseTensor(
            row = edges[:,0],
            col = edges[:,1],
            sparse_sizes = num_nodes
        )

    def __iter__(self):

        data_path = os.path.join(DATA_DIR, self.config.corpus, "clean_triples", "data.jsonl")

        yield self.make_info_event(
            progress=None,
            message=f"Loading data file : {data_path}"
        )

        reader = FileReader.from_filename(data_path)

        ## Construct adjency matrices :
        entities = {}
        articles = {}
        authors = {}
        keywords  = {}
        predicates = {}
        labels = {}
        triples = {}

        article_author = set()
        article_keyword  = set()
        entity_entity  = []
        article_label = set()
        triple_article = set()

        articles_abstracts = []
        entities_names = []
        edge_type = []

        for record, _ in reader:
            
            ### Head & Tail entities
            if record['head'] not in entities:
                entities[record['head']] = len(entities)
                entities_names.append(record['head'])

            if record['tail'] not in entities:
                entities[record['tail']] = len(entities)
                entities_names.append(record['tail'])

            ### Article
            article_id = record['sentence']['document']["id"]

            if article_id not in articles:
                articles[article_id] = len(articles)
                articles_abstracts.append(record['sentence']['document']['abstract'])

            ### Keywords
            for keyword in record['sentence']['document']["keywords"]:

                if keyword not in keywords:
                    keywords[keyword] = len(keywords)

                article_keyword.add((articles[article_id], keywords[keyword]))

            ### Authors
            for author in record['sentence']['document']["authors"]:

                if author not in authors:
                    authors[author] = len(authors)

                article_author.add((articles[article_id], authors[author]))

            ### Authors
            for label in record['sentence']['document']["labels"]:

                if label not in labels:
                    labels[label] = len(labels)

                article_label.add((articles[article_id], labels[label]))

            ### Predicate
            if record['predicate'] not in predicates:
                predicates[record['predicate']] = len(predicates)

            ### Triple
            t = (record['head'], record['predicate'], record['tail'])

            if t not in triples:
                triples[t] = len(triples)
                edge_type.append(predicates[record['predicate']])
                entity_entity.append((entities[record['head']], entities[record['tail']]))

            triple_article.add((triples[t], articles[article_id]))

            self.make_debug_event(
                progress=None,
                message=(
                    f"Found {len(entities)} entities, {len(articles)} articles, {len(authors)} authors,",
                    f"{len(keywords)} keywords and {len(predicates)} predicates."
                )
            )

        ### Feature Extraction
        self.make_info_event(
            progress=None,
            message=f"Loading model {self.config.model_name} for feature extraction..."
        )

        model = SentenceTransformer(hf_model_dir(self.config.model_name), self.config.device)

        self.make_info_event(
            progress=None,
            message=f"Starting feature extraction..."
        )

        data = dict()

        ### Features
        data['entities'] = model.encode(
            sentences = entities_names,
            batch_size = self.config.batch_size,
            device = self.config.device,
            show_progress_bar = True,
            convert_to_tensor = True,
            normalize_embeddings = True
        ).cpu()

        data['articles'] = model.encode(
            sentences = articles_abstracts,
            batch_size = self.config.batch_size,
            device = self.config.device,
            show_progress_bar = True,
            convert_to_tensor = True,
            normalize_embeddings = True
        ).cpu()

        data['predicates'] = model.encode(
            sentences = list(predicates.keys()),
            batch_size = self.config.batch_size,
            device = self.config.device,
            show_progress_bar = True,
            convert_to_tensor = True,
            normalize_embeddings = True
        ).cpu()

        ### Edge metadata
        data['edge_type'] = torch.tensor(edge_type).long()
        data['triple_article'] = torch.tensor(list(triple_article)).long()

        ### Adjacency matrices
        article_author = self.create_adjency_matrix(
            edges=torch.tensor(list(article_author)).long(),
            num_nodes=(len(articles), len(authors))
        )

        article_keyword = self.create_adjency_matrix(
            edges=torch.tensor(list(article_keyword)).long(),
            num_nodes=(len(articles), len(keywords))
        )

        data[('article', 'share-author-with', 'article')] = torch.stack(matmul(article_author, article_author.t()).coo()[:2])
        data[('article', 'share-keyword-with', 'article')] = torch.stack(matmul(article_keyword, article_keyword.t()).coo()[:2])
        data[('entity','has-relation-with', 'entity')] = torch.tensor(list(entity_entity)).long()

        ### Labels 
        article_label = self.create_adjency_matrix(
            edges=torch.tensor(list(article_label)).long(),
            num_nodes=(len(articles), len(labels))
        ).to_dense()

        data['labels'] = article_label

        ### Debug Messages
        yield self.make_debug_event(
            progress=None,
            message=f"labels.size = {data['labels'].size()}."
        )

        yield self.make_debug_event(
            progress=None,
            message=f"entities.size = {data['entities'].size()}, articles.size = {data['articles'].size()}, predicates.size = {data['predicates'].size()}."
        )

        yield self.make_debug_event(
            progress=None,
            message=f"('entity','has-relation-with', 'entity').size = {data[('entity','has-relation-with', 'entity')].size()}."
        )

        yield self.make_debug_event(
            progress=None,
            message=f"('article', 'share-author-with', 'article').size = {data[('article', 'share-author-with', 'article')].size()}."
        )

        yield self.make_debug_event(
            progress=None,
            message=f"('article', 'share-keyword-with', 'article').size = {data[('article', 'share-keyword-with', 'article')].size()}."
        )

        data['metadata'] = {
            "predicates" : list(predicates.keys()),
            "entities" : entities_names,
            "articles" : list(articles.keys()),
            "labels" : list(labels.keys())
        }

        torch.save(data, os.path.join(DATA_DIR, self.config.corpus, 'graph.pt'))



    # def __iter__(self):

    #     ### Load data :
    #     data_path = os.path.join(DATA_DIR, self.config.corpus, "clean_triples", "data.jsonl")
        
    #     yield self.make_info_event(
    #         progress=None,
    #         message=f"Loading data file : {data_path}"
    #     )

        

    #     data = load_json_file(data_path)

    #     yield self.make_info_event(
    #         progress=None,
    #         message=f"Data file loaded : {len(data)} relations to process...."
    #     )

    #     ### Construct adjency matrices :
    #     entities = {}
    #     articles = {}
    #     authors = {}
    #     venues  = {}
    #     predicates = {}
    #     labels = {}

    #     article_author = set()
    #     article_venue  = set()
    #     entity_article = set()
    #     entity_entity  = set()
    #     article_label = set()

    #     articles_abstracts = []
    #     entities_names = []
    #     predicate_ids = []

    #     for relation in data:

    #         if relation['label'] == 'APPEARS_IN':

    #             entity = relation['start']
    #             article = relation['end']

    #             if entity['properties']['id'] not in entities:
    #                 entities[entity['properties']['id']] = len(entities)
    #                 entities_names.append(entity['properties']['name'])

    #             if article['properties']['id'] not in articles:
    #                 articles[article['properties']['id']] = len(articles)
    #                 articles_abstracts.append(article['properties']['abstract'])

    #             entity_article.add((entities[entity['properties']['id']], articles[article['properties']['id']]))

    #             for author in article['properties']['authors']:

    #                 if author not in authors:
    #                     authors[author] = len(authors)

    #                 article_author.add((articles[article['properties']['id']], authors[author]))

    #             for venue in article['properties']['venues']:

    #                 if venue not in venues:
    #                     venues[venue] = len(venues)

    #                 article_venue.add((articles[article['properties']['id']], venues[venue]))     

    #             for label in article['properties']['labels']:

    #                 if label not in labels:
    #                     labels[label] = len(labels)

    #                 article_label.add((articles[article['properties']['id']], labels[label]))  

    #         elif relation['label'] == 'Predicate': 

    #             e1 = relation['start']
    #             e2 = relation['end']
    #             predicate = relation['properties']['type']

    #             if e1['properties']['id'] not in entities:
    #                 entities[e1['properties']['id']] = len(entities)
    #                 entities_names.append(e1['properties']['name'])

    #             if e2['properties']['id'] not in entities:
    #                 entities[e2['properties']['id']] = len(entities)
    #                 entities_names.append(e2['properties']['name'])

    #             if predicate not in predicates:
    #                 predicates[predicate] = len(predicates)

    #             entity_entity.add((entities[e1['properties']['id']], entities[e2['properties']['id']]))
    #             predicate_ids.append(predicates[predicate])

    #         else:
    #             raise Exception(f"Unhandled relation type : {relation['label']}.\n")     
            
    #     self.make_debug_event(
    #         progress=None,
    #         message=f"Found {len(entities)} entities, {len(articles)} articles, {len(authors)} authors, {len(venues)} venues and {len(predicates)} predicates."
    #     )

    #     ### Feature extraction
    #     self.make_info_event(
    #         progress=None,
    #         message=f"Loading model {self.config.model_name} for feature extraction..."
    #     )

    #     model = SentenceTransformer(hf_model_dir(self.config.model_name), self.config.device)

    #     self.make_info_event(
    #         progress=None,
    #         message=f"Starting feature extraction..."
    #     )

    #     data = dict()

    #     data['entities'] = model.encode(
    #         sentences = entities_names,
    #         batch_size = self.config.batch_size,
    #         device = self.config.device,
    #         show_progress_bar = True,
    #         convert_to_tensor = True,
    #         normalize_embeddings = True
    #     ).cpu()

    #     data['articles'] = model.encode(
    #         sentences = articles_abstracts,
    #         batch_size = self.config.batch_size,
    #         device = self.config.device,
    #         show_progress_bar = True,
    #         convert_to_tensor = True,
    #         normalize_embeddings = True
    #     ).cpu()

    #     data['predicates'] = model.encode(
    #         sentences = list(predicates.keys()),
    #         batch_size = self.config.batch_size,
    #         device = self.config.device,
    #         show_progress_bar = True,
    #         convert_to_tensor = True,
    #         normalize_embeddings = True
    #     ).cpu()

    #     data['edge_type'] = torch.tensor(predicate_ids).long()

    #     data[('entity','appears-in', 'article')] = torch.tensor(list(entity_article)).long()
    #     data[('entity','has-relation-with', 'entity')] = torch.tensor(list(entity_entity)).long()
        
    #     article_author = self.create_adjency_matrix(
    #         edges=torch.tensor(list(article_author)).long(),
    #         num_nodes=(len(articles), len(authors))
    #     )

    #     article_venue = self.create_adjency_matrix(
    #         edges=torch.tensor(list(article_venue)).long(),
    #         num_nodes=(len(articles), len(venues))
    #     )

    #     data[('article', 'share-author-with', 'article')] = torch.stack(matmul(article_author, article_author.t()).coo()[:2])
    #     data[('article', 'share-venue-with', 'article')] = torch.stack(matmul(article_venue, article_venue.t()).coo()[:2])

    #     ### Labels 
    #     article_label = self.create_adjency_matrix(
    #         edges=torch.tensor(list(article_label)).long(),
    #         num_nodes=(len(articles), len(labels))
    #     ).to_dense()

    #     data['labels'] = article_label

    #     ### Debug Messages
    #     yield self.make_debug_event(
    #         progress=None,
    #         message=f"labels.size = {data['labels'].size()}."
    #     )

    #     yield self.make_debug_event(
    #         progress=None,
    #         message=f"entities.size = {data['entities'].size()}, articles.size = {data['articles'].size()}, predicates.size = {data['predicates'].size()}."
    #     )

    #     yield self.make_debug_event(
    #         progress=None,
    #         message=f"('entity','appears-in', 'article').size = {data[('entity','appears-in', 'article')].size()}."
    #     )

    #     yield self.make_debug_event(
    #         progress=None,
    #         message=f"('entity','has-relation-with', 'entity').size = {data[('entity','has-relation-with', 'entity')].size()}."
    #     )

    #     yield self.make_debug_event(
    #         progress=None,
    #         message=f"('article', 'share-author-with', 'article').size = {data[('article', 'share-author-with', 'article')].size()}."
    #     )

    #     yield self.make_debug_event(
    #         progress=None,
    #         message=f"('article', 'share-venue-with', 'article').size = {data[('article', 'share-venue-with', 'article')].size()}."
    #     )

    #     data['metadata'] = {
    #         "predicates" : list(predicates.keys()),
    #         "entities" : entities_names,
    #         "articles" : list(articles.keys())
    #     }

    #     torch.save(data, os.path.join(DATA_DIR, self.config.corpus, 'graph.pt'))

    @classmethod
    def from_config(cls, config : Config) -> Self:

        downloader = FeatureExtractor(config=config)
        
        downloader \
            .add_handler(ConsoleLogger("DataDownloader"))

        return downloader