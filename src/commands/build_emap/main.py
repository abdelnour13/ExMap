import os
import numpy as np
import torch
import math
from torch import Tensor
from src import helpers
from src.commands.feature_extraction.graph_builder import EdgeTypes, NodeTypes
from src.commands.graph_learning.main import (
    Config as TrainingConfig,
    HeteroData, Graph, TopicModel,
    Context
)
from src.commands.build_emap.clustering import (
    CommunityDetectionBasedClustering,
    COMMUNITY_DETECTION_ALG,
    MODULARITY,
    GRAPH_TYPE
)
from src.commands.build_emap.representative_nodes_selection import select_representative_nodes
from src.helpers.torch_utils import TensorDict
from src.shared.handlers import ConsoleLogger, Event
from src.shared.command_tools import BaseCommand
from pydantic import BaseModel, computed_field
from typing import Tuple, Dict, Any, List, TypedDict, Callable, Generator
from definitions import EMS_DIR
from src.graph_dl.layers.reduction import reduce
from src.commands.build_emap.scores import score, SCORING_METHOD
from src.shared.file_tools import Batched, Map
from src.shared.platforms import LLM, RequestReturnType
from returns.pipeline import is_successful


class CluseringConfig(BaseModel):
    alg : COMMUNITY_DETECTION_ALG
    graph_type : GRAPH_TYPE
    threshold_or_num_neighbors: int | float
    modularity: MODULARITY
    resolution: float
    weighted: bool
    min_cluster_size : int


class EMCConfig(BaseModel):

    ### Graph & Model
    expertise_id : str
    model_id : str

    ### Entities Filtering
    scoring_method : SCORING_METHOD
    scoring_threshold : float
    scoring_kwargs : Dict[str, Any]

    ### Clustering Parameters
    clustering_config : CluseringConfig

    ### Representative nodes selection
    max_representative_nodes : int

    ### Expertise mining
    min_articles_per_expertise : int

    ## LLM
    llm : str
    llm_batch_size : int

    @property
    def expriment(self) -> str:
        return '/'.join(self.model_id.split('/')[:-1])

    @property
    def model_name(self) -> str:
        return self.model_id.split('/')[-1]

    @property
    def graph_dir(self) -> str:
        return os.path.join(EMS_DIR, self.expertise_id)

class Config(BaseModel):

    expertise_id : str

    @computed_field
    @property
    def conf(self) -> EMCConfig:

        return EMCConfig(**{
            'expertise_id' : self.expertise_id,
            **helpers.load_json_file(os.path.join(EMS_DIR, self.expertise_id, 'config.json'))
        })


class EMCContext(TypedDict, total=False):
    """
    Shared state of the ExpertiseMapConstructor pipeline.
    Filled progressively by the stages, in this order.
    """
    data : HeteroData           # load_data
    graph : Graph               # load_data
    results : Dict[str, Any]    # load_results
    scores : Tensor             # compute_scores
    entities : TensorDict       # remove_noisy_entities -> enriched by later stages
    centroids : Tensor          # compute_centroids
    topic_article : Tensor      # mine_expertises : [2, *] (topic_id, article_idx)
    topic_entity : Tensor       # mine_expertises : [2, *] (topic_id, entity_idx)
    expertise_ids : Tensor      # mine_expertises : [E]
    expertise_pairs : Tensor    # mine_expertises : [2, E] (topic_id_a, topic_id_b), a < b
    expertise_num_articles : Tensor  # mine_expertises : [E] articles supporting each pair
    expertise_article : Tensor  # mine_expertises : [2, *] (expertise_id, article_idx)
    expertise_topic : Tensor    # mine_expertises : [2, *] (expertise_id, topic_id)
    topics : List[Dict]         # name_topics
    topic_names : Dict[int, str]     # name_topics : topic_id -> name
    expertise_names : List[str]      # name_expertises : expertise_id -> "topic_a x topic_b"


Stage = Callable[[int], Generator[Event, None, None]]


def pairs_common_articles(
    pairs : Tensor,
    node_article : Tensor,
    num_nodes : int,
    num_articles : int
) -> Tensor:
    """
    For each pair p = (i, j) (column p of `pairs`), returns the articles linked to
    both nodes i and j, as an edge index [2, *] of (pair_id, article_idx).
    `node_article` ([2, *] of (node_idx, article_idx)) must be deduplicated.

    Vectorized : for each pair, the articles of the node with the smaller degree are
    enumerated, then looked up in the (node, article) set of the other node.
    """
    device = node_article.device
    num_pairs = pairs.size(1)

    if num_pairs == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    ### Sorted (node, article) keys + CSR pointers per node
    keys = node_article[0] * num_articles + node_article[1]
    keys, order = torch.sort(keys)
    sorted_articles = node_article[1][order]

    deg = torch.bincount(node_article[0], minlength=num_nodes)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(deg, dim=0)])

    ### Enumerate articles of the lowest-degree endpoint of each pair
    i, j = pairs
    swap = deg[i] > deg[j]
    small = torch.where(swap, j, i)
    other = torch.where(swap, i, j)

    n = deg[small]
    pair_ids = torch.arange(num_pairs, device=device)
    pair_rep = torch.repeat_interleave(pair_ids, n)
    offsets = torch.arange(n.sum(), device=device) - torch.repeat_interleave(torch.cumsum(n, dim=0) - n, n)
    articles = sorted_articles[torch.repeat_interleave(ptr[small], n) + offsets]

    ### Keep those also linked to the other endpoint
    query = other[pair_rep] * num_articles + articles
    pos = torch.searchsorted(keys, query).clamp(max=keys.numel() - 1)
    found = keys[pos] == query

    return torch.stack([pair_rep[found], articles[found]])

def invalidate_graph_caches(graph : Graph) -> None:
    """Drops the cached_property values of a Graph and its stores after it has been modified."""

    for attr in ('pygeo', 'raw_data', 'degrees'):
        graph.__dict__.pop(attr, None)

    for store in graph._nodes.values():
        store.__dict__.pop('nodes', None)

    for store in graph._edges.values():
        for attr in ('source_edges', 'target_edges', 'edges'):
            store.__dict__.pop(attr, None)

def minmax(s) -> Tuple[float | int, float | int]:
    return min(s), max(s)

def minmax_norm(v : float, min : float, max : float, base : float = 0.0) -> float:
    return (v - min + base) / (max - min + base)


class ExpertiseMapConstructor(BaseCommand):

    ### Key of the topic name in the 'ClusterTopicNaming' LLM output
    TOPIC_NAME_KEY : str = 'topic'

    def __init__(self, config : EMCConfig, llm : LLM) -> None:

        super().__init__()

        self.config = config
        self.train_config = TrainingConfig(experiment=config.expriment, configs=None)
        self.experiment_config = self.train_config.training_config.configs[config.model_name]
        self.llm = llm

        self._context : EMCContext = {}
        self._num_stages : int = 1

    ############################################################
    ### Pipeline
    ############################################################

    def get_stages(self) -> List[Stage]:
        return [
            self.load_data,
            self.load_results,
            self.compute_scores,
            self.remove_noisy_entities,
            self.cluster,
            self.remove_clustering_noise,
            self.compute_centroids,
            self.compute_distances_to_centroids,
            self.mine_expertises,
            self.name_topics,
            self.name_expertises,
            self.update_graph,
        ]

    def _progress(self, stage_id : int, fraction : float = 0.0) -> float:
        """Main progress at `fraction` of the way through stage `stage_id`."""
        return (stage_id + fraction) / self._num_stages

    def __iter__(self):

        stages = self.get_stages()
        self._num_stages = len(stages)

        for stage_id, stage in enumerate(stages):
            yield from stage(stage_id)

        topics = self._context.get('topics', [])
        expertises = self._context.get('expertise_names', [])
        yield self.make_info_event(
            progress=1.0,
            message=f"Expertise map constructed : {len(topics)} topics, {len(expertises)} expertises."
        )

    ############################################################
    ### Stages
    ############################################################

    def load_data(self, stage_id : int) -> Generator[Event, None, None]:

        yield self.make_info_event(progress=self._progress(stage_id), message="Loading data...")

        data, graph = self.experiment_config.load_data(return_raw=True)
        self._context['data'] = data
        self._context['graph'] = graph

        yield self.make_info_event(progress=self._progress(stage_id, 1.0), message="Data loaded.")

    def load_results(self, stage_id : int) -> Generator[Event, None, None]:

        yield self.make_info_event(progress=self._progress(stage_id), message="Loading results...")

        results = self.experiment_config.load_results(final_only=True)['final']
        self._context['results'] = results

        yield self.make_debug_event(
            progress=self._progress(stage_id, 0.5),
            message=f"Results keys : `{'|'.join(results.keys())}`."
        )
        yield self.make_debug_event(
            progress=self._progress(stage_id, 0.5),
            message=f"results['entities_sem_embeddings'].size() = {results['entities_sem_embeddings'].shape}"
        )
        yield self.make_info_event(progress=self._progress(stage_id, 1.0), message="Results loaded.")

    def compute_scores(self, stage_id : int) -> Generator[Event, None, None]:

        yield self.make_info_event(progress=self._progress(stage_id), message="Computing entities scores...")

        data = self._context['data']
        results = self._context['results']

        scores = score(
            scoring_method=self.config.scoring_method,
            entity_article=data[EdgeTypes.Entity_APPEARNSIN_Article.as_tuple()].edge_index,
            attn_weights=results['attn_weights'],
            num_entities=data['Entity'].num_nodes,
            num_articles=data['Article'].num_nodes,
            **self.config.scoring_kwargs
        )
        self._context['scores'] = scores

        yield self.make_debug_event(progress=self._progress(stage_id, 0.5), message=f"scores.size() = {scores.size()}")
        yield self.make_info_event(progress=self._progress(stage_id, 1.0), message="Entities scores computed.")

    def remove_noisy_entities(self, stage_id : int) -> Generator[Event, None, None]:

        yield self.make_info_event(progress=self._progress(stage_id), message="Removing noisy entities...")

        entities = self._make_entities()
        total = entities['x'].size(0)

        entities = entities[entities['scores'] >= self.config.scoring_threshold]
        self._context['entities'] = entities

        yield self.make_debug_event(
            progress=self._progress(stage_id, 1.0),
            message=f"Number of entities after noise removal = {entities['x'].size(0)}/{total}."
        )

    def cluster(self, stage_id : int) -> Generator[Event, None, None]:

        yield self.make_info_event(progress=self._progress(stage_id), message="Running clustering algorithm...")

        entities = self._context['entities']
        alg = CommunityDetectionBasedClustering(**self.config.clustering_config.model_dump())
        entities['clusters'] = alg.cluster(entities['x'])

        uniques, counts = entities['clusters'].unique(return_counts=True)
        counts = counts[uniques != -1]
        p = self._progress(stage_id, 1.0)

        yield self.make_debug_event(progress=p, message=f"Num. Topics = {torch.sum(uniques != -1)}.")
        yield self.make_debug_event(progress=p, message=f"Num. Noisy Nodes = {torch.sum(entities['clusters'] == -1)}.")
        yield self.make_debug_event(
            progress=p,
            message=f"Max(sizes) = {counts.max().item()} | "
                f"Median(sizes)  = {counts.float().median().item()} | "
                f"Mean(sizes)    = {counts.float().mean().item()}"
        )

    def remove_clustering_noise(self, stage_id : int) -> Generator[Event, None, None]:

        entities = self._context['entities']
        total = entities['x'].size(0)

        entities = entities[entities['clusters'] != -1]
        self._context['entities'] = entities

        yield self.make_debug_event(
            progress=self._progress(stage_id, 1.0),
            message=f"Entities kept after clustering noise removal = {entities['x'].size(0)}/{total}."
        )

    def compute_centroids(self, stage_id : int) -> Generator[Event, None, None]:

        yield self.make_info_event(progress=self._progress(stage_id), message="Computing centroids...")

        entities = self._context['entities']
        x, c = entities['x'], entities['clusters']

        centroids, _, _ = reduce(
            x=x,
            edge_index=torch.stack([torch.arange(x.size(0), device=x.device), c], dim=0),
            num_nodes=c.max() + 1,
            weights=entities['attn_weights'],
            kind='attn'
        ).astuple()
        
        self._context['centroids'] = centroids

        yield self.make_debug_event(progress=self._progress(stage_id, 1.0), message=f"C.size() = {centroids.size()}")

    def compute_distances_to_centroids(self, stage_id : int) -> Generator[Event, None, None]:

        yield self.make_info_event(progress=self._progress(stage_id), message="Computing distances to centroids...")

        entities = self._context['entities']
        C = self._context['centroids']
        entities['distances'] = torch.sum(entities['x'] * C[entities['clusters'], :], dim=-1)

        yield self.make_info_event(progress=self._progress(stage_id, 1.0), message="Distances computed.")

    def mine_expertises(self, stage_id : int) -> Generator[Event, None, None]:

        yield self.make_info_event(progress=self._progress(stage_id), message="Mining expertises...")

        data = self._context['data']
        entities = self._context['entities']

        num_entities = data['Entity'].num_nodes
        num_articles = data['Article'].num_nodes

        entity_article = data[EdgeTypes.Entity_APPEARNSIN_Article.as_tuple()].edge_index
        device = entity_article.device

        ### Topic of each entity in the full graph (-1 = filtered out / noise)
        entity_topic = torch.full((num_entities,), -1, dtype=torch.long, device=device)
        entity_topic[entities['indices'].to(device)] = entities['clusters'].to(device).long()

        ### Keep only (entity, article) relations of retained entities, deduplicated
        keep = entity_topic[entity_article[0]] != -1
        entity_article = torch.unique(entity_article[:, keep], dim=1)

        ### Topic - Entity
        topic_entity = torch.stack([entity_topic[entities['indices'].to(device)], entities['indices'].to(device)])

        ### Topic - Article : an article belongs to a topic if it contains at least one of its entities
        topic_article = torch.unique(
            torch.stack([entity_topic[entity_article[0]], entity_article[1]]),
            dim=1
        )

        num_topics = int(entities['clusters'].max().item()) + 1

        ### Topic pairs co-occurrence : C = T @ T^T, C[s, t] = num. articles belonging to both s and t
        T = torch.sparse_coo_tensor(
            topic_article,
            torch.ones(topic_article.size(1), device=device),
            size=(num_topics, num_articles)
        ).coalesce()

        C = torch.sparse.mm(T, T.t()).coalesce()
        s, t = C.indices()
        counts = C.values().round().long()

        ### C is symmetric : keeping s < t makes pairs unordered, (t1, t2) and (t2, t1) are the same expertise
        mask = (s < t) & (counts >= self.config.min_articles_per_expertise)
        expertise_pairs = torch.stack([s[mask], t[mask]])
        expertise_num_articles = counts[mask]
        expertise_ids = torch.arange(expertise_pairs.size(1), device=device)

        ### Expertise - Article : articles belonging to both topics of the pair
        expertise_article = pairs_common_articles(
            pairs=expertise_pairs,
            node_article=topic_article,
            num_nodes=num_topics,
            num_articles=num_articles
        )

        ### Expertise - Topic : the two topics of each pair
        expertise_topic = torch.stack([
            torch.cat([expertise_ids, expertise_ids]),
            torch.cat([expertise_pairs[0], expertise_pairs[1]])
        ])

        self._context['topic_article'] = topic_article
        self._context['topic_entity'] = topic_entity
        self._context['expertise_ids'] = expertise_ids
        self._context['expertise_pairs'] = expertise_pairs
        self._context['expertise_num_articles'] = expertise_num_articles
        self._context['expertise_article'] = expertise_article
        self._context['expertise_topic'] = expertise_topic

        ### Stats
        p = self._progress(stage_id, 1.0)
        num_expertises = expertise_ids.numel()

        yield self.make_debug_event(progress=p, message=f"Num. Expertises = {num_expertises}.")
        yield self.make_debug_event(
            progress=p,
            message=f"Num. edges : "
                f"Topic-Article = {topic_article.size(1)} | "
                f"Topic-Entity = {topic_entity.size(1)} | "
                f"Expertise-Article = {expertise_article.size(1)} | "
                f"Expertise-Topic = {expertise_topic.size(1)}"
        )

        if num_expertises > 0:
            sizes = expertise_num_articles.float()
            yield self.make_debug_event(
                progress=p,
                message=f"Expertise sizes (num. articles) : "
                    f"Max = {sizes.max().item()} | "
                    f"Mean = {sizes.mean().item():.2f} | "
                    f"Median = {sizes.median().item()}"
            )
        else:
            yield self.make_warn_event(
                progress=p,
                message=f"No expertise found with min_articles_per_expertise = {self.config.min_articles_per_expertise}."
            )

    def name_topics(self, stage_id : int) -> Generator[Event, None, None]:

        yield self.make_info_event(progress=self._progress(stage_id), message="Naming topics with the LLM...")

        entities = self._context['entities']
        clusters = entities['clusters'].unique().tolist()
        task = "llm-topic-nomenclature"
        topics = []
        topic_names : Dict[int, str] = {}
        k = 0   # position in `clusters` : batches are processed in order

        it = Map(
            iterable=Batched(
                iterable=Map(
                    iterable=clusters,
                    fn=lambda cluster_id : self.represent_cluster(entities, cluster_id),
                ),
                n=self.config.llm_batch_size
            ),
            fn=self._query_llm
        )

        for batch in self.track(it=it, task=task):
            for result in batch:

                topic_id = clusters[k]
                k += 1

                if is_successful(result):
                    topic = result.unwrap()
                    topic.setdefault('id', topic_id)
                    topics.append(topic)
                    topic_names[topic_id] = topic[self.TOPIC_NAME_KEY]
                else:
                    yield self.make_error_event(
                        progress=self.get_task_progress(task=task),
                        task=task,
                        message=f"Error occured during topics nomenclature phase (topic {topic_id}) : {result.failure()}."
                    )

        self._context['topics'] = topics
        self._context['topic_names'] = topic_names

        yield self.make_info_event(
            progress=self._progress(stage_id, 1.0),
            message=f"Topics named : {len(topics)}/{len(clusters)}."
        )

    def name_expertises(self, stage_id : int) -> Generator[Event, None, None]:

        yield self.make_info_event(progress=self._progress(stage_id), message="Naming expertises...")

        topic_names = self._context['topic_names']
        pairs = self._context['expertise_pairs']

        def topic_name(topic_id : int) -> str:
            return topic_names.get(topic_id, f"Topic {topic_id}")

        expertise_names = [
            f"{topic_name(t1)} x {topic_name(t2)}"
            for t1, t2 in pairs.t().tolist()
        ]
        self._context['expertise_names'] = expertise_names

        unnamed = sorted({t for pair in pairs.t().tolist() for t in pair if t not in topic_names})

        if unnamed:
            yield self.make_warn_event(
                progress=self._progress(stage_id, 1.0),
                message=f"{len(unnamed)} topics without LLM name used in expertises, fallback 'Topic <id>' used : {unnamed}."
            )

        yield self.make_info_event(
            progress=self._progress(stage_id, 1.0),
            message=f"Expertises named : {len(expertise_names)}."
        )

    def update_graph(self, stage_id : int) -> Generator[Event, None, None]:

        yield self.make_info_event(progress=self._progress(stage_id), message="Adding topics and expertises to the graph...")

        graph = self._context['graph']
        topic_names = self._context['topic_names']
        topics_by_id = {t['id'] : t for t in self._context['topics']}
        expertise_names = self._context['expertise_names']
        expertise_pairs = self._context['expertise_pairs'].t().tolist()
        expertise_num_articles = self._context['expertise_num_articles'].tolist()

        ### Graph node ids of entities / articles, indexed like data['Entity'] / data['Article']
        entity_ids = [n.id for n in graph.get_nodes(NodeTypes.Entity)._nodes]
        article_ids = [n.id for n in graph.get_nodes(NodeTypes.Article)._nodes]

        def topic_node_id(t : int) -> str:
            return f"Topic-{t}"

        def expertise_node_id(e : int) -> str:
            return f"Expertise-{e}"

        ### Topic nodes
        topic_ids = self._context['topic_entity'][0].unique().tolist()

        for t in self._tracked(topic_ids, task="adding-topic-nodes"):
            graph.add_node(NodeTypes.Topic, topic_node_id(t), {
                **{k : v for k, v in topics_by_id.get(t, {}).items() if k != 'id'},
                'topic_id' : t,
                'name' : topic_names.get(t, f"Topic {t}"),
            })

        ### Expertise nodes
        for e in self._tracked(list(range(len(expertise_pairs))), task="adding-expertise-nodes"):
            t1, t2 = expertise_pairs[e]
            graph.add_node(NodeTypes.Expertise, expertise_node_id(e), {
                'expertise_id' : e,
                'name' : expertise_names[e],
                'topics' : [t1, t2],
                'num_articles' : expertise_num_articles[e],
            })

        ### Topic - Entity edge properties : topic_entity columns follow the rows of `entities`
        entities = self._context['entities']
        distances = entities['distances'].tolist()
        importances = entities['scores'].tolist()
        assert len(distances) == self._context['topic_entity'].size(1)

        def topic_entity_props(k : int) -> Dict[str, Any]:
            return {'distance' : distances[k], 'importance' : importances[k]}

        ### Edges : (edge type, edge index, src id fn, dst id fn, properties fn of the edge position)
        edges = [
            (EdgeTypes.Topic_INCLUDES_Entity, 'topic_entity', topic_node_id, lambda i : entity_ids[i], topic_entity_props),
            (EdgeTypes.Topic_COVERS_Article, 'topic_article', topic_node_id, lambda i : article_ids[i], None),
            (EdgeTypes.Expertise_COVERS_Article, 'expertise_article', expertise_node_id, lambda i : article_ids[i], None),
            (EdgeTypes.Expertise_COMBINES_Topic, 'expertise_topic', expertise_node_id, topic_node_id, None),
        ]

        for edge_type, key, src_id, dst_id, props in edges:

            pairs = self._context[key].t().tolist()
            task = f"adding-{edge_type.src.lower()}-{edge_type.dst.lower()}-edges"

            for k, (src, dst) in enumerate(self._tracked(pairs, task=task)):
                graph.add_edge(
                    edge_type, src_id(src), dst_id(dst),
                    properties=props(k) if props is not None else None
                )

        ### Cached views (pygeo, degrees, raw_data, ...) are now stale
        invalidate_graph_caches(graph)

        p = self._progress(stage_id, 1.0)
        node_counts = graph.node_counts
        edge_counts = graph.edge_counts

        yield self.make_debug_event(
            progress=p,
            message=f"Graph nodes : "
                f"Topic = {node_counts.get(NodeTypes.Topic, 0)} | "
                f"Expertise = {node_counts.get(NodeTypes.Expertise, 0)}"
        )

        yield self.make_debug_event(
            progress=p,
            message="Graph edges : " + " | ".join(
                f"{edge_type} = {edge_counts.get(edge_type, 0)}" for edge_type, *_ in edges
            )
        )

        yield self.make_info_event(progress=p, message="Graph updated.")

        graph_dir = self.config.graph_dir
        self._context['graph'].save(graph_dir)
        yield self.make_info_event(progress=p, message=f"Graph saved to path {graph_dir}.")

    ############################################################
    ### Helpers
    ############################################################

    def _tracked(self, items : List, task : str, num_updates : int = 20) -> Generator[Any, None, None]:
        """`self.track` with ~`num_updates` progress events instead of one per item."""
        yield from self.track(it=items, task=task, update_every=max(1, len(items) // num_updates))

    def load_model(self) -> TopicModel:
        return self.experiment_config.load_model(ctx=Context(data=self._context['data']))

    def _make_entities(self) -> TensorDict:

        results = self._context['results']
        graph = self._context['graph']

        X = results['entities_sem_embeddings']
        X = X / torch.norm(X, p=2, dim=-1, keepdim=True)

        return TensorDict(
            x=X,
            indices=torch.arange(X.size(0), device=X.device),
            attn_weights=results['attn_weights'],
            scores=self._context['scores'],
            names=np.array([n.properties['name'] for n in graph._nodes['Entity']._nodes])
        )

    def represent_cluster(self, data : TensorDict, cluster_idx : int) -> Dict:

        cluster = data[data['clusters'] == cluster_idx]
        x = cluster['x'].size(0)

        k = math.floor(
            x / min(
                math.ceil(
                    int(x < 30) * x + int(x >= 30) * (29 + (50*x)/(500+x))
                ),
                x
            )
        ) - 1

        selected = select_representative_nodes(
            cluster=cluster,
            max_num_representatives=self.config.max_representative_nodes,
            k=k,
            th_eliminate=0.3,
            th_merge=0.45 if x <= self.config.max_representative_nodes else None
        )

        subset = cluster[selected]

        entities = [
            dict(zip(['id', 'name', 'local_importance', 'global_importance'], row))
            for row in zip(
                subset['indices'].tolist(),
                subset['names'].tolist(),
                subset['distances'].tolist(),
                subset['scores'].tolist()
            )
        ]

        min_local, max_local = minmax([e['local_importance'] for e in entities])
        min_global, max_global = minmax([e['global_importance'] for e in entities])

        for e in entities:
            e['local_importance'] = minmax_norm(e['local_importance'], min=min_local, max=max_local, base=0.1)
            e['global_importance'] = minmax_norm(e['global_importance'], min=min_global, max=max_global, base=0.1)

        return {
            "id" : cluster_idx,
            "entities" : entities
        }

    def _query_llm(self, clusters : List[Dict]) -> List[RequestReturnType[Dict]]:

        return self.llm.batch_make_async_request(
            prompt_name='ClusterTopicNaming',
            inputs=clusters,
            lang='en',
            validate_input=True,
            validate_output=True
        )

    @classmethod
    def from_config(cls, config: Config) -> 'ExpertiseMapConstructor':
        builder = ExpertiseMapConstructor(config=config.conf, llm=LLM(config.conf.llm))
        builder.add_handler(ConsoleLogger("ExpertiseMapConstructor"))
        return builder


if __name__ == '__main__':
    config = Config(expertise_id="test")
    constructor = ExpertiseMapConstructor.from_config(config=config)
    constructor.run()