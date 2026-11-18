import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric import utils as gnn_utils
from typing import Callable, Dict, Literal
from warnings import warn

SCORING_METHOD = Literal[
    'average_document_importance',
    'clipped_sum_document_importance',
    'raw_importance',
    'average_document_normalized_rank'
]

def average_document_importance(
    entity_article : Tensor,
    attn_weights : Tensor,
    num_articles : int,
    num_entities : int
) -> Tensor:

    entities_ids, articles_ids = entity_article

    entities_weights = gnn_utils.softmax(
        src=attn_weights[entities_ids],
        index=articles_ids,
        dim=0,
        num_nodes=num_articles
    )

    average_entities_weights = scatter(
        src=entities_weights,
        index=entities_ids,
        dim=0,
        dim_size=num_entities,
        reduce='mean'
    )

    return average_entities_weights.flatten()

def clipped_sum_document_importance(
    entity_article : Tensor,
    attn_weights : Tensor,
    num_articles : int,
    num_entities : int,
    min_importance : float = 0.0
) -> Tensor:

    entities_ids, articles_ids = entity_article
    
    entities_weights = gnn_utils.softmax(
        src=attn_weights[entities_ids],
        index=articles_ids,
        dim=0,
        num_nodes=num_articles
    )

    average_entities_weights = scatter(
        src=entities_weights - min_importance,
        index=entities_ids,
        dim=0,
        dim_size=num_entities,
        reduce='sum'
    )

    return average_entities_weights.flatten()

def raw_importance(
    entity_article : Tensor,
    attn_weights : Tensor,
    num_articles : int,
    num_entities : int,
) -> Tensor:
    return attn_weights.sigmoid().flatten()

def average_document_normalized_rank(
    entity_article : Tensor,
    attn_weights : Tensor,
    num_articles : int,
    num_entities : int,
) -> Tensor:

    entities_ids, articles_ids = entity_article
    indices = torch.arange(len(articles_ids))
    ranks = torch.zeros_like(entities_ids, dtype=torch.float)

    for doc in articles_ids.unique():

        mask = articles_ids == doc

        weights = attn_weights[mask]
        entities_indices = indices[mask]

        rank = weights.argsort(descending=True) + 1
        relative_rank = 1 - (rank.float() / mask.float().sum())
        ranks[entities_indices] = relative_rank

    entities_relative_rank = scatter(
        src=ranks,
        index=entities_ids,
        dim=0,
        dim_size=num_entities,
        reduce='mean'
    )

    return entities_relative_rank.flatten()

_registrey : Dict[str, Callable] = {
    fn.__name__ : fn
    for fn in [
        average_document_importance, clipped_sum_document_importance,
        raw_importance, average_document_normalized_rank
    ]
}

def score(
    scoring_method : str,
    entity_article : Tensor,
    attn_weights : Tensor,
    num_articles : int,
    num_entities : int,
    **kwargs
) -> Tensor:

    ### Check function exists
    if scoring_method not in _registrey:

        raise KeyError(
            f"method {scoring_method} is not available. "
            f"Available methods : {'|'.join(_registrey.keys())}"
        )

    ### Filter kwargs
    fn = _registrey[scoring_method]

    annotations_dict = fn.__annotations__
    annotations_dict.pop('entity_article')
    annotations_dict.pop('attn_weights')
    annotations_dict.pop('num_articles')
    annotations_dict.pop('num_entities')

    new_kwargs = {}

    for k, v in kwargs.items():

        if k not in annotations_dict:
            warn(f"argument `{k}` is ignored.")
            continue

        new_kwargs[k] = v

    return fn(
        entity_article,
        attn_weights,
        num_articles,
        num_entities,
        **new_kwargs
    )