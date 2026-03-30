import os
import sys
import torch
import pandas as pd
from tqdm.auto import tqdm
from torch import Tensor, nn, optim
from torch.nn import functional as F
from definitions import DATA_DIR, EXPERIMENTS_DIR
from typing import Literal, Tuple, Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from argparse import ArgumentParser
from torch_geometric import utils as gnn_utils
from torch_sparse import matmul
from torchmetrics.functional import accuracy,f1_score
from sklearn.model_selection import KFold
from torch_scatter import scatter
from sklearn.cluster import kmeans_plusplus
from src import helpers
from src.gnn import ArticleAsKG, ArticlesGraph, KnowledgeGraph, HybridSampler, HetModelReturn, HetModel, HetGNNConfig

@dataclass
class Args:
    task : str = 'hierarchical_classification'
    experiment : str = 'default'

@dataclass
class TrainingConfig:

    ### DATA
    corpus : str = 'large'
    filename : str = 'data'
    val_size : float = 0.25
    label : Literal['teams','sub_dsis','dsis'] = 'teams'
    num_splits : int = 4
    adjs : Optional[List[str]] = None
    keep_largest_cc : bool = False
    split : Literal['k-fold','train_test'] = 'k-fold'

    ### Losses
    add_structure_loss : bool = False
    add_constrastive_loss : bool = False
    add_clustering_loss : bool = False

    ### Sampling
    batch_size : int = 8
    num_articles_neighbors : tuple[int,int] = (10,6)
    num_entities_neighbors : tuple[int,int] = ()

    ### Model
    model_params : HetGNNConfig = field(default_factory=HetGNNConfig)

    ### Learning Rate
    lr : float = 0.00075
    num_epochs : int = 10
    accumelation_steps : int = 1
    weight_decay : float = 1e-4

    ### Device
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu'

    ### Negative Sampling
    do_negative_sampling : bool = False

    ### Loggin
    log_every : int = 1

@dataclass
class Config(TrainingConfig):

    ### SEED & Reproducibility
    seed : int = 56

    ### Grid-Search
    hparams : Dict[str,Dict[str, Any]] = field(default_factory = lambda : { "default" : {} })

def get_configs(config : Config) -> Dict[str, TrainingConfig]:

    configs = dict()

    for exp_name,hparam in config.hparams.items():

        config = utils.load_dataclass_from_dict(
            TrainingConfig, 
            utils.override_dict(asdict(config), hparam)
        )
        
        configs[exp_name] = config

    return configs

def negative_sample(y : Tensor) -> Tuple[Tensor, Tensor]:

    pos_edge_index = torch.argwhere(y).t()

    neg_edge_index = gnn_utils.negative_sampling(
        edge_index = pos_edge_index,
        num_nodes = (y.size(0), y.size(1))
    )

    row, col = torch.cat([pos_edge_index,neg_edge_index],dim = -1)

    return row, col

def load_data(
    config : TrainingConfig, 
    splits_path : Optional[str] = None, 
    verbose : bool = True
) -> ArticleAsKG:

    #####################################################################
    ############ Load Articles Graph ####################################
    #####################################################################

    ### Load the raw data
    articles = os.path.join(DATA_DIR, config.corpus, f'{config.filename}.articles.pt')
    articles = torch.load(f = articles, map_location = 'cpu', weights_only = False)

    ### Create the adjecency matrices
    adj = {}

    adj['share-author'] = gnn_utils.to_edge_index(matmul(
        src = articles['adj'][('article','author','has-author')],
        other = articles['adj'][('article','author','has-author')].t()
    ))[0]

    adj['share-keyword'] = gnn_utils.to_edge_index(matmul(
        src = articles['adj'][('article','keyword','has-keyword')],
        other = articles['adj'][('article','keyword','has-keyword')].t()
    ))[0]

    edge_index = []
    edge_type = []

    if config.adjs is not None:
        
        for rel in config.adjs:
            adj.pop(rel)

    for i,A in enumerate(adj.values()):
        edge_index.append(A)
        edge_type.append(torch.zeros_like(A[0]) + i)

    edge_index = torch.cat(edge_index, dim = -1).long()
    edge_type = torch.cat(edge_type, dim = -1).long()
    y = articles[config.label].to_dense()

    #####################################################################
    ############ Load Knowledge Graph ###################################
    #####################################################################

    ### Load the raw data
    kg = os.path.join(DATA_DIR, config.corpus, f'{config.filename}.features.pt')
    kg = torch.load(f = kg, map_location = 'cpu', weights_only = False)

    #####################################################################
    ############ Convert to PyG Format ##################################
    #####################################################################

    articles = ArticlesGraph(
        x = articles['nodes']['article'],
        edge_index = edge_index,
        edge_type = edge_type,
        y = y
    )

    edge_article = kg['edge_article'].t()
    node_type = kg['node_type']

    kg = KnowledgeGraph(
        x = kg['x'],
        e = kg['e'],
        r = kg['r'],
        edge_index = kg['edge_index'],
        edge_type = kg['edge_type'],
    )

    kg.node_type = node_type

    data = ArticleAsKG(
        articles = articles,
        kg = kg,
        edge_article = edge_article
    )

    ### Filter Articles With Few entities
    indices = torch.arange(data.articles.num_nodes)
    entities_per_node = gnn_utils.degree(data.entity_article[1], num_nodes=data.articles.num_nodes)

    ### Splitting
    is_labled = (data.articles.y.sum(-1) > 0) & (entities_per_node >= 5)
    labled_indices = torch.argwhere(is_labled)
    k_fold = KFold(n_splits = config.num_splits, shuffle = True)

    split = torch.zeros_like(is_labled).long() - 1

    for fold,(train_indices, val_indices) in enumerate(k_fold.split(labled_indices)):
        split[labled_indices[val_indices]] = fold

    if splits_path is not None:

        if os.path.exists(splits_path) and verbose:
            split = torch.load(splits_path, weights_only = False, map_location = 'cpu')
            print("Loading prexisting split.\n")
        else:
            torch.save(split, f = splits_path)

    data.articles.split = split

    ### KG Labels
    data.kg.y = scatter(
        src = data.articles.y[data.entity_article[1]],
        index = data.entity_article[0],
        dim = 0,
        dim_size = data.kg.num_nodes,
        reduce = "max"
    )

    ### Normalization
    mean, std = data.articles.x.mean(0, keepdim = True), data.articles.x.std(0, keepdim = True)
    data.articles.x = (data.articles.x - mean) / (std + 1e-3)

    mean, std = data.kg.x.mean(0, keepdim = True), data.kg.x.std(0, keepdim = True)
    data.kg.x = (data.kg.x - mean) / (std + 1e-3)

    mean, std = data.kg.r.mean(0, keepdim = True), data.kg.r.std(0, keepdim = True)
    data.kg.r = (data.kg.r - mean) / (std + 1e-3)

    mean, std = data.articles.x.mean(0, keepdim = True), data.articles.x.std(0, keepdim = True)
    data.articles.x = (data.articles.x - mean) / (std + 1e-3)

    ### Describe
    if config.keep_largest_cc:
        labels = utils.torch_connected_components(edge_index=data.kg.edge_index, num_nodes=data.kg.num_nodes)
        ccs, counts = labels.unique(return_counts=True)
        largest_cc = ccs[counts.argmax()]
        indices = torch.arange(data.kg.num_nodes)
        data = data.subgraph_entities(indices[labels == largest_cc])

    if verbose:
        print(f"Labled Articles : {is_labled.sum()} / {len(is_labled)}")
        data.describe()
        print(data.articles.y.sum(0))

    return data

def create_dataloaders(config : TrainingConfig, data : ArticleAsKG) -> List[Tuple[HybridSampler,HybridSampler]]:

    samplers = []

    for fold in range(config.num_splits):

        train_mask = (data.articles.split != -1) & (data.articles.split != fold)
        val_mask = (data.articles.split != -1) & (data.articles.split == fold)

        train_split = data.subgraph(data.articles.article_id[train_mask])
        val_split = data.subgraph(data.articles.article_id[val_mask])

        ### Create Sampler
        samplers.append(tuple([
            HybridSampler(
                articles = split.articles,
                kg = split.kg,
                edge_article = split.edge_article,
                articles_loader_args = {
                    'batch_size' : config.batch_size,
                    'num_neighbors' : config.num_articles_neighbors,
                    'drop_last' : split_id % 2 == 0,
                    "shuffle" : split_id % 2 == 0
                },
                kg_sampler_args = {
                    'num_neighbors' : config.num_entities_neighbors
                }
            )
            for split_id, split in enumerate([train_split, val_split])
        ]))

    if config.split == "train_test":
        samplers = [samplers[0]]

    return samplers

def create_model(config : TrainingConfig, data : ArticleAsKG) -> HetModel:

    config.model_params.ssl = config.add_structure_loss
    config.model_params.cluster = config.add_clustering_loss

    return HetModel.from_config(
        in_channels = data.kg.num_features,
        num_labels = data.articles.y.size(-1),
        num_relations = (data.kg.num_relations, data.articles.num_relations),
        config = config.model_params
    )

def drop_edges(batch : ArticleAsKG, rate : float = 0.25) -> ArticleAsKG:

    edge_article = batch.edge_article
    indices = torch.randint(0, edge_article.size(-1), size = (int((1 - rate) * edge_article.size(-1)), ))

    return ArticleAsKG(
        articles = batch.articles,
        kg = batch.kg,
        edge_article = batch.edge_article[:,indices]
    )

def sample_negative_edges(
    edge_index : Tensor,
    edge_type : Tensor,
    num_nodes : int
) -> Tuple[Tensor, Tensor, Tensor]:
    
    negative_edge_index = gnn_utils.negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes
    )

    new_edge_index = torch.cat([negative_edge_index, edge_index], dim = -1)
    y_hat = torch.cat([torch.zeros_like(edge_index[0]), torch.ones_like(edge_index[1])], dim = 0).float()

    edge_type = torch.cat([edge_type, edge_type], dim = 0)

    return new_edge_index, edge_type, y_hat

def loss(config : TrainingConfig,
    batch : ArticleAsKG, 
    results : HetModelReturn,
    batch_size : int,
    labels : Optional[Tensor] = None,
    negative_sample : Optional[Tuple[Tensor, Tensor]] = None,
    weights : Optional[Dict] = None
) -> Dict[str, Tensor]:

    losses = {}
    weights = weights or {}

    y_hat = results.articles_predictions[:batch_size]
    y = batch.articles.y[:batch_size]

    ### Supervised Loss
    if negative_sample is not None:
        row, col = negative_sample
        y_hat = y_hat[row, col]
        y = y[row, col]

    losses['guidance_loss'] = F.binary_cross_entropy_with_logits(y_hat, y.float())

    ### Structure Loss
    if config.add_structure_loss:
        losses['ssl_loss'] = F.binary_cross_entropy_with_logits(results.structure_y_hat, labels.float())

    ### CLIP like loss
    if config.add_constrastive_loss:

        losses["constrastive_loss"] = utils.constrastive_loss(
            entities=results.entities_embeddings, 
            articles=results.articles_embeddings,
            entity_article=batch.entity_article,
            tau=0.02
        )

    ### Clustering Loss
    if config.add_clustering_loss:
        # losses['cluster_loss'] = utils.drink_net_loss(C = centroids, z = results.entities_embeddings)
        losses['cluster_loss'] = utils.dmon_cluster_loss(
            C = results.C,
            A = results.A
        )

    loss = torch.scalar_tensor(0.0).to(y_hat.device)

    for loss_name, loss_value in losses.items():

        loss_weight =  weights.get(loss_name, 1.0) 

        if loss_weight > 0:
            loss += loss_weight * loss_value

    losses["loss"] = loss

    return losses

def evaluate(config : TrainingConfig,
    batch : ArticleAsKG, 
    results : HetModelReturn,
    batch_size : int,
    labels : Optional[Tensor] = None,
    negative_sample : Optional[Tuple[Tensor, Tensor]] = None
) -> Dict[str, Tensor]:
    
    metrics = {}
    
    y = batch.articles.y[:batch_size]
    y_hat = results.articles_predictions[:batch_size]

    if negative_sample is not None:

        row, col = negative_sample
        y = y[row, col]
        y_hat = y_hat[row, col]

        metrics["guidance_acc"] = accuracy(y_hat, y, task="binary").item()
        metrics["guidance_f1"] = f1_score(y_hat, y, task="binary").item()
    
    else:

        metrics["guidance_acc"] = accuracy(
            y_hat, 
            y, 
            task="multilabel", 
            num_labels=batch.articles.y.size(-1), 
            average="macro"
        ).item()

        metrics["guidance_f1"] = f1_score(
            y_hat, 
            y, 
            task="multilabel", 
            num_labels=batch.articles.y.size(-1), 
            average="macro"
        ).item()

    if config.add_structure_loss:

        metrics["structure_acc"] = accuracy(results.structure_y_hat, labels, task="binary").item()
        metrics["structure_f1"] = f1_score(results.structure_y_hat, labels, task="binary").item()

        row, col = batch.kg.edge_index
        h, t = results.entities_embeddings[row], results.entities_embeddings[col]
        r_hat = (h - t) @ results.relations_embeddings.t()

        metrics['HITS@5'] = accuracy(
            r_hat, 
            batch.kg.edge_type, 
            task='multiclass', 
            num_classes=batch.kg.r.size(0),
            top_k = 5
        )

        metrics['HITS@10'] = accuracy(
            r_hat, 
            batch.kg.edge_type, 
            task='multiclass', 
            num_classes=batch.kg.r.size(0),
            top_k = 10
        )

    return metrics

def set_centroids(config : TrainingConfig,
    model : HetModel,
    data : ArticleAsKG
) -> None:

    model = model.eval().cpu()
    data = data.cpu()

    sample = None

    if model.ssl:
        sample = (data.kg.edge_index, data.kg.edge_type)

    with torch.no_grad():
        entities_embeddings = model.forward(data, sample).entities_embeddings
        centers, _ = kmeans_plusplus(entities_embeddings.detach().cpu().numpy(), n_clusters=model.num_clusters)
        centers = torch.from_numpy(centers).to(config.device)
        model.to(config.device)
        model.set_clusters(centers)
        model.train()

    print("Clusters are set.")

def run_experiment(
    config : TrainingConfig,
    data : ArticleAsKG,
    train_loader : HybridSampler,
    val_loader : HybridSampler
) -> Tuple[pd.DataFrame, nn.Module]:
    
    ### Creat Model
    model = create_model(config = config, data = data).to(config.device)
    optimizer = optim.Adam(params = model.parameters(), lr = config.lr, weight_decay = config.weight_decay)
    history = []
    loaders = { "train" : train_loader, "val" : val_loader }

    batch : ArticleAsKG = None

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimizer, 
        T_max = config.num_epochs, 
        eta_min = 1e-2 * config.lr
    )

    flag : bool = False

    ### Training Loop
    for epoch in range(config.num_epochs):

        """if epoch == config.num_epochs // 2:
            set_centroids(config=config, model=model, data=data)
            flag = True"""

        local_history = []

        for phase in ["train", "val"]:

            is_training = phase == "train"
            model = model.train(mode = is_training)
            loader = loaders[phase]

            iteration = 0

            for batch in tqdm(loader, desc = f'Epoch {epoch+1}/{config.num_epochs} | Phase {phase.upper()}'):
                
                batch = utils.move_data_to_device(batch, device = config.device)
                # augmented_batch = drop_edges(batch, rate = 0.2)

                ### Negative Sampling
                sample = None
                labels = None

                if config.model_params.ssl:

                    edge_index, edge_type, labels = sample_negative_edges(
                        batch.kg.edge_index, 
                        batch.kg.edge_type, 
                        num_nodes=batch.kg.num_nodes
                    )

                    sample = (edge_index, edge_type)

                with torch.set_grad_enabled(is_training):

                    results = model.forward(batch, sample)

                    ### Label Negative Sampling
                    y = batch.articles.y[:config.batch_size]
                    neg_sample = None

                    if is_training and config.do_negative_sampling:
                        neg_sample = negative_sample(y)

                    ### Calculate Loss
                    losses = loss(config,
                        batch = batch,
                        results = results,
                        batch_size=config.batch_size,
                        labels = labels,
                        negative_sample=neg_sample,
                        weights={
                            "guidance_loss" : 1.0,
                            "ssl_loss" : 0.05,
                            "constrastive_loss" : 0.02,
                            "cluster_loss" : 0.2
                        }                    
                    )

                if is_training:
                    optimizer.zero_grad()
                    losses["loss"].backward()
                    optimizer.step()

                metrics = evaluate(config,
                    batch=batch,
                    results=results,
                    batch_size=config.batch_size,
                    labels=labels,
                    negative_sample=neg_sample
                )

                local_history.append({
                    'epoch' : epoch,
                    'phase' : phase,
                    'iteration' : iteration,
                    **{
                        loss_name : loss_value.item()
                        for loss_name, loss_value in losses.items()
                    },
                    **metrics
                })

                iteration += 1

            print(end = "\n")

        scheduler.step()

        local_history = pd.DataFrame(local_history)
        history.append(local_history)

        ### Early Stopping
        local_history = local_history \
            .drop(columns=['iteration','epoch']) \
            .groupby(by=['phase']) \
            .mean()
        
        ### Logging
        if (epoch + 1) % config.log_every == 0:
            
            for col in local_history.columns:
                if not col.endswith('loss'):
                    local_history[col] = local_history[col].apply(lambda x : f'{100 * x:.2f}%')

            print("#" * 144)
            print(local_history)
            print("#" * 144, end = "\n\n")

            print(f"\nLearning Rate = {scheduler.get_last_lr()[0]}\n")

    return pd.concat(history), model

def inference(
    model : HetModel,
    data : ArticleAsKG
) -> Tensor:
    
    model = model.eval().cpu()
    data = data.cpu()

    model.ssl = False
    model.cluster = False

    with torch.no_grad():
        Y_hat = model.forward(data).entities_predictions

    return Y_hat

def main(base_config : Config) -> None:

    utils.seed_everything(base_config.seed)

    configs = get_configs(base_config)
    
    all_history = []
    exp_dir = os.path.join(EXPERIMENTS_DIR, args.task, args.experiment)

    models_dir = os.path.join(exp_dir, 'models')
    results_dir = os.path.join(exp_dir, 'results')

    os.makedirs(models_dir, exist_ok = True)
    os.makedirs(results_dir, exist_ok = True)

    print(f"There is a total of {len(configs)} Experiments.\n")

    splits_path = os.path.join(EXPERIMENTS_DIR, args.task, f'split_{base_config.label}_{base_config.num_splits}.pt')

    for (exp_name,config),hparams in zip(configs.items(),base_config.hparams.values()):

        data = load_data(config, splits_path)
        loaders = create_dataloaders(config, data)

        print(f"Experiment {exp_name} With Hyper-Parameters : \n")
        print(utils.format_dict(hparams), end = "\n\n")

        for trial,(train_loader,val_loader) in enumerate(loaders):

            print(f"Trial = {trial} : \n")
            
            ### Run Experiment
            history,model = run_experiment(config, data, train_loader, val_loader)
            history['exp'] = exp_name
            history['trial'] = trial

            all_history.append(history)

            ### Save Results
            torch.save(model.state_dict(), f = os.path.join(models_dir, f'model_{exp_name}_{trial}.pt'))
            predictions = inference(model, data)
            torch.save(predictions, f = os.path.join(results_dir, f'predictions_{exp_name}_{trial}.pt'))

            print("\n")

    all_history = pd.concat(all_history)

    all_history.to_csv(
        path_or_buf = os.path.join(exp_dir, 'history.csv'),
        index = False
    )

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--task', type=str, default=Args.task)
    parser.add_argument('--experiment', type=str, default=Args.experiment)

    args = Args(**vars(parser.parse_args()))

    config = utils.load_dataclass(Config, filename = os.path.join(EXPERIMENTS_DIR, args.task, args.experiment, 'config.json'))
    
    main(config)