import torch
import os
import pandas as pd
from torchmetrics.functional import accuracy,f1_score
from rich.progress import Progress
from pydantic import BaseModel, Field, computed_field
from typing import Self, Dict, Any, List, Tuple, Optional, Generator
from definitions import EXPERIMENTS_DIR, DATA_DIR
from torch_geometric import utils as gnn_utils
from sklearn.model_selection import KFold
from torch import optim, Tensor
from torch.nn import functional as F
from src import helpers
from src.shared.command_tools import BaseCommand
from src.shared.handlers import Event
from src.gnn import HetGNNConfig, ArticleAsKG, ArticlesGraph, KnowledgeGraph, HybridSampler, HetModel, HetModelReturn
from src.shared.handlers import ConsoleLogger
from instance import cli

class Config(BaseModel):
    
    experiment : str = 'default'

    @computed_field
    @property
    def training_config(self) -> 'TrainingConfig':
        config_path = os.path.join(EXPERIMENTS_DIR, self.experiment, 'config.json')
        conf = helpers.load_json_file(config_path)
        return TrainingConfig(**conf)

class BaseTrainingConfig(BaseModel):

    ### Data
    graph : str = 'ls2n-duke'           # Graph found at $DATA_DIR/$graph/graph.pt
    keep_largest_cc : bool = True       # Whether to keep only largest connected component or not
    min_entities_per_article : int = 5  # The minimum number of entities for an article to be considered

    ### Losses
    add_ssl_loss : bool = False    # Whether to add self-supervised (reconstruction) loss or not.

    ### Sampling
    batch_size : int = 8
    num_articles_neighbors : tuple[int,int] = (10,6)  # SAGE sampling for articles
    num_entities_neighbors : tuple[int,int] = ()      # SAGE sampling for entities

    ### Model
    model_params : HetGNNConfig = Field(default_factory=HetGNNConfig) # Model parameters

    ### Hyper-parameters
    lr : float = 0.00075          # Learning rate        
    num_epochs : int = 10         # Number of epochs
    accumelation_steps : int = 1  # Accumelation steps
    weight_decay : float = 1e-4   # Weight decay

    ### Device
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device

    ### Negative Sampling
    do_negative_sampling : bool = False # Whether to do negative sampling when learning connections between articles & labels

    ### Logging
    log_every : int = 1 # Logging frequency

class TrainingConfig(BaseTrainingConfig):
    seed : int = 56      # SEED for reproducibility
    num_splits : int = 4 # number of splits used in KFold
    hparams : Dict[str,Dict[str, Any]] = Field(default_factory = lambda : { "default" : {} }) # Variations for hyper-parameters

@cli.register_stage(
    name="train",
    description=(
        "Trains the Knowledge Representation Learning module based on the configuration specified in "
        "the experiment path."
    ),
    add_from_conf=False,
    add_save_to=False
)
class Trainer(BaseCommand):

    def __init__(self, experiment : str, config : TrainingConfig):
        super().__init__()

        self.experiment = experiment
        self.config = config
        self.current_config : Optional[BaseTrainingConfig] = None

    def get_configs(self) -> Optional[Dict[str, BaseTrainingConfig]]:

        self.send_info_event(
            progress=0.0,
            message="Loading all configurations combinations..."
        )

        configs = dict()

        for exp_name,hparam in self.config.hparams.items():

            base_config = self.config.model_dump()
            base_config.pop('hparams')

            config = BaseTrainingConfig(**helpers.override_dict(
                base_config,
                hparam
            ))
            
            configs[exp_name] = config

        self.send_debug_event(
            progress=0.0,
            message=f"Found {len(configs)} configuration combinations"
        )

        return configs
    
    def load_data(self) -> ArticleAsKG:

        data_path = os.path.join(DATA_DIR, *self.config.graph.split('/'), "graph.pt") # os.path.join(DATA_DIR, *self.config.graph.split('/'), "graph.pt")

        self.send_debug_event(
            progress=0.0,
            message=f"Loading data at {data_path}."
        )
        
        raw = torch.load(data_path, weights_only=False)

        kg = KnowledgeGraph(
            x = raw['entities'],
            e = raw['predicates'][raw['edge_type']],
            r = raw['predicates'],
            edge_index = raw[('entity','has-relation-with', 'entity')].t(),
            edge_type = raw['edge_type'],
            entity_names=raw['metadata']['entities']
        )

        edge_index = torch.cat([
            raw[('article', 'share-author-with', 'article')],
            raw[('article', 'share-keyword-with', 'article')]
        ], dim = -1)

        edge_type = torch.cat([
            torch.zeros_like(raw[('article', 'share-author-with', 'article')][0]),
            torch.zeros_like(raw[('article', 'share-keyword-with', 'article')][0])
        ])

        articles = ArticlesGraph(
            x = raw['articles'],
            edge_index = edge_index,
            edge_type = edge_type,
            y = raw['labels']
        )

        data = ArticleAsKG(
            articles = articles,
            kg = kg,
            edge_article = raw['triple_article'].t()
        )

        data.metadata = raw['metadata']

        return self.normalize_data(data)

    def normalize_data(self, data : ArticleAsKG) -> ArticleAsKG:

        mean, std = data.articles.x.mean(0, keepdim = True), data.articles.x.std(0, keepdim = True)
        data.articles.x = (data.articles.x - mean) / (std + 1e-3)

        mean, std = data.kg.x.mean(0, keepdim = True), data.kg.x.std(0, keepdim = True)
        data.kg.x = (data.kg.x - mean) / (std + 1e-3)

        mean, std = data.kg.r.mean(0, keepdim = True), data.kg.r.std(0, keepdim = True)
        data.kg.r = (data.kg.r - mean) / (std + 1e-3)

        mean, std = data.kg.edge_attr.mean(0, keepdim = True), data.kg.edge_attr.std(0, keepdim = True)
        data.kg.edge_attr = (data.kg.edge_attr - mean) / (std + 1e-3)

        return data
    
    def keep_largest_cc(self, data : ArticleAsKG) -> ArticleAsKG:

        labels = helpers.torch_connected_components(edge_index=data.kg.edge_index, num_nodes=data.kg.num_nodes)
        ccs, counts = labels.unique(return_counts=True)
        largest_cc = ccs[counts.argmax()]
        indices = torch.arange(data.kg.num_nodes)

        self.send_info_event(
            progress=0.0,
            message=f"Largest Connected Componenet contains {counts.argmax()} entities."
        )

        return data.subgraph_entities(indices[labels == largest_cc])
    
    def filter_data(self, data : ArticleAsKG) -> Generator[Event, None, None]:

        ### Filter Articles With Few entities
        indices = torch.arange(data.articles.num_nodes)
        entities_per_node = gnn_utils.degree(data.entity_article[1], num_nodes=data.articles.num_nodes) 
        mask = (data.articles.y.sum(-1) > 0) # & (entities_per_node >= 5)

        self.send_debug_event(
            progress=0.0,
            message=f"Number of articles before/after filtering : {mask.size(0)}/{mask.sum()}."
        )

        return data.subgraph_articles(data.articles.article_id[mask])
    
    def create_or_load_split(self, data : ArticleAsKG) -> Tensor:

        split_path = os.path.join(EXPERIMENTS_DIR, self.experiment, f'split_{self.config.graph.split('/')[-1]}_{self.config.num_splits}.pt')

        if os.path.exists(split_path):

            self.send_debug_event(
                progress=0.0,
                message="Loading split tensor from disk"
            )

            split_tensor = torch.load(split_path, map_location='cpu', weights_only=False)

        else:

            k_fold = KFold(n_splits = self.config.num_splits, shuffle = True)

            split_tensor = torch.zeros((data.articles.num_nodes)).long() - 1
            indices = torch.arange(data.articles.num_nodes)

            for fold,(_, val_indices) in enumerate(k_fold.split(indices)):
                split_tensor[val_indices] = fold

            torch.save(split_tensor, split_path)

            self.send_info_event(
                progress=0.0,
                message="Split tensor saved to disk."
            )

        return split_tensor
    
    def create_samplers(self, data : ArticleAsKG) -> List[Tuple[HybridSampler, HybridSampler]]:

        split_tensor = self.create_or_load_split(data)
        
        samplers = []

        for fold in range(self.config.num_splits):

            train_mask = split_tensor != fold
            val_mask = split_tensor == fold

            self.send_debug_event(
                progress=0.0,
                message=f"Fold {fold} : N training samples = {train_mask.sum()}, N validation samples = {val_mask.sum()}."
            )

            train_split = data.subgraph_articles(data.articles.article_id[train_mask])
            val_split = data.subgraph_articles(data.articles.article_id[val_mask])

            ### Create Sampler
            samplers.append(tuple([
                HybridSampler(
                    articles = split.articles,
                    kg = split.kg,
                    edge_article = split.edge_article,
                    articles_loader_args = {
                        'batch_size' : self.config.batch_size,
                        'num_neighbors' : self.config.num_articles_neighbors,
                        'drop_last' : split_id % 2 == 0,
                        "shuffle" : split_id % 2 == 0
                    },
                    kg_sampler_args = {
                        'num_neighbors' : self.config.num_entities_neighbors
                    }
                )
                for split_id, split in enumerate([train_split, val_split])
            ]))

        return samplers
    
    def create_model(self, config : BaseTrainingConfig, data : ArticleAsKG) -> HetModel:

        self.send_info_event(
            progress=0.0,
            message="Creating GNN model..."
        )

        config.model_params.ssl = config.add_ssl_loss
        config.model_params.cluster = False

        model = HetModel.from_config(
            in_channels = data.kg.num_features,
            num_labels = data.articles.y.size(-1),
            num_relations = (data.kg.num_relations, data.articles.num_relations),
            config = config.model_params
        )

        self.send_debug_event(
            progress=0.0,
            message=f"Model size {helpers.model_size_mb(model):.3}MB."
        )

        return model
    
    @staticmethod
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
    
    @staticmethod
    def negative_sample(y : Tensor) -> Tuple[Tensor, Tensor]:

        pos_edge_index = torch.argwhere(y).t()

        neg_edge_index = gnn_utils.negative_sampling(
            edge_index = pos_edge_index,
            num_nodes = (y.size(0), y.size(1))
        )

        row, col = torch.cat([pos_edge_index,neg_edge_index],dim = -1)

        return row, col
    
    @staticmethod
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
            losses['guidance_loss'] = F.binary_cross_entropy_with_logits(y_hat[row, col], y[row, col].float())
        else:
            losses['guidance_loss'] = F.binary_cross_entropy_with_logits(y_hat, y.float())

        ### Structure Loss
        if config.add_ssl_loss:
            losses['ssl_loss'] = F.binary_cross_entropy_with_logits(results.structure_y_hat, labels.float())

        loss = torch.scalar_tensor(0.0).to(y_hat.device)

        for loss_name, loss_value in losses.items():

            loss_weight =  weights.get(loss_name, 1.0) 

            if loss_weight > 0:
                loss += loss_weight * loss_value

        losses["loss"] = loss

        return losses
    
    @staticmethod
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

        if config.add_ssl_loss:

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
            ).item()

            metrics['HITS@10'] = accuracy(
                r_hat, 
                batch.kg.edge_type, 
                task='multiclass', 
                num_classes=batch.kg.r.size(0),
                top_k = 10
            ).item()

        return metrics
    
    def run_experiment(self,
        config : BaseTrainingConfig,
        data : ArticleAsKG,
        train_loader : HybridSampler,
        val_loader : HybridSampler
    ) -> Tuple[pd.DataFrame, torch.nn.Module]:
        
        ### Creat Model
        model = self.create_model(config = config, data = data).to(config.device)
        optimizer = optim.Adam(params = model.parameters(), lr = config.lr, weight_decay = config.weight_decay)
        history = []
        loaders = { "train" : train_loader, "val" : val_loader }

        batch : ArticleAsKG = None

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer = optimizer, 
            T_max = config.num_epochs, 
            eta_min = 1e-2 * config.lr
        )

        ### Training Loop
        for epoch in range(config.num_epochs):

            local_history = []

            for phase in ["train", "val"]:

                is_training = phase == "train"
                model = model.train(mode = is_training)
                loader = loaders[phase]

                iteration = 0

                # tqdm(loader, desc = f'Epoch {epoch+1}/{config.num_epochs} | Phase {phase.upper()}')

                pbar = Progress()
                task = pbar.add_task(f"[blue]{f'Epoch {epoch+1}/{config.num_epochs} | Phase {phase.upper()}'}", total=len(loader))
                pbar.start()

                for iteration_id, batch in enumerate(loader):
                    
                    batch = helpers.move_data_to_device(batch, device = config.device)

                    ### Negative Sampling
                    sample = None
                    labels = None

                    if config.model_params.ssl:

                        edge_index, edge_type, labels = self.sample_negative_edges(
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
                            neg_sample = self.negative_sample(y)

                        ### Calculate Loss
                        losses = self.loss(config,
                            batch = batch,
                            results = results,
                            batch_size=config.batch_size,
                            labels = labels,
                            negative_sample=neg_sample,
                            weights={
                                "guidance_loss" : 1.0,
                                "ssl_loss" : 0.05,
                            }                    
                        )

                    if is_training:
                        optimizer.zero_grad()
                        losses["loss"].backward()
                        optimizer.step()

                    metrics = self.evaluate(config,
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

                    pbar.update(task, advance=1, refresh=True)

                pbar.remove_task(task)

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

                self.send_info_event(
                    progress=0.0,
                    message=f"Epoch {epoch+1}/{config.num_epochs} | Phase : Training",
                    extra=local_history.loc['train'].to_dict()
                )

                self.send_info_event(
                    progress=0.0,
                    message=f"Epoch {epoch+1}/{config.num_epochs} | Phase : Validation",
                    extra=local_history.loc['val'].to_dict()
                )

                self.send_info_event(
                    progress=0.0,
                    message=f"Learning rate = {scheduler.get_last_lr()[0]:.5f}."
                )

                print('\n')

        return pd.concat(history), model
    
    @staticmethod
    def inference(
        model : HetModel,
        data : ArticleAsKG
    ) -> Tensor:
        
        model = model.eval().cpu()
        data = data.cpu()

        model.ssl = False
        model.cluster = False

        with torch.no_grad():
            Y_hat = model.forward(data).articles_predictions

        return Y_hat
    
    def __iter__(self):

        ### Seed everything
        helpers.seed_everything(self.config.seed)
        
        ### Load all Configs :
        configs = self.get_configs()

        ### Data Loading
        data = self.load_data()

        MODELS_DIR = os.path.join(EXPERIMENTS_DIR, self.experiment, 'models')
        RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, self.experiment, 'results')
        HISTORIES_PATH = os.path.join(EXPERIMENTS_DIR, self.experiment, 'histories')

        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(HISTORIES_PATH, exist_ok=True)

        total = len(self.config.hparams) * self.config.num_splits

        for exp_id, (config_name, config) in enumerate(configs.items()):

            ### Preprocessing
            data = self.filter_data(data)

            if config.keep_largest_cc:
                data = self.keep_largest_cc(data)

            self.send_info_event(
                progress=0.0,
                message=f"Starting experiment for hyper-parameters combination \"{config_name}\"."
            )

            ### Data samplers
            samplers = self.create_samplers(data)

            for trail_id, (train_sampler, val_sampler) in enumerate(samplers):

                history_path = os.path.join(HISTORIES_PATH, f'history_{config_name}_{trail_id}.csv')
                model_path = os.path.join(MODELS_DIR, f'model_{config_name}_{trail_id}.pt')
                result_path = os.path.join(RESULTS_DIR, f'predictions_{config_name}_{trail_id}.pt')

                if os.path.exists(result_path):
                    continue

                self.send_info_event(
                    progress=(exp_id * self.config.num_splits + trail_id + 1) / total,
                    message=f"Starting trial {trail_id} of experiment \"{config_name}\""
                )

                trial_history, model = self.run_experiment(
                    config=config,
                    data=data,
                    train_loader=train_sampler,
                    val_loader=val_sampler
                )

                ### TODO: redefine triples membership to articles for triples that are added from
                ### external knowledge base.

                self.send_info_event(
                    progress=(exp_id * self.config.num_splits + trail_id + 1) / total,
                    message=f"Trial {trail_id} of experiment {config_name} completed."
                )

                trial_history['exp'] = config_name
                trial_history['trial'] = trail_id

                trial_history.to_csv(history_path)

                torch.save(model.state_dict(), f = model_path)

                self.send_info_event(
                    progress=(exp_id * self.config.num_splits + trail_id + 1) / total,
                    message=f"Making inference..."
                )

                predictions = self.inference(model, data)
                torch.save(predictions, f = result_path)

                self.send_info_event(
                    progress=(exp_id * self.config.num_splits + trail_id + 1) / total,
                    message=f"Model, inference results and history saved to disk."
                )

        self.send_info_event(
            progress=1.0,
            message=f"History saved to disk"
        )

        yield from []

    @classmethod
    def from_config(cls, config : Config) -> Self:

        trainer = Trainer(experiment=config.experiment, config=config.training_config)
        
        trainer \
            .add_handler(ConsoleLogger("Trainer"))

        return trainer