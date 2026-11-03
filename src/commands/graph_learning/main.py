"""
GNN Training Pipeline
=====================
Trains the Knowledge Representation Learning module using K-Fold cross-validation.

For each hyper-parameter configuration:
  • Run K trials (one per fold), saving per-trial history + predictions.
  • After all K trials complete, retrain on the full graph and save that
    model as the canonical artefact for the configuration.

Only the full-graph model is saved as a .pt checkpoint.
Per-trial histories (.csv) and predictions (.pt) are always preserved.

Entry-point (CLI):
    cli run train --experiment default
"""

import os
import torch
import pandas as pd
import pydantic
from pydantic import BaseModel, Field, computed_field, BeforeValidator, AfterValidator
from typing import Any, Dict, List, Self, Tuple, Annotated, Type, Union, Optional
from sklearn.model_selection import KFold
from torch import Tensor, optim
from torch.nn import functional as F
from torch_geometric import utils as gnn_utils
from dataclasses import asdict
from instance import cli
from torch_geometric.data import HeteroData
from src import helpers
from src.commands.feature_extraction.graph_utils import Graph
from src.graph_dl.models import TopicModel, Context, IterationContext
from src.commands.graph_learning.preprocessing import (
    _add_metapaths, _augment, _norm_features, 
    _select_nodes, _remove_self_loops, AugmentationConfig
)
from src.commands.graph_learning.sampler import HybridSampler, _subgraph_by_articles
from src.shared.command_tools import BaseCommand
from src.shared.handlers import ConsoleLogger
from definitions import EXPERIMENTS_DIR, FEATURES_DIR
from rich.progress import Progress
from sklearn.metrics import f1_score, accuracy_score


# ---------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------- 

class Config(BaseModel):

    experiment: str = "default"
    configs : Optional[List[str]] = None

    @computed_field
    @property
    def training_config(self) -> "TrainingConfig":

        config_path = os.path.join(EXPERIMENTS_DIR, self.experiment, "config.json")
        conf = helpers.load_json_file(config_path)

        return TrainingConfig(**{
            "experiment" : self.experiment,
            **conf
        })

class Label(BaseModel):
    name : str
    bounds : Annotated[Optional[Tuple[float, float]], BeforeValidator(func=lambda x : x if x is None else tuple(x))]
    alpha : float

class LossConfig(BaseModel):
    weight : float = 1.0
    params : Dict[str, Any] = Field(default_factory=dict)

class BaseTrainingConfig(BaseModel):

    # Seed
    seed : int = 42

    # Data
    graph: str = "ls2n-duke"  
    labels : List[Label] = Field(default_factory=lambda : [])               

    # Sampling
    batch_size: int = 8
    num_articles_neighbors: Tuple[int, int] = (10, 6)
    domain : Optional[str] = None
    full_batch_training : bool = False

    # Model
    model_name : str = 'ConstrastiveModel'
    model_params: Union[Dict, BaseModel] = Field(default_factory=dict)

    # Hyper-parameters
    lr: float = 0.00075
    num_epochs: int = 10
    accumulation_steps: int = 1
    weight_decay: float = 1e-4

    # Augmentation
    augmentation : Optional[AugmentationConfig] = None

    # Losses
    losses : Dict[str, LossConfig] = Field(default_factory=dict)

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_every: int = 1

    @pydantic.model_validator(mode="after")
    def validate_model_params(self):
        TopicModel.validate_model_config(self.model_name, self.model_params)
        return self

class ExperimentConfig(BaseTrainingConfig):
    experiment : str
    name : str

    @computed_field
    @property
    def model_path(self) -> str:
        return os.path.join(EXPERIMENTS_DIR, self.experiment, "models", f"model_{self.name}.pt")

    @computed_field
    @property
    def results_path(self) -> str:
        return os.path.join(EXPERIMENTS_DIR, self.experiment, "results", f"predictions_{self.name}.pt")

    @computed_field
    @property
    def history_path(self) -> str:
        return os.path.join(EXPERIMENTS_DIR, self.experiment, "histories", f"history_{self.name}.csv")

class TrainingConfig(BaseTrainingConfig):
    experiment : str
    num_splits: int = 4
    hparams: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {"default": {}})

    @computed_field
    @property
    def models_dir(self) -> str:
        return os.path.join(EXPERIMENTS_DIR, self.experiment, "models")

    @computed_field
    @property
    def results_dir(self) -> str:
        return os.path.join(EXPERIMENTS_DIR, self.experiment, "results")

    @computed_field
    @property
    def histories_dir(self) -> str:
        return os.path.join(EXPERIMENTS_DIR, self.experiment, "histories")

    @computed_field
    @property
    def configs(self) -> Dict[str, ExperimentConfig]:
    
        configs: Dict[str, ExperimentConfig] = {}

        for exp_name, hparam in self.hparams.items():
            base = self.model_dump(exclude_computed_fields=True)
            base.pop("hparams")
            configs[exp_name] = ExperimentConfig(**
                {
                    "experiment" : self.experiment,
                    "name" : exp_name,
                    **helpers.override_dict(base, hparam)
                }
            )

        return configs

# ---------------------------------------------------------------------------
# COMMAND
# ---------------------------------------------------------------------------

@cli.register_stage(
    name="train_ssl",
    description=(
        "Trains the Knowledge Representation Learning module based on the "
        "configuration specified in the experiment path."
    ),
    add_from_conf=False,
    add_save_to=False,
)
class Trainer(BaseCommand):

    def __init__(self, 
        experiment: str, 
        config: TrainingConfig,
        configs : Optional[List[str]]
    ) -> None:
        super().__init__()
        self.experiment = experiment
        self.config = config
        self.configs = configs if configs is not None else config.configs

        print(f"len(self.configs) = {len(self.configs)}")

    # ------------------------------------------------------------------
    # Data loading  (new graph.pt format — no filtering)
    # ------------------------------------------------------------------

    def load_data(self, 
        config : ExperimentConfig, 
        return_raw : bool = False,
    ) -> Tuple[HeteroData, Graph] | HeteroData:

        graph_path = os.path.join(FEATURES_DIR, config.graph)

        self.send_info_event(
            progress=None,
            message=f"[Data] Loading graph from: {graph_path}",
        )

        graph = Graph.load(graph_path)

        G = graph.pygeo

        G = _add_metapaths(G, types=['Entity']) # ['Author', 'Keyword']
        # G = _add_labels(G, ['SubDomain', 'Lab'])
        G = _select_nodes(G, ['Article', 'Entity'])
        G = _norm_features(G)

        if config.domain is not None:

            domains = list(graph._nodes['SubDomain']._id2idx.keys())
            labels  = G['Article']['llm:deepseek-chat']
            labels  = labels / labels.max()
            domain_id = domains.index(config.domain)
            assert domain_id != -1
            mask = labels[:,domain_id].gt(0.5)

            G = _subgraph_by_articles(
                data=G,
                article_ids=G['Article']['node_id'][mask],
                save_old_idx=True
            )

        

        G.raw_metadata = {
            # "predicates2id" : G.predicates2id,
            "labels" : {
                "subdomain" : [node.id for node in graph.get_nodes("SubDomain").nodes_generator],
                "lab" : [node.id for node in graph.get_nodes("Lab").nodes_generator]
            }
        }

        if return_raw:
            return G, graph

        ### Processing Labels
        # for label in config.labels:

        #     name = label.name
        #     alpha = label.alpha

        #     X = G['Article'][name]
        #     X = (X - X.min()) / (X.max() - X.min() + 1e-7)

        #     if label.bounds is not None:

        #         lower_bound, upper_bound = label.bounds

        #         upper_mask = X.gt(upper_bound)
        #         lower_mask = X.lt(lower_bound)

        #         X[upper_mask] = 1 - torch.pow(1 - X[upper_mask], torch.scalar_tensor(alpha))
        #         X[lower_mask] = torch.pow(X[lower_mask], torch.scalar_tensor(alpha))

        #         X[(~lower_mask) & (~upper_mask)] = -1

        #     else:

        #         X = torch.where(
        #             X < 0.5,
        #             torch.pow(X[lower_mask], torch.scalar_tensor(alpha)),
        #             1 - torch.pow(1 - X[upper_mask], torch.scalar_tensor(alpha))
        #         )

        #     G['Article'][name] = X
        
        return G

    # ------------------------------------------------------------------
    # Splits & samplers
    # ------------------------------------------------------------------

    def create_or_load_split(self, data: HeteroData) -> Tensor:

        graph_slug = self.config.graph

        split_path = os.path.join(
            EXPERIMENTS_DIR, self.experiment,
            f"split_{graph_slug}_{self.config.num_splits}.pt",
        )

        if os.path.exists(split_path):

            self.send_debug_event(
                progress=None,
                message=f"[Split] Loading cached split from {split_path}",
            )

            return torch.load(split_path, map_location="cpu", weights_only=False)

        self.send_debug_event(
            progress=None,
            message=f"[Split] Creating new {self.config.num_splits}-fold split …",
        )

        kf = KFold(n_splits=self.config.num_splits, shuffle=True)
        n  = data['Article'].x.size(0)
        split_tensor = torch.full((n,), -1, dtype=torch.long)

        for fold, (_, val_idx) in enumerate(kf.split(range(n))):
            split_tensor[val_idx] = fold

        torch.save(split_tensor, split_path)

        self.send_info_event(
            progress=None,
            message=f"[Split] {self.config.num_splits}-fold split saved to {split_path}",
        )

        return split_tensor

    def _make_sampler(self, 
        split: HeteroData, 
        config : ExperimentConfig,
        is_train: bool
    ) -> HybridSampler:

        if not config.full_batch_training:

            return HybridSampler(
                data=split,
                num_neighbors=config.num_articles_neighbors,
                batch_size=config.batch_size,
                shuffle=is_train,
                drop_last=is_train
            )

        else:

            return HybridSampler(
                data=split,
                num_neighbors=[0, 0],
                batch_size=split['Article'].node_id.size(0),
                shuffle=is_train,
                drop_last=is_train,
                repeats=(split['Article'].node_id.size(0) // config.batch_size)
            )

    def create_samplers(self, data: HeteroData, config : ExperimentConfig) -> Tuple[List[Tuple[HybridSampler, HybridSampler]], Tensor]:
        
        split_tensor = self.create_or_load_split(data)
        samplers: List[Tuple[HybridSampler, HybridSampler]] = []

        for fold in range(self.config.num_splits):

            train_mask = split_tensor != fold
            val_mask   = split_tensor == fold

            self.send_debug_event(
                progress=None,
                message=f"[Split] Fold {fold}: "
                        f"train={train_mask.sum().item():,}  "
                        f"val={val_mask.sum().item():,}",
            )

            train_split = _subgraph_by_articles(data, data['Article'].node_id[train_mask])
            val_split   = _subgraph_by_articles(data, data['Article'].node_id[val_mask])

            samplers.append((
                self._make_sampler(train_split, is_train=True, config=config),
                self._make_sampler(val_split,   is_train=False, config=config),
            ))

        return samplers, split_tensor

    def create_full_sampler(self, data: HeteroData, config : ExperimentConfig) -> HybridSampler:
        """Sampler over the entire graph (used for final retraining)."""
        sampler = self._make_sampler(data, is_train=True, config=config)
        return sampler

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def create_model(self, 
        config: ExperimentConfig, 
        ctx : Context
    ) -> TopicModel:

        self.send_info_event(progress=0.0, message="[Model] Instantiating GNN model …")

        model = TopicModel.create_model(
            name=config.model_name,
            config=config.model_params,
            ctx=ctx,
            labels=[l.name for l in config.labels]
        )

        self.send_info_event(
            progress=0.0,
            message=f"[Model] Size: {helpers.model_size_mb(model):.3f} MB  |  "
                    f"Parameters: {sum(p.numel() for p in model.parameters()):,}",
        )

        return model

    # ------------------------------------------------------------------
    # Loss & metrics
    # ------------------------------------------------------------------

    @staticmethod
    def sample_negative_edges(
        edge_index: Tensor, 
        edge_type: Tensor, 
        num_nodes: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        
        neg_ei  = gnn_utils.negative_sampling(edge_index=edge_index, num_nodes=num_nodes)
        new_ei  = torch.cat([neg_ei, edge_index], dim=-1)

        y_hat   = torch.cat([
            torch.zeros_like(edge_index[0]),
            torch.ones_like(edge_index[1]),
        ]).float()

        etype   = torch.cat([edge_type, edge_type], dim=0)

        return new_ei, etype, y_hat

    @staticmethod
    def negative_sample(y: Tensor) -> Tuple[Tensor, Tensor]:
        pos = torch.argwhere(y).t()
        neg = gnn_utils.negative_sampling(edge_index=pos, num_nodes=(y.size(0), y.size(1)))
        row, col = torch.cat([pos, neg], dim=-1)
        return row, col
    
    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def loss(
        batch : HeteroData,
        config : ExperimentConfig,
        model_output : TopicModel.Output, 
        batch_size : int,
    ) -> Dict[str, Tensor]:
            
        ### Losses dictionary
        losses = {}
        losses['loss'] = torch.scalar_tensor(0.0).to(batch['Article'].x.device)

        ctx = model_output.context(batch=batch, batch_size=batch_size)
        
        for loss_name, loss_config in config.losses.items():

            _losses = ctx[loss_name].loss(params=loss_config.params)

            if isinstance(_losses, Tensor):
                losses[loss_name] = _losses
            else:
                losses = {
                    **losses,
                    **{
                        f'{loss_name}__{k}' if k != 'loss' else loss_name : v
                        for k, v in _losses.items()
                        if k != 'loss'
                    },
                    loss_name : _losses['loss']
                }

            if loss_name in losses:
                losses["loss"] += loss_config.weight * losses[loss_name]

        return losses

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        config_name : str,
        config: ExperimentConfig,
        data: HeteroData,
        train_loader: HybridSampler,
        val_loader: HybridSampler,
        label: str = "",                # e.g. "trial 2/4" or "full graph"
        total_steps: int = 1,           # for progress denominator
        completed_steps: int = 0,       # steps already done before this run
    ) -> Tuple[pd.DataFrame, TopicModel]:
        """
        Runs a complete training loop.

        ``label``            is shown in log messages to identify the run.
        ``total_steps``      is the denominator used for progress values.
        ``completed_steps``  is the number of steps finished before this call.
        """

        total_iterations = len(train_loader) * config.num_epochs

        model     = self.create_model(
            config=config, 
            ctx=Context(data=data)
        ).to(config.device)

        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config.num_epochs, eta_min=1e-2 * config.lr
        )

        loaders = {"train": train_loader, "val": val_loader}
        history: List[pd.DataFrame] = []

        pbar = Progress()
        pbar.start()

        config_task = pbar.add_task(description=f"Experiment : {config_name}", total=1.0)

        for epoch in range(config.num_epochs):

            epoch_rows: List[Dict[str, Any]] = []

            for phase in ("train", "val"):

                is_training = phase == "train"
                model.train(mode=is_training)
                loader = loaders[phase]
                
                ### Progress
                task_name = f"{config_name}::{phase.upper()}::{epoch+1}/{config.num_epochs}"
                epoch_task = pbar.add_task(description=task_name, total=1.0)

                for i, batch in enumerate(loader):

                    batch = _remove_self_loops(batch)

                    if config.augmentation is not None:
                        batch = _augment(batch=batch, config=config.augmentation, training=is_training)
                    
                    batch = helpers.move_data_to_device(batch, device=config.device)

                    with torch.set_grad_enabled(is_training):

                        iter_ctx = IterationContext(
                            num_epochs=config.num_epochs,
                            epoch=epoch+1,
                            iterations_per_epoch=len(loader),
                            iteration=i+1,
                            training=is_training
                        )

                        results   = model.forward(batch, ctx=iter_ctx)

                        losses = self.loss(
                            batch, 
                            config, 
                            results, 
                            config.batch_size
                        )

                    if is_training:
                        optimizer.zero_grad()
                        losses["loss"].backward()
                        optimizer.step()

                    epoch_rows.append({
                        "epoch": epoch,
                        "phase": phase,
                        "grad_min" : min(param.min().abs().item() for param_name, param in model.named_parameters()),
                        "grad_max" : max(param.max().abs().item() for param_name, param in model.named_parameters()),
                        "grad_mean" : sum(param.mean().abs().item() for param_name, param in model.named_parameters()) / len(list((model.named_parameters()))),
                        **{k: v.item() for k, v in losses.items()},
                    })

                    pbar.update(epoch_task, advance=1/len(loader), refresh=True)

            pbar.update(config_task, advance=1/config.num_epochs, refresh=True)

            scheduler.step()

            epoch_df = pd.DataFrame(epoch_rows)
            history.append(epoch_df)

            # ── Epoch summary logging ─────────────────────────────────
            if (epoch + 1) % config.log_every == 0:

                summary = (
                    epoch_df
                    .drop(columns=["epoch"])
                    .groupby("phase")
                    .mean()
                )

                for phase in ("train", "val"):

                    if phase not in summary.index:
                        continue

                    row = summary.loc[phase]

                    metrics = {
                        **row.to_dict()
                    }

                    self.send_info_event(
                        progress=None,
                        message=" ".join([
                            f"[{label}] Epoch {epoch + 1}/{config.num_epochs}",
                            f"[{phase.upper():5s}]",
                            *[
                                f"{key}={value:.4f}"
                                for key, value in metrics.items()
                            ],
                        ]),
                    )

                self.send_debug_event(
                    progress=None,
                    message=f"[{label}] Epoch {epoch + 1}/{config.num_epochs}  "
                            f"lr={scheduler.get_last_lr()[0]:.6f}",
                )

        return pd.concat(history, ignore_index=True), model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @staticmethod
    def inference(
        model: TopicModel, 
        data: HeteroData,
    ) -> Dict[str, Tensor]:

        model = model.eval().cpu()
        data  = data.cpu()

        with torch.no_grad():
            results = model.forward(data)  
        
        return asdict(results)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def save_output(
        self,
        config : ExperimentConfig,
        model : TopicModel,
        history : pd.DataFrame,
        result : TopicModel.Output
    ) -> None:
        torch.save(model.state_dict(), config.model_path)
        torch.save(result, config.results_path)
        history['config'] = config.name
        history.to_csv(config.history_path, index=False)
        
    def __iter__(self):

        helpers.seed_everything(self.config.seed)

        configs = self.config.configs

        self.send_info_event(
            progress=None,
            message=f"[Config] {len(configs)} configuration(s) loaded: {list(configs.keys())}",
        )

        os.makedirs(self.config.models_dir,    exist_ok=True)
        os.makedirs(self.config.results_dir,   exist_ok=True)
        os.makedirs(self.config.histories_dir, exist_ok=True)

        n_configs = len(configs)
        total_steps = n_configs # * (self.config.num_splits + 1)
        completed   = 0

        for config_idx, (config_name, config) in enumerate(configs.items()):

            if config_name not in self.configs:
                continue

            if (
                os.path.exists(config.history_path) and 
                os.path.exists(config.results_path) and
                os.path.exists(config.model_path)
            ):
                continue

            data    = self.load_data(config=config)

            self.send_info_event(
                progress=completed / total_steps,
                message=f"[Config '{config_name}'] Starting  "
                        f"({config_idx + 1}/{n_configs} configurations) "
            )

            # ----------------------------------------------------------
            # Full-graph retrain  (only this model is saved as .pt)
            # ----------------------------------------------------------

            full_label     = f"config '{config_name}' | full retrain"

            self.send_info_event(
                progress=completed / total_steps,
                message=f"[{full_label}] All {self.config.num_splits} trials done — "
                        f"retraining on full graph …",
            )

            full_sampler = self.create_full_sampler(data, config=config)

            # For the full retrain we still track a val signal using the full
            # loader (same data) so the run_experiment interface is satisfied.
            history, final_model = self.run_experiment(
                config_name=config_name,
                config=config,
                data=data,
                train_loader=full_sampler,
                val_loader=full_sampler,      # no held-out split
                label=full_label,
                total_steps=total_steps,
                completed_steps=completed,
            )

            # Full graph inference
            result = self.inference(final_model, data)

            # Save Results
            self.save_output(
                config, 
                model=final_model, 
                history=history, 
                result=result
            )

            completed += 1

            self.send_info_event(
                progress=completed / total_steps,
                message=f"[{full_label}] Final model saved → {config.model_path}",
            )

            # Save trial history
            step_label  = f"config '{config_name}'"
            self.send_info_event(
                progress=completed / total_steps,
                message=f"[{step_label}] History saved → {config.history_path}",
            )

            self.send_info_event(
                progress=completed / total_steps,
                message=f"[{step_label}] Predictions saved → {config.results_path}",
            )

            del result
            del final_model

            self.send_info_event(
                progress=completed / total_steps,
                message=f"[Config '{config_name}'] Completed",
            )

        self.send_info_event(
            progress=1.0,
            message="===== All experiments completed =====",
        )

        yield from []

    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Config) -> Self:
        trainer = cls(experiment=config.experiment, config=config.training_config, configs=config.configs)
        trainer.add_handler(ConsoleLogger("Trainer"))
        return trainer