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
from pydantic import BaseModel, Field, computed_field, BeforeValidator
from typing import Any, Dict, List, Self, Tuple, Annotated, Union, Optional, Literal
from sklearn.model_selection import KFold
from torch import Tensor, optim
from dataclasses import asdict
from instance import cli
from torch_geometric.data import HeteroData
from src.shared.file_tools import Range, Enumerate
from src import helpers
from src.commands.feature_extraction.graph_utils import Graph, NodeTypes
from src.graph_learning.models import TopicModel, Context, IterationContext
from src.commands.graph_learning.preprocessing import (
    _add_metapaths, _augment, _norm_features, _add_labels,
    _select_nodes, _remove_self_loops, AugmentationConfig,
    _subgraph_by_articles, _merge_labels
)
from src.commands.feature_extraction.main import Config as FEConfig
from src.commands.graph_learning.sampler import HybridSampler
from src.shared.command_tools import BaseCommand, add_handlers, ConnectionOption, OutputDirOption
from src.shared.handlers import ConsoleLogger
from definitions import EXPERIMENTS_DIR, FEATURES_DIR
from rich.progress import Progress


# ---------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------- 

class Config(ConnectionOption, OutputDirOption):

    experiment: str = "default"
    configs : Optional[List[str]] = None

    def output_dir(self):
        return os.path.join(EXPERIMENTS_DIR, *self.experiment.split('/'))

    @computed_field
    @property
    def training_config(self) -> "TrainingConfig":

        config_path = os.path.join(self.output_dir(), "config.json")
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

    # Data
    graph: str = "ls2n-duke"  
    labels : List[Label] = Field(default_factory=list)     
    domain : Optional[str] = None       
    merges : List[List[str]] = Field(default_factory=list)   

    # Sampling
    batch_size: int = 8
    num_articles_neighbors: Tuple[int, int] = (10, 6)
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

    @computed_field
    @property
    def fe_config(self) -> FEConfig:
        return FEConfig(features_id=self.graph)

class ExperimentConfig(BaseTrainingConfig):
    experiment : str
    name : str

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

    def model_path(self, postfix : str = 'final') -> str:
        return os.path.join(self.models_dir, f"model_{self.name}_{postfix}.pt")

    def results_path(self, postfix : str = 'final') -> str:
        return os.path.join(self.results_dir, f"predictions_{self.name}_{postfix}.pt")

    def history_path(self, postfix : str = 'final') -> str:
        return os.path.join(self.histories_dir, f"history_{self.name}_{postfix}.csv")

    def load_data(self, return_raw : bool = False) -> Tuple[HeteroData, Graph] | HeteroData:

        graph_path = os.path.join(FEATURES_DIR, self.graph)

        graph = Graph.load(graph_path)

        G = graph.pygeo

        G = _add_metapaths(G, types=['Entity'])
        G = _add_labels(G, ['Lab'])
        G = _select_nodes(G, ['Article', 'Entity'])
        G = _norm_features(G)

        if self.domain is not None:

            domains = list(graph._nodes['SubDomain']._id2idx.keys())
            labels  = G['Article']['llm:deepseek-chat']
            labels  = labels / labels.max()
            domain_id = domains.index(self.domain)
            assert domain_id != -1
            mask = labels[:,domain_id].gt(0.5)

            G = _subgraph_by_articles(
                data=G,
                article_ids=G['Article']['node_id'][mask],
                save_old_idx=True
            )

        G.raw_metadata = {
            **(G.raw_metadata if hasattr(G, 'raw_metadata') else {}),
            "labels" : {
                "subdomain" : [node.id for node in graph.get_nodes(NodeTypes.SubDomain).nodes_generator],
                "lab" : [node.id for node in graph.get_nodes(NodeTypes.Lab).nodes_generator]
            }
        }

        if len(self.merges) > 0:
            G = _merge_labels(data=G, label="lab", merges=self.merges)

        if return_raw:
            return G, graph
        
        return G

    def create_model(self, ctx : Context) -> TopicModel:

        model = TopicModel.create_model(
            name=self.model_name,
            config=self.model_params,
            ctx=ctx,
            labels=[l.name for l in self.labels]
        )

        return model

    def exists(self, postfix : str) -> bool:

        return os.path.exists(self.history_path(postfix)) and \
            os.path.exists(self.results_path(postfix)) and \
            os.path.exists(self.model_path(postfix))

    def load_model(self, ctx : Context, postfix : str = 'final') -> TopicModel:

        model_path = self.model_path(postfix=postfix)

        if not os.path.exists(model_path):

            raise Exception(
                f"Instance `{postfix}` of model `{self.experiment}/{self.name}` doesn't exist. \n"
                f"No such a file {model_path}."
            )

        model = self.create_model(ctx=ctx).cpu()
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

        print(model.load_state_dict(state_dict))

        return model

    def load_results(self, final_only : bool = True) -> Dict[str, Dict[str, Any]]:

        results = {}

        for filename in os.listdir(self.results_dir):

            parts = os.path.splitext(filename)[0].split('_')
            postfix = parts[-1]
            config_name = '_'.join(parts[1:-1])

            if config_name != self.name:
                continue

            if (final_only) and (postfix != 'final'):
                continue

            results_path = os.path.join(self.results_dir, filename)
            results[postfix] = torch.load(results_path, weights_only=False)

        return results

SplitType = Literal['k_fold', 'train_test', 'none']

class TrainingConfig(BaseTrainingConfig):
    experiment : str
    seed : int = 42
    num_splits: int = 4
    hparams: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {"default": {}})
    split : SplitType = 'none'

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

    def load_history(
        self,
        configs : Optional[List[str]] = None,
        include_final : bool = False
    ) -> pd.DataFrame:

        histories = []

        for filename in os.listdir(self.histories_dir):

            parts = os.path.splitext(filename)[0].split('_')
            postfix = parts[-1]
            config_name = '_'.join(parts[1:-1])

            if (not include_final) and (postfix == 'final'):
                continue

            if (configs is not None) and (config_name not in configs):
                continue

            history_path = os.path.join(self.histories_dir, filename)
            history = pd.read_csv(history_path)
            history['trial'] = int(postfix[-1]) if postfix != 'final' else None
            histories.append(history)

        return pd.concat(histories)

    def load_results(
        self,
        configs : Optional[List[str]] = None,
        final_only : bool = True
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:

        results = {}

        for filename in os.listdir(self.results_dir):

            parts = os.path.splitext(filename)[0].split('_')
            postfix = parts[-1]
            config_name = '_'.join(parts[1:-1])

            if (final_only) and (postfix != 'final'):
                continue

            if (configs is not None) and (config_name not in configs):
                continue

            results_path = os.path.join(self.results_dir, filename)
            results[(config_name, postfix)] = torch.load(results_path, weights_only=False)

        return results

# ---------------------------------------------------------------------------
# COMMAND
# ---------------------------------------------------------------------------

@cli.register_stage(
    name="graph_learning",
    description=(
        "Trains the Knowledge Representation Learning module based on the "
        "configuration specified in the experiment path."
    ),
    add_from_conf=False,
    add_save_to=False,
)
@add_handlers
class Trainer(BaseCommand):

    def __init__(self, 
        experiment: str, 
        config: TrainingConfig,
        configs : Optional[List[str]]
    ) -> None:
        
        super().__init__()
        self.experiment = experiment
        self.config = config
        self.configs = configs if configs is not None else list(config.configs.keys())
        self.progress = 0.0

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
                progress=self.progress,
                message=f"[Split] Loading cached split from {split_path}",
            )

            return torch.load(split_path, map_location="cpu", weights_only=False)

        self.send_debug_event(
            progress=self.progress,
            message=f"[Split] Creating new {self.config.num_splits}-fold split …",
        )

        kf = KFold(n_splits=self.config.num_splits, shuffle=True)
        n  = data['Article'].x.size(0)
        split_tensor = torch.full((n,), -1, dtype=torch.long)

        for fold, (_, val_idx) in enumerate(kf.split(range(n))):
            split_tensor[val_idx] = fold

        torch.save(split_tensor, split_path)

        self.send_info_event(
            progress=self.progress,
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

    def create_samplers(
        self, 
        data: HeteroData, 
        config : ExperimentConfig
    ) -> List[Tuple[HybridSampler, HybridSampler]]:
        
        split_tensor = self.create_or_load_split(data)
        samplers: List[Tuple[HybridSampler, HybridSampler]] = []

        for fold in range(self.config.num_splits):

            train_mask = split_tensor != fold
            val_mask   = split_tensor == fold

            self.send_debug_event(
                progress=self.progress,
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

        return samplers

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

        model = config.create_model(ctx=ctx)

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
        config: ExperimentConfig,
        data: HeteroData,
        train_loader: HybridSampler,
        val_loader: HybridSampler,
        label: str,                # e.g. "trial 2/4" or "full graph"
    ) -> Tuple[pd.DataFrame, TopicModel]:
        """
        Runs a complete training loop.

        ``label``            is shown in log messages to identify the run.
        ``total_steps``      is the denominator used for progress values.
        ``completed_steps``  is the number of steps finished before this call.
        """

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

        epochs_it = self.track(
            it=Range(config.num_epochs),
            task=label,
            log_message=False
        )

        for epoch in epochs_it:

            epoch_rows: List[Dict[str, Any]] = []

            for phase in ("train", "val"):

                is_training = phase == "train"
                model.train(mode=is_training)
                loader = loaders[phase]
                
                ### Progress
                it = self.track(
                    it=Enumerate(loader),
                    task=f"{label}::{phase.upper()}::{epoch+1}/{config.num_epochs}",
                    log_message=False
                )

                for i, batch in it:

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
                        **{k: v.item() for k, v in losses.items()},
                    })

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
                        progress=self.progress,
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
                    progress=self.progress,
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
        result : TopicModel.Output,
        postfix : str
    ) -> None:
        torch.save(model.state_dict(), config.model_path(postfix))
        torch.save(result, config.results_path(postfix))
        history['config'] = config.name
        history.to_csv(config.history_path(postfix), index=False)

    def samplers(self, 
        data : HeteroData, 
        config : ExperimentConfig,
        split : SplitType
    ) -> Dict[str, Tuple[HybridSampler, HybridSampler]]:

        full_sampler = self.create_full_sampler(data=data, config=config)

        if split == 'none':
            return { "final" : (full_sampler, full_sampler) }

        samplers = self.create_samplers(data=data, config=config)

        if split == 'train_test':

            return {
                'train_test' : samplers[0],
                'final' : (full_sampler, full_sampler)
            }

        return {
            **{
                f'trial{i}' : sampler_pair
                for i, sampler_pair in enumerate(samplers)
            },
            'final' : (full_sampler, full_sampler)
        }
        
    def __iter__(self):

        helpers.seed_everything(self.config.seed)

        configs = self.config.configs

        self.send_info_event(
            progress=self.progress,
            message=f"[Config] {len(configs)} configuration(s) loaded: {list(configs.keys())}",
        )

        os.makedirs(self.config.models_dir,    exist_ok=True)
        os.makedirs(self.config.results_dir,   exist_ok=True)
        os.makedirs(self.config.histories_dir, exist_ok=True)

        n_configs = len(configs)

        total_steps = sum([
            1 + (
                (self.config.split == 'k_fold') * self.config.num_splits +
                (self.config.split == 'train_test') * 1
            )
            for _ in configs.values()
        ])

        completed   = 0

        for config_idx, config_name in enumerate(self.configs):

            config = configs[config_name]

            ### Load Data
            self.progress = completed / total_steps
            data = config.load_data()

            ### Samplers
            samplers = self.samplers(data=data, config=config, split=self.config.split)

            for postfix, (train_sampler, val_sampler) in samplers.items():

                if (
                    os.path.exists(config.history_path(postfix)) and 
                    os.path.exists(config.results_path(postfix)) and
                    os.path.exists(config.model_path(postfix))
                ):
                    continue

                self.progress = completed / total_steps
                label   = f"config '{config_name}' | {postfix}"

                if postfix == 'final':
                
                    self.send_info_event(
                        progress=self.progress,
                        message=f"[config '{config_name}'] All {self.config.num_splits} trials done — "
                                f"retraining on full graph …",
                    )

                else:

                    self.send_info_event(
                        progress=self.progress,
                        message=f"[{label}] Starting  "
                                f"({config_idx + 1}/{n_configs} configurations) "
                    )

                # ----------------------------------------------------------
                # Training
                # ----------------------------------------------------------

                history, model = self.run_experiment(
                    config=config,
                    data=data,
                    train_loader=train_sampler,
                    val_loader=val_sampler, 
                    label=label,
                )

                # Full graph inference
                result = self.inference(model, data)

                # Save Results
                self.save_output(
                    config, 
                    model=model, 
                    history=history, 
                    result=result,
                    postfix=postfix
                )

                self.send_info_event(
                    progress=self.progress,
                    message=f"[{label}] Model saved → {config.model_path(postfix)}",
                )

                self.send_info_event(
                    progress=self.progress,
                    message=f"[{label}] History saved → {config.history_path(postfix)}",
                )

                self.send_info_event(
                    progress=self.progress,
                    message=f"[{label}] Predictions saved → {config.results_path(postfix)}",
                )

                del result
                del model

                if postfix == "final":

                    self.send_info_event(
                        progress=self.progress,
                        message=f"[Config '{config_name}'] Completed",
                    )

                else:

                    self.send_info_event(
                        progress=self.progress,
                        message=f"[{label}] Completed",
                    )

                completed += 1

        self.send_info_event(
            progress=self.progress,
            message="===== All experiments completed =====",
        )

        yield from []

    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Config) -> Self:
        trainer = cls(experiment=config.experiment, config=config.training_config, configs=config.configs)
        trainer.add_handler(ConsoleLogger("Trainer"))
        return trainer