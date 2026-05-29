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
import pandas as pd
import torch
from rich.progress import Progress as RichProgress
from typing import Any, Dict, List, Optional, Self, Tuple
from pydantic import BaseModel, Field, computed_field
from sklearn.model_selection import KFold
from torch import Tensor, optim
from torch.nn import functional as F
from torch_geometric import utils as gnn_utils
from torch_sparse import SparseTensor
from dataclasses import asdict
from definitions import EXPERIMENTS_DIR, FEATURES_DIR
from instance import cli
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData, Data
from src import helpers
from src.commands.feature_extraction import Graph
from src.gnn import (
    HetGNNConfigV2 as HetGNNConfig,
    HetModelV2 as HetModel,
    HetModelReturnV2 as HetModelReturn,
)
from src.shared.command_tools import BaseCommand
from src.shared.handlers import ConsoleLogger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config(BaseModel):

    experiment: str = "default"

    @computed_field
    @property
    def training_config(self) -> "TrainingConfig":
        config_path = os.path.join(EXPERIMENTS_DIR, self.experiment, "config.json")
        conf = helpers.load_json_file(config_path)
        return TrainingConfig(**conf)


class BaseTrainingConfig(BaseModel):

    # Data
    graph: str = "ls2n-duke"  
    labels : List[str] = Field(default_factory=lambda : ['subdomain', 'lab'])               

    # Sampling
    batch_size: int = 8
    num_articles_neighbors: tuple[int, int] = (10, 6)
    num_entities_neighbors: tuple[int, int] = (10, 6)

    # Model
    model_params: HetGNNConfig = Field(default_factory=HetGNNConfig)

    # Hyper-parameters
    lr: float = 0.00075
    num_epochs: int = 10
    accumulation_steps: int = 1
    weight_decay: float = 1e-4

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_every: int = 1


class TrainingConfig(BaseTrainingConfig):
    seed: int = 56
    num_splits: int = 4
    hparams: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {"default": {}})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def redundancy_reduction_loss(
    z1 : Tensor, 
    z2 : Tensor, 
    lambd : float = 0.005, 
    eps : float = 1e-5,
    max_entities : int = 512
) -> Tensor:
    """
    Barlow Twins-style redundancy reduction loss.

    Args:
        z1: Tensor [B, D] (view 1 embeddings)
        z2: Tensor [B, D] (view 2 embeddings)
        lambd: weight for off-diagonal penalty
        eps: numerical stability

    Returns:
        scalar loss
    """

    if z1.shape[0] >= max_entities:
        perm = torch.randperm(z1.shape[0])
        perm = perm[:max_entities]
        z1 = z1[perm,:]
        z2 = z2[perm,:]

    # 1. Normalize each dimension (important)
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)

    z1 = z1 / (z1.std(dim=0) + eps)
    z2 = z2 / (z2.std(dim=0) + eps)

    # 2. Cross-correlation matrix
    # shape: [D, D]
    c = torch.matmul(z1.T, z2) / z1.shape[0]

    # 3. On-diagonal loss (invariance)
    on_diag = torch.diagonal(c).add(-1).pow(2).sum()

    # 4. Off-diagonal loss (redundancy reduction)
    off_diag = c.fill_diagonal_(0).pow(2).sum()

    loss = on_diag + lambd * off_diag

    return loss

def clip_loss(za : Tensor, ze : Tensor, temperature : float = 0.1):
    """
    za: article graph embeddings [B, D]
    ze: entity-derived article embeddings [B, D]
    """

    # normalize embeddings
    za = F.normalize(za, dim=1)
    ze = F.normalize(ze, dim=1)

    # similarity matrix
    logits = za @ ze.T

    # temperature scaling
    logits = logits / temperature

    # positives are diagonal entries
    labels = torch.arange(za.size(0), device=za.device)

    # symmetric InfoNCE
    loss_a2e = F.cross_entropy(logits, labels)
    loss_e2a = F.cross_entropy(logits.T, labels)

    loss = (loss_a2e + loss_e2a) / 2

    return loss

def loss(
    batch : HeteroData,
    model_output : HetModelReturn, 
    batch_size : int,
    temperature : float = 0.1
) -> Dict[str, Tensor]:

    losses = {}
    
    losses['l1'] = clip_loss(
        model_output.articles_embeddings_v1[:batch_size],
        model_output.articles_embeddings_v2[:batch_size],
        temperature=temperature
    )

    losses['loss'] = losses['l1']

    return losses

def _augment(batch : HeteroData, rate : float = 0.1, training : bool = True) -> HeteroData:

    if not training:
        return batch

    batch['Entity'].x = gnn_utils.mask_feature(
        batch['Entity'].x,
        p=rate,
        mode='row'
    )

    # batch['Article'].x = gnn_utils.mask_feature(
    #     batch['Article'].x,
    #     p=rate,
    #     mode='row'
    # )

    return batch

def _subgraph_by_articles(data : HeteroData, article_ids : torch.Tensor) -> HeteroData:

    entities, articles = data['Entity', 'APPEARIN', 'Article'].edge_index

    entity_ids = torch.unique(entities[torch.isin(articles, article_ids)])

    subset = data.subgraph({
        "Article" : article_ids,
        "Entity" : entity_ids
    })

    for n_type in subset.node_types:
        subset[n_type].node_id = torch.arange(len(subset[n_type].x))

    for e_type in subset.edge_types:
        subset[e_type].edge_id = torch.arange(len(subset[e_type].edge_index[0]))

    return subset

def _select_nodes(data: HeteroData, n_types: List[str]) -> HeteroData:

    G = HeteroData()

    for n_type in n_types:
        for key, value in data[n_type].items():
            G[n_type][key] = value.clone() if torch.is_tensor(value) else value

    for e_type in data.edge_types:
        if e_type[0] in n_types and e_type[2] in n_types:
            for key, value in data[e_type].items():
                G[e_type][key] = value.clone() if torch.is_tensor(value) else value

    G.predicates2id = data.predicates2id

    return G

def _add_labels(data : HeteroData, n_types : List[str]) -> HeteroData:

    for type_ in n_types:

        e_type = [e for e in data.edge_types if e[0] == 'Article' and e[2] == type_][0]

        row, col = data[e_type].edge_index
        num_cols = data['Article'].num_nodes
        num_rows = data[type_].num_nodes

        A = SparseTensor(row=row,col=col,sparse_sizes=(num_cols,num_rows))

        data['Article'][f'y_{type_.lower()}'] = A.to_dense().clone()

    return data

def _add_metapaths(data : HeteroData, types : Optional[List[str]] = None) -> HeteroData:

    types = ['Author'] if types is None else types

    edge_index = []
    edge_type  = []

    for i, type_ in enumerate(types):

        e_type = [e for e in data.edge_types if e[0] == 'Article' and e[2] == type_][0]

        row, col = data[e_type].edge_index
        num_cols = data['Article'].num_nodes
        num_rows = data[type_].num_nodes

        A = SparseTensor(row=row,col=col,sparse_sizes=(num_cols,num_rows))
        A = A @ A.t()
        row, col, _ = A.coo()

        edge_index.append(torch.stack([row, col], dim=0))
        edge_type.append(torch.zeros_like(row).fill_(i))

    edge_index = torch.cat(edge_index, dim=-1)
    edge_type = torch.cat(edge_type, dim=-1)

    data['Article', 'relate-to', 'Article'].edge_index = edge_index
    data['Article', 'relate-to', 'Article'].edge_type = edge_type
    data['Article', 'relate-to', 'Article'].edge_id = torch.arange(len(row))

    return data

def _norm_features(data : HeteroData) -> HeteroData:

    for n_type in data.node_types:
        t = data[n_type].x
        mean = t.mean(0, keepdim=True)
        std  = t.std(0,  keepdim=True)
        data[n_type].x = (t - mean) / (std + 1e-3)

    for e_type in data.edge_types:
        if 'x' in data[e_type]:
            t = data[e_type].x
            mean = t.mean(0, keepdim=True)
            std  = t.std(0,  keepdim=True)
            data[e_type].x = (t - mean) / (std + 1e-3)

    return data

# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class HybridSampler:

    def __init__(
        self,
        data : HeteroData,
        num_neighbors : List[int],
        batch_size : int,
        shuffle : bool = True,
        **kwargs
    ) -> None:
        
        self.data = data
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._sampler = self._create_sampler(**kwargs)

    def _create_sampler(self, **kwargs) -> NeighborLoader:

        d = Data()

        d.x = self.data['Article'].x
        d.node_id = self.data['Article'].node_id

        d.edge_index = self.data['Article', 'relate-to', 'Article'].edge_index

        return NeighborLoader(
            data=d,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            **kwargs
        )

    def __iter__(self):
        
        for batch in self._sampler:
            article_ids = batch.node_id
            yield _subgraph_by_articles(self.data, article_ids)

    def __len__(self):
        return len(self._sampler)
    
# ---------------------------------------------------------------------------
# Trainer
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

    def __init__(self, experiment: str, config: TrainingConfig) -> None:
        super().__init__()
        self.experiment = experiment
        self.config = config

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def get_configs(self) -> Dict[str, BaseTrainingConfig]:

        self.send_info_event(
            progress=0.0,
            message="[Config] Loading all hyper-parameter combinations …",
        )

        configs: Dict[str, BaseTrainingConfig] = {}

        for exp_name, hparam in self.config.hparams.items():
            base = self.config.model_dump()
            base.pop("hparams")
            configs[exp_name] = BaseTrainingConfig(**helpers.override_dict(base, hparam))

        self.send_info_event(
            progress=0.0,
            message=f"[Config] {len(configs)} configuration(s) loaded: {list(configs.keys())}",
        )

        for name, cfg in configs.items():
            
            self.send_debug_event(
                progress=0.0,
                message=f"[Config] '{name}' — lr={cfg.lr}, epochs={cfg.num_epochs}, "
                        f"batch={cfg.batch_size}",
            )

        return configs

    # ------------------------------------------------------------------
    # Data loading  (new graph.pt format — no filtering)
    # ------------------------------------------------------------------

    def load_data(self, return_raw : bool = False) -> Tuple[HeteroData, Graph] | HeteroData:

        graph_path = os.path.join(FEATURES_DIR, self.config.graph)

        self.send_info_event(
            task="Data Loading",
            progress=0.0,
            message=f"[Data] Loading graph from: {graph_path}",
        )

        graph = Graph.load(graph_path)
        G = graph.pygeo

        self.send_info_event(
            task="Data Loading",
            progress=0.25,
            message=f"[Data] Adding Metapaths",
        )

        G = _add_metapaths(G, types=['Author', 'Keyword'])

        self.send_info_event(
            task="Data Loading",
            progress=0.5,
            message=f"[Data] Computing labels...",
        )

        G = _add_labels(G, ['SubDomain', 'Lab'])

        self.send_info_event(
            task="Data Loading",
            progress=0.75,
            message=f"[Data] Nodes selection",
        )

        G = _select_nodes(G, ['Article', 'Entity'])

        self.send_info_event(
            task="Data Loading",
            progress=1.0,
            message=f"[Data] Data normalization",
        )

        G = _norm_features(G)

        G.raw_metadata = {
            "predicates2id" : G.predicates2id,
            "labels" : {
                "subdomain" : [node.id for node in graph.get_nodes("SubDomain")],
                "lab" : [node.id for node in graph.get_nodes("Lab")]
            }
        }

        if return_raw:
            return G, graph
        
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

    def _make_sampler(self, split: HeteroData, is_train: bool) -> HybridSampler:

        return HybridSampler(
            data=split,
            num_neighbors=self.config.num_articles_neighbors,
            batch_size=self.config.batch_size,
            shuffle=is_train,
            drop_last=is_train
        )

    def create_samplers(self, data: HeteroData) -> Tuple[List[Tuple[HybridSampler, HybridSampler]], Tensor]:
        
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
                self._make_sampler(train_split, is_train=True),
                self._make_sampler(val_split,   is_train=False),
            ))

        return samplers, split_tensor

    def create_full_sampler(self, data: HeteroData) -> HybridSampler:
        """Sampler over the entire graph (used for final retraining)."""
        return self._make_sampler(data, is_train=True)

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def create_model(self, config: BaseTrainingConfig, data: HeteroData) -> HetModel:

        self.send_info_event(progress=0.0, message="[Model] Instantiating GNN model …")

        model = HetModel.from_config(
            in_channels=data['Entity'].x.size(-1),
            num_labels={
                label : data['Article'][f'y_{label}'].size(-1)
                for label in self.config.labels
            },
            num_relations=(len(data.predicates2id), len(torch.unique(data['Article','relate-to', 'Article'].edge_type))),
            config=config.model_params,
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
    # Training loop
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        config: BaseTrainingConfig,
        data: HeteroData,
        train_loader: HybridSampler,
        val_loader: HybridSampler,
        label: str = "",                # e.g. "trial 2/4" or "full graph"
        total_steps: int = 1,           # for progress denominator
        completed_steps: int = 0,       # steps already done before this run
    ) -> Tuple[pd.DataFrame, HetModel]:
        """
        Runs a complete training loop.

        ``label``            is shown in log messages to identify the run.
        ``total_steps``      is the denominator used for progress values.
        ``completed_steps``  is the number of steps finished before this call.
        """

        model     = self.create_model(config=config, data=data).to(config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config.num_epochs, eta_min=1e-2 * config.lr
        )

        loaders = {"train": train_loader, "val": val_loader}
        history: List[pd.DataFrame] = []

        for epoch in range(config.num_epochs):

            # Progress value advances linearly within this run
            epoch_progress = (completed_steps + epoch / config.num_epochs) / total_steps

            epoch_rows: List[Dict[str, Any]] = []

            for phase in ("train", "val"):

                is_training = phase == "train"
                model.train(mode=is_training)
                loader = loaders[phase]
                n_batches = len(loader)

                self.send_debug_event(
                    progress=epoch_progress,
                    message=f"[{label}] Epoch {epoch + 1}/{config.num_epochs} | "
                            f"{phase.upper()} — {n_batches} batches",
                )

                with RichProgress() as pbar:

                    task = pbar.add_task(
                        f"[blue]Epoch {epoch + 1}/{config.num_epochs} | {phase.upper()}",
                        total=n_batches,
                    )

                    for batch in loader:
                        
                        batch = helpers.move_data_to_device(batch, device=config.device)

                        with torch.set_grad_enabled(is_training):
                            results   = model.forward(batch)
                            losses = loss(batch, results, config.batch_size)

                        if is_training:
                            optimizer.zero_grad()
                            losses["loss"].backward()
                            optimizer.step()

                        epoch_rows.append({
                            "epoch": epoch,
                            "phase": phase,
                            **{k: v.item() for k, v in losses.items()},
                        })

                        pbar.advance(task)

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

                    row = summary.loc[phase] #.to_dict()

                    loss_val  = row.get("loss", float("nan"))

                    metrics = {
                        **row.to_dict()
                    }

                    self.send_info_event(
                        progress=epoch_progress,
                        message=" ".join([
                            f"[{label}] Epoch {epoch + 1}/{config.num_epochs}",
                            f"[{phase.upper():5s}]",
                            f"loss={loss_val:.4f}",
                            *[
                                f"{key}={value:.4f}"
                                for key, value in metrics.items()
                            ],
                        ]),
                    )

                self.send_debug_event(
                    progress=epoch_progress,
                    message=f"[{label}] Epoch {epoch + 1}/{config.num_epochs}  "
                            f"lr={scheduler.get_last_lr()[0]:.6f}",
                )

        return pd.concat(history, ignore_index=True), model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @staticmethod
    def inference(
        model: HetModel, 
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

    def __iter__(self):

        helpers.seed_everything(self.config.seed)

        configs = self.get_configs()
        data    = self.load_data()

        # print(data)

        MODELS_DIR    = os.path.join(EXPERIMENTS_DIR, self.experiment, "models")
        RESULTS_DIR   = os.path.join(EXPERIMENTS_DIR, self.experiment, "results")
        HISTORIES_DIR = os.path.join(EXPERIMENTS_DIR, self.experiment, "histories")

        os.makedirs(MODELS_DIR,    exist_ok=True)
        os.makedirs(RESULTS_DIR,   exist_ok=True)
        os.makedirs(HISTORIES_DIR, exist_ok=True)

        n_configs = len(configs)
        total_steps = n_configs * (self.config.num_splits + 1)
        completed   = 0

        for config_idx, (config_name, config) in enumerate(configs.items()):

            self.send_info_event(
                progress=completed / total_steps,
                message=f"[Config '{config_name}'] Starting  "
                        f"({config_idx + 1}/{n_configs} configurations)",
            )

            samplers, split_tensor = self.create_samplers(data)

            # ----------------------------------------------------------
            # K-Fold trials
            # ----------------------------------------------------------

            # for trial_id, (train_sampler, val_sampler) in enumerate(samplers):

            #     history_path = os.path.join(
            #         HISTORIES_DIR, 
            #         f"history_{config_name}_trial{trial_id}.csv"
            #     )

            #     result_path = os.path.join(
            #         RESULTS_DIR, 
            #         f"predictions_{config_name}_trial{trial_id}.pt"
            #     )

            #     step_label   = f"config '{config_name}' | trial {trial_id + 1}/{self.config.num_splits}"
            #     step_progress = completed / total_steps

            #     if os.path.exists(result_path) and os.path.exists(history_path):
                    
            #         self.send_info_event(
            #             progress=step_progress,
            #             message=f"[{step_label}] Already completed — skipping",
            #         )
            #         completed += 1
            #         continue

            #     self.send_info_event(
            #         progress=step_progress,
            #         message=f"[{step_label}] Starting trial …",
            #     )

            #     trial_history, model = self.run_experiment(
            #         config=config,
            #         data=data,
            #         train_loader=train_sampler,
            #         val_loader=val_sampler,
            #         label=step_label,
            #         total_steps=total_steps,
            #         completed_steps=completed,
            #     )

            #     completed += 1

            #     # Save trial history
            #     trial_history["config"] = config_name
            #     trial_history["trial"]  = trial_id
            #     trial_history.to_csv(history_path, index=False)
            #     self.send_info_event(
            #         progress=completed / total_steps,
            #         message=f"[{step_label}] History saved → {history_path}",
            #     )

            #     # Save trial predictions
            #     result = self.inference(model, data)

            #     torch.save(
            #         result,
            #         result_path,
            #     )

            #     self.send_info_event(
            #         progress=completed / total_steps,
            #         message=f"[{step_label}] Predictions saved → {result_path}",
            #     )

            #     del result
            #     del model  # free VRAM between trials

            # ----------------------------------------------------------
            # Full-graph retrain  (only this model is saved as .pt)
            # ----------------------------------------------------------

            full_label     = f"config '{config_name}' | full retrain"
            full_step_prog = completed / total_steps

            self.send_info_event(
                progress=full_step_prog,
                message=f"[{full_label}] All {self.config.num_splits} trials done — "
                        f"retraining on full graph …",
            )

            full_sampler = self.create_full_sampler(data)

            history_path = os.path.join(
                HISTORIES_DIR, 
                f"history_{config_name}.csv"
            )

            result_path = os.path.join(
                RESULTS_DIR, 
                f"predictions_{config_name}.pt"
            )

            # For the full retrain we still track a val signal using the full
            # loader (same data) so the run_experiment interface is satisfied.
            history, final_model = self.run_experiment(
                config=config,
                data=data,
                train_loader=full_sampler,
                val_loader=full_sampler,      # no held-out split
                label=full_label,
                total_steps=total_steps,
                completed_steps=completed,
            )

            completed += 1

            model_path = os.path.join(MODELS_DIR, f"model_{config_name}.pt")
            torch.save(final_model.state_dict(), model_path)
            self.send_info_event(
                progress=completed / total_steps,
                message=f"[{full_label}] Final model saved → {model_path}",
            )

            # Save trial history
            history["config"] = config_name
            step_label  = f"config '{config_name}'"
            history.to_csv(history_path, index=False)
            self.send_info_event(
                progress=completed / total_steps,
                message=f"[{step_label}] History saved → {history_path}",
            )

            # Save trial predictions
            result = self.inference(final_model, data)

            torch.save(
                result,
                result_path,
            )

            self.send_info_event(
                progress=completed / total_steps,
                message=f"[{step_label}] Predictions saved → {result_path}",
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
        trainer = cls(experiment=config.experiment, config=config.training_config)
        trainer.add_handler(ConsoleLogger("Trainer"))
        return trainer