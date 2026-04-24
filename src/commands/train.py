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
from typing import Any, Dict, List, Optional, Self, Tuple
from pydantic import BaseModel, Field, computed_field
from sklearn.model_selection import KFold
from torch import Tensor, optim
from torch.nn import functional as F
from torch_geometric import utils as gnn_utils
from torch_sparse import SparseTensor
from torchmetrics.functional import accuracy, f1_score

from definitions import EXPERIMENTS_DIR, FEATURES_DIR
from instance import cli
from src import helpers
from src.gnn import (
    ArticleAsKG,
    ArticlesGraph,
    HetGNNConfig,
    HetModel,
    HetModelReturn,
    HybridSampler,
    KnowledgeGraph,
)
from src.shared.command_tools import BaseCommand
from src.shared.handlers import ConsoleLogger, Event


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
    label : str = "domains"                        

    # Losses
    add_ssl_loss: bool = False

    # Sampling
    batch_size: int = 8
    num_articles_neighbors: tuple[int, int] = (10, 6)
    num_entities_neighbors: tuple[int, int] = ()

    # Model
    model_params: HetGNNConfig = Field(default_factory=HetGNNConfig)

    # Hyper-parameters
    lr: float = 0.00075
    num_epochs: int = 10
    accumulation_steps: int = 1
    weight_decay: float = 1e-4

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Negative Sampling
    do_negative_sampling: bool = False

    # Logging
    log_every: int = 1


class TrainingConfig(BaseTrainingConfig):
    seed: int = 56
    num_splits: int = 4
    hparams: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {"default": {}})


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@cli.register_stage(
    name="train",
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
                        f"batch={cfg.batch_size}, ssl={cfg.add_ssl_loss}",
            )

        return configs

    # ------------------------------------------------------------------
    # Data loading  (new graph.pt format — no filtering)
    # ------------------------------------------------------------------

    def load_data(self) -> ArticleAsKG:

        graph_path = os.path.join(FEATURES_DIR, *self.config.graph.split("/"), "graph.pt")

        self.send_info_event(
            progress=0.0,
            message=f"[Data] Loading graph from: {graph_path}",
        )

        raw: Dict[str, Any] = torch.load(graph_path, weights_only=False)

        # ── Entity / KG tensors ──────────────────────────────────────
        #
        # Entity_HasRelationWith_Entity is [3, E]: row0=src, row1=dst, row2=pred_idx
        triple_edge = raw["edges"][("Entity", "HasRelationWith", "Entity")]
        src, dst, edge_type = triple_edge[0], triple_edge[1], triple_edge[2]

        self.send_debug_event(
            progress=0.0,
            message=f"[Data] KG triples: {src.shape[0]:,}  |  "
                    f"Entities: {len(raw['nodes']['Entity']['id']):,}  |  "
                    f"Predicates: {len(raw['predicates']['id'])}",
        )

        kg = KnowledgeGraph(
            x=raw["nodes"]["Entity"]["embeddings"],
            e=raw["predicates"]["embeddings"][edge_type],
            r=raw["predicates"]["embeddings"],
            edge_index=torch.stack([src, dst]),
            edge_type=edge_type,
            entity_names=raw["nodes"]["Entity"]["id"],
        )

        kg.generality_score = raw["nodes"]["Entity"]["generality_score"]
        kg.relevancy_score  = raw["nodes"]["Entity"]["relevancy_score"]

        # ── Article–Article edges  (pre-computed by the pipeline) ────
        #
        # Both ShareKeyword and ShareAuthor are [2, E] tensors stored
        # under tuple keys.  We concatenate them into a single edge_index
        # with an edge_type discriminator (0 = keyword, 1 = author).

        share_kw  = raw["edges"].get(("Article", "ShareKeyword", "Article"))
        share_auth = raw["edges"].get(("Article", "ShareAuthor",  "Article"))

        edge_parts, type_parts = [], []

        if share_kw is not None:
            edge_parts.append(share_kw)
            type_parts.append(torch.zeros(share_kw.shape[1], dtype=torch.long))
            self.send_debug_event(
                progress=0.0,
                message=f"[Data] Article–ShareKeyword edges: {share_kw.shape[1]:,}",
            )

        if share_auth is not None:
            edge_parts.append(share_auth)
            type_parts.append(torch.ones(share_auth.shape[1], dtype=torch.long))
            self.send_debug_event(
                progress=0.0,
                message=f"[Data] Article–ShareAuthor edges: {share_auth.shape[1]:,}",
            )

        if edge_parts:
            article_edge_index = torch.cat(edge_parts, dim=-1)
            article_edge_type  = torch.cat(type_parts, dim=0)
        else:
            # Fallback: no article–article edges
            n_art = len(raw["nodes"]["Article"]["id"])
            article_edge_index = torch.zeros((2, 0), dtype=torch.long)
            article_edge_type  = torch.zeros(0, dtype=torch.long)
            self.send_warn_event(
                progress=0.0,
                message="[Data] No Article–Article edges found in graph.pt",
            )

        # ── Labels  (Article → SubDomain) ───────────────────────────
        n_articles  = len(raw["nodes"]["Article"]["id"])
        n_labels = len(raw["labels"][self.config.label])
        label_name = "SubDomain" if self.config.label == "domains" else "Lab"

        belongs_to = raw["edges"].get(("Article", "BelongsTo", label_name))

        if belongs_to is not None:
            row, col = belongs_to[0], belongs_to[1]
            labels = SparseTensor(
                row=row, col=col,
                sparse_sizes=(n_articles, n_labels),
            ).to_dense()
        else:
            labels = torch.zeros((n_articles, n_labels))
            self.send_warn_event(
                progress=0.0,
                message="[Data] Article–BelongsTo–SubDomain edges not found; labels are all-zero",
            )

        self.send_debug_event(
            progress=0.0,
            message=f"[Data] Articles: {n_articles:,}  |  SubDomains: {n_labels}  |  "
                    f"Labelled articles: {(labels.sum(-1) > 0).sum().item():,}",
        )

        # ── Triple → Article provenance ──────────────────────────────
        triple_article = raw["edges"].get("Triple_ExtractedFrom_Article")
        if triple_article is None:
            self.send_warn_event(
                progress=0.0,
                message="[Data] Triple_ExtractedFrom_Article edge not found in graph.pt",
            )

        articles = ArticlesGraph(
            x=raw["nodes"]["Article"]["embeddings"],
            edge_index=article_edge_index,
            edge_type=article_edge_type,
            y=labels,
        )

        data = ArticleAsKG(
            articles=articles,
            kg=kg,
            edge_article=triple_article,
        )

        data.metadata = {
            "predicates": raw["predicates"]["id"],
            "labels":     raw["labels"],
        }

        self.send_info_event(
            progress=0.0,
            message=f"[Data] Graph loaded — "
                    f"{n_articles:,} articles, "
                    f"{len(raw['nodes']['Entity']['id']):,} entities, "
                    f"{n_labels} subdomain labels",
        )

        return self.normalize_data(data)

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize_data(self, data: ArticleAsKG) -> ArticleAsKG:
        
        self.send_debug_event(progress=0.0, message="[Data] Normalising node features …")

        def _norm(t: Tensor) -> Tensor:
            mean = t.mean(0, keepdim=True)
            std  = t.std(0,  keepdim=True)
            return (t - mean) / (std + 1e-3)

        data.articles.x        = _norm(data.articles.x)
        data.kg.x              = _norm(data.kg.x)
        data.kg.r              = _norm(data.kg.r)
        data.kg.edge_attr      = _norm(data.kg.edge_attr)

        return data

    # ------------------------------------------------------------------
    # Splits & samplers
    # ------------------------------------------------------------------

    def create_or_load_split(self, data: ArticleAsKG) -> Tensor:
        graph_slug = self.config.graph.replace("/", "_")
        split_path = os.path.join(
            EXPERIMENTS_DIR, self.experiment,
            f"split_{graph_slug}_{self.config.num_splits}.pt",
        )

        if os.path.exists(split_path):
            self.send_debug_event(
                progress=0.0,
                message=f"[Split] Loading cached split from {split_path}",
            )
            return torch.load(split_path, map_location="cpu", weights_only=False)

        self.send_debug_event(
            progress=0.0,
            message=f"[Split] Creating new {self.config.num_splits}-fold split …",
        )

        kf = KFold(n_splits=self.config.num_splits, shuffle=True)
        n  = data.articles.num_nodes
        split_tensor = torch.full((n,), -1, dtype=torch.long)

        for fold, (_, val_idx) in enumerate(kf.split(range(n))):
            split_tensor[val_idx] = fold

        torch.save(split_tensor, split_path)
        self.send_info_event(
            progress=0.0,
            message=f"[Split] {self.config.num_splits}-fold split saved to {split_path}",
        )
        return split_tensor

    def _make_sampler(self, split: ArticleAsKG, is_train: bool) -> HybridSampler:

        return HybridSampler(
            articles=split.articles,
            kg=split.kg,
            edge_article=split.edge_article,
            articles_loader_args={
                "batch_size":    self.config.batch_size,
                "num_neighbors": self.config.num_articles_neighbors,
                "drop_last":     is_train,
                "shuffle":       is_train,
            },
            kg_sampler_args={
                "num_neighbors": self.config.num_entities_neighbors,
            },
        )

    def create_samplers(
        self, 
        data: ArticleAsKG
    ) -> List[Tuple[HybridSampler, HybridSampler]]:
        split_tensor = self.create_or_load_split(data)
        samplers: List[Tuple[HybridSampler, HybridSampler]] = []

        for fold in range(self.config.num_splits):
            train_mask = split_tensor != fold
            val_mask   = split_tensor == fold

            self.send_debug_event(
                progress=0.0,
                message=f"[Split] Fold {fold}: "
                        f"train={train_mask.sum().item():,}  "
                        f"val={val_mask.sum().item():,}",
            )

            train_split = data.subgraph_articles(data.articles.article_id[train_mask])
            val_split   = data.subgraph_articles(data.articles.article_id[val_mask])

            samplers.append((
                self._make_sampler(train_split, is_train=True),
                self._make_sampler(val_split,   is_train=False),
            ))

        return samplers

    def create_full_sampler(self, data: ArticleAsKG) -> HybridSampler:
        """Sampler over the entire graph (used for final retraining)."""
        return self._make_sampler(data, is_train=True)

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def create_model(self, config: BaseTrainingConfig, data: ArticleAsKG) -> HetModel:

        self.send_info_event(progress=0.0, message="[Model] Instantiating GNN model …")

        config.model_params.ssl     = config.add_ssl_loss
        config.model_params.cluster = False

        model = HetModel.from_config(
            in_channels=data.kg.num_features,
            num_labels=data.articles.y.size(-1),
            num_relations=(data.kg.num_relations, data.articles.num_relations),
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
        edge_index: Tensor, edge_type: Tensor, num_nodes: int
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

    @staticmethod
    def loss(
        config: BaseTrainingConfig,
        batch: ArticleAsKG,
        results: HetModelReturn,
        batch_size: int,
        labels: Optional[Tensor] = None,
        negative_sample: Optional[Tuple[Tensor, Tensor]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Tensor]:
        
        weights = weights or {}
        losses: Dict[str, Tensor] = {}

        y_hat = results.articles_predictions[:batch_size]
        y     = batch.articles.y[:batch_size]

        if negative_sample is not None:
            row, col = negative_sample
            losses["guidance_loss"] = F.binary_cross_entropy_with_logits(
                y_hat[row, col], y[row, col].float()
            )
        else:
            losses["guidance_loss"] = F.binary_cross_entropy_with_logits(y_hat, y.float())

        if config.add_ssl_loss:
            losses["ssl_loss"] = F.binary_cross_entropy_with_logits(
                results.structure_y_hat, labels.float()
            )

        total = torch.scalar_tensor(0.0).to(y_hat.device)
        for name, value in losses.items():
            w = weights.get(name, 1.0)
            if w > 0:
                total = total + w * value

        losses["loss"] = total
        return losses

    @staticmethod
    def evaluate(
        config: BaseTrainingConfig,
        batch: ArticleAsKG,
        results: HetModelReturn,
        batch_size: int,
        labels: Optional[Tensor] = None,
        negative_sample: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        y     = batch.articles.y[:batch_size]
        y_hat = results.articles_predictions[:batch_size]

        if negative_sample is not None:
            row, col = negative_sample
            y, y_hat = y[row, col], y_hat[row, col]
            metrics["guidance_acc"] = accuracy(y_hat, y, task="binary").item()
            metrics["guidance_f1"]  = f1_score(y_hat, y, task="binary").item()
        else:
            n = batch.articles.y.size(-1)
            metrics["guidance_acc"] = accuracy(y_hat, y, task="multilabel", num_labels=n, average="macro").item()
            metrics["guidance_f1"]  = f1_score(y_hat, y, task="multilabel", num_labels=n, average="macro").item()

        if config.add_ssl_loss:
            metrics["structure_acc"] = accuracy(results.structure_y_hat, labels, task="binary").item()
            metrics["structure_f1"]  = f1_score(results.structure_y_hat, labels, task="binary").item()

            row, col = batch.kg.edge_index
            h, t = results.entities_embeddings[row], results.entities_embeddings[col]
            r_hat = (h - t) @ results.relations_embeddings.t()
            n_r   = batch.kg.r.size(0)

            metrics["HITS@5"]  = accuracy(r_hat, batch.kg.edge_type, task="multiclass", num_classes=n_r, top_k=5).item()
            metrics["HITS@10"] = accuracy(r_hat, batch.kg.edge_type, task="multiclass", num_classes=n_r, top_k=10).item()

        return metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        config: BaseTrainingConfig,
        data: ArticleAsKG,
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

                from rich.progress import Progress as RichProgress
                with RichProgress() as pbar:
                    task = pbar.add_task(
                        f"[blue]Epoch {epoch + 1}/{config.num_epochs} | {phase.upper()}",
                        total=n_batches,
                    )

                    for batch in loader:
                        batch = helpers.move_data_to_device(batch, device=config.device)

                        sample, labels = None, None
                        if config.model_params.ssl:
                            ei, et, labels = self.sample_negative_edges(
                                batch.kg.edge_index, batch.kg.edge_type, batch.kg.num_nodes
                            )
                            sample = (ei, et)

                        with torch.set_grad_enabled(is_training):
                            results   = model.forward(batch, sample)
                            neg_smpl  = (
                                self.negative_sample(batch.articles.y[:config.batch_size])
                                if is_training and config.do_negative_sampling
                                else None
                            )
                            losses = self.loss(
                                config, batch=batch, results=results,
                                batch_size=config.batch_size,
                                labels=labels, negative_sample=neg_smpl,
                                weights={"guidance_loss": 1.0, "ssl_loss": 0.05},
                            )

                        if is_training:
                            optimizer.zero_grad()
                            losses["loss"].backward()
                            optimizer.step()

                        metrics = self.evaluate(
                            config, batch=batch, results=results,
                            batch_size=config.batch_size,
                            labels=labels, negative_sample=neg_smpl,
                        )

                        epoch_rows.append({
                            "epoch": epoch,
                            "phase": phase,
                            **{k: v.item() for k, v in losses.items()},
                            **metrics,
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

                    row = summary.loc[phase]
                    loss_val  = row.get("loss", float("nan"))
                    guid_loss = row.get("guidance_loss", float("nan"))
                    acc_val   = row.get("guidance_acc",  float("nan"))
                    f1_val    = row.get("guidance_f1",   float("nan"))

                    # Build optional extras string
                    extras: List[str] = []
                    if "ssl_loss" in row:
                        extras.append(f"ssl_loss={row['ssl_loss']:.4f}")
                    if "structure_f1" in row:
                        extras.append(f"struct_f1={100 * row['structure_f1']:.2f}%")
                    if "HITS@5" in row:
                        extras.append(f"HITS@5={100 * row['HITS@5']:.2f}%")
                    if "HITS@10" in row:
                        extras.append(f"HITS@10={100 * row['HITS@10']:.2f}%")

                    extras_str = ("  |  " + "  ".join(extras)) if extras else ""

                    self.send_info_event(
                        progress=epoch_progress,
                        message=(
                            f"[{label}] Epoch {epoch + 1}/{config.num_epochs} "
                            f"[{phase.upper():5s}]  "
                            f"loss={loss_val:.4f}  "
                            f"guid_loss={guid_loss:.4f}  "
                            f"acc={100 * acc_val:.2f}%  "
                            f"f1={100 * f1_val:.2f}%"
                            f"{extras_str}"
                        ),
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
    def inference(model: HetModel, data: ArticleAsKG) -> Tensor:
        model = model.eval().cpu()
        data  = data.cpu()

        model.ssl     = False
        model.cluster = False

        with torch.no_grad():
            return model.forward(data).articles_predictions

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def __iter__(self):

        helpers.seed_everything(self.config.seed)

        configs = self.get_configs()
        data    = self.load_data()

        # Filter to articles with at least one label (required for training)
        labeled_mask = data.articles.y.sum(-1) > 0
        data = data.subgraph_articles(data.articles.article_id[labeled_mask])

        self.send_info_event(
            progress=0.0,
            message=f"[Data] Articles after label filter: {data.articles.num_nodes:,}",
        )

        MODELS_DIR    = os.path.join(EXPERIMENTS_DIR, self.experiment, "models")
        RESULTS_DIR   = os.path.join(EXPERIMENTS_DIR, self.experiment, "results")
        HISTORIES_DIR = os.path.join(EXPERIMENTS_DIR, self.experiment, "histories")

        os.makedirs(MODELS_DIR,    exist_ok=True)
        os.makedirs(RESULTS_DIR,   exist_ok=True)
        os.makedirs(HISTORIES_DIR, exist_ok=True)

        n_configs = len(configs)
        # Each config contributes (num_splits trials + 1 full retrain) steps
        total_steps = n_configs * (self.config.num_splits + 1)
        completed   = 0

        for config_idx, (config_name, config) in enumerate(configs.items()):

            self.send_info_event(
                progress=completed / total_steps,
                message=f"[Config '{config_name}'] Starting  "
                        f"({config_idx + 1}/{n_configs} configurations)",
            )

            samplers = self.create_samplers(data)

            # ----------------------------------------------------------
            # K-Fold trials
            # ----------------------------------------------------------

            for trial_id, (train_sampler, val_sampler) in enumerate(samplers):

                history_path = os.path.join(
                    HISTORIES_DIR, f"history_{config_name}_trial{trial_id}.csv"
                )
                result_path = os.path.join(
                    RESULTS_DIR, f"predictions_{config_name}_trial{trial_id}.pt"
                )

                step_label   = f"config '{config_name}' | trial {trial_id + 1}/{self.config.num_splits}"
                step_progress = completed / total_steps

                if os.path.exists(result_path) and os.path.exists(history_path):
                    self.send_info_event(
                        progress=step_progress,
                        message=f"[{step_label}] Already completed — skipping",
                    )
                    completed += 1
                    continue

                self.send_info_event(
                    progress=step_progress,
                    message=f"[{step_label}] Starting trial …",
                )

                trial_history, model = self.run_experiment(
                    config=config,
                    data=data,
                    train_loader=train_sampler,
                    val_loader=val_sampler,
                    label=step_label,
                    total_steps=total_steps,
                    completed_steps=completed,
                )

                completed += 1

                # Save trial history
                trial_history["config"] = config_name
                trial_history["trial"]  = trial_id
                trial_history.to_csv(history_path, index=False)
                self.send_info_event(
                    progress=completed / total_steps,
                    message=f"[{step_label}] History saved → {history_path}",
                )

                # Save trial predictions
                preds = self.inference(model, data)
                torch.save(preds, result_path)
                self.send_info_event(
                    progress=completed / total_steps,
                    message=f"[{step_label}] Predictions saved → {result_path}",
                )

                del model  # free VRAM between trials

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
            # For the full retrain we still track a val signal using the full
            # loader (same data) so the run_experiment interface is satisfied.
            _, final_model = self.run_experiment(
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