"""
Knowledge Graph Feature Extraction Pipeline
============================================
Processes JSONL triples into an embedded, annotated graph ready for GNN training.

Stages:
  1. Parse  – Build graph structure from raw JSONL records
  2. Filter – Prune by connectivity, coverage, and label criteria
  3. Embed  – Encode nodes/edges with SentenceTransformer
  4. Score  – Add generality, relevancy, PageRank, degree metadata
  5. Save   – Write training tensors + Neo4j-ready metadata (link preserved)

Entry-point:
    python feature_extraction.py                          # defaults
    python feature_extraction.py --corpus my-corpus
    python feature_extraction.py --min-entity-articles 3
"""

import json
import networkx as nx
import numpy as np
import torch
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, computed_field
from rich.progress import BarColumn,Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from sentence_transformers import SentenceTransformer
from definitions import hf_model_dir, FEATURES_DIR, KGS_DIR, domain_dir
from instance import cli
from src import helpers
from src.shared.command_tools import BaseCommand
from src.shared.handlers import ConsoleLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_progress() -> Progress:

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        transient=False,
    )


def _slugify(text: str) -> str:
    return text.strip().replace(" ", "_").lower()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Domain:
    domain : str
    sub_domain : str
    definition : str

class Config(BaseModel):
    corpus: str = "ls2n-duke"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Filtering
    largest_component_only: bool = True
    min_component_size: int = 1
    require_subdomain: bool = True
    min_entity_articles: int = 1
    remove_isolated: bool = True

    domain : str = "computer_science"

    @computed_field
    @property
    def domains(self) -> List[Domain]:
        return [Domain(**domain) for domain in helpers.load_json_file(domain_dir(self.domain))]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NodeStore:
    """Typed accumulator – deduplicates by string id."""
    _data: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add(self, node_id: str, props: Dict[str, Any]) -> None:
        self._data.setdefault(node_id, props)

    def as_list(self) -> List[Dict[str, Any]]:
        return list(self._data.values())

    def ids(self) -> List[str]:
        return list(self._data.keys())

    def id2idx(self) -> Dict[str, int]:
        return {nid: i for i, nid in enumerate(self._data)}

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class EdgeStore:
    """
    Typed edge accumulator.
    Entity–HasRelationWith–Entity merges article lists on collision so that
    the same triple extracted from multiple articles keeps all provenance.
    """
    _data: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add(self, edge_id: str, props: Dict[str, Any]) -> None:
        if edge_id not in self._data:
            self._data[edge_id] = props
        else:
            existing = self._data[edge_id].get("articles", [])
            incoming = props.get("articles", [])
            self._data[edge_id]["articles"] = list(dict.fromkeys(existing + incoming))

    def as_list(self) -> List[Dict[str, Any]]:
        return list(self._data.values())

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class RawGraph:
    nodes: Dict[str, NodeStore]
    edges: Dict[str, EdgeStore]
    predicates: Dict[str, int]

# ---------------------------------------------------------------------------
# Stage 1 – Parse
# ---------------------------------------------------------------------------

class GraphParser:
    """
    Reads the JSONL file line-by-line and populates NodeStores / EdgeStores.

    Node types : Article, Author, Venue, Lab, SubDomain, Keyword, Entity
    Edge types : Article_WrittenBy_Author, Article_BelongsTo_Venue,
                 Article_BelongsTo_Lab, Article_BelongsTo_SubDomain,
                 Article_HasKeyword_Keyword, Entity_AppearsIn_Article,
                 Entity_HasRelationWith_Entity
    """

    def __init__(self, domains : List[Domain], data_path: Path, host: "FeatureExtractor") -> None:
        self.data_path = data_path
        self.domains = domains
        self._host = host

    # ------------------------------------------------------------------

    def parse(self) -> RawGraph:

        self._host.send_info_event(
            progress=None,
            message="[Stage 1 · Parse] Starting — reading JSONL triples",
        )

        self._host.send_debug_event(
            progress=None,
            message=f"[Stage 1 · Parse] Input file: {self.data_path}",
        )

        nodes: Dict[str, NodeStore] = defaultdict(NodeStore)
        edges: Dict[str, EdgeStore] = defaultdict(EdgeStore)
        predicates: Dict[str, int] = {}
        skipped = 0

        self._host.send_debug_event(
            progress=None,
            message=f"[Stage 1 · Parse] Adding Links between domains and sub-domains.",
        )

        ### TODO: make god node attributes dynamic
        nodes['ExpertiseMap'].add("nantes_uni_2020_2026_PUI", {
            "id" : "nantes_uni_2020_2026_PUI",
            "description" : (
                "A comprehensive map of research expertise within 11 University of Nantes laboratories. "
                "Spanning 2020–2026, this dataset is categorized according to the PUI’s 18 strategic areas of interest."
            )
        })

        for domain in self.domains:

            domain_id = _slugify(domain.domain)
            subdomain_id = _slugify(domain.sub_domain)

            nodes['Domain'].add(domain_id, {
                "id" : domain_id,
                "name" : domain.domain,
                "type" : "Domain"
            })

            nodes["SubDomain"].add(subdomain_id, {
                "id" : subdomain_id,
                "name" : domain.sub_domain,
                "type" : "SubDomain"
            })

            edges['SubDomain_IsSubDomainOf_Domain'].add(f'{domain_id}-{subdomain_id}', {
                "id" : f'{domain_id}-{subdomain_id}',
                "src" : subdomain_id,
                "dst" : domain_id
            })

            edges['Domain_BelongsTo_ExpertiseMap'].add(f'{domain_id}-nantes_uni_2020_2026_PUI', {
                "id" : f'{domain_id}-nantes_uni_2020_2026_PUI',
                "src" : domain_id,
                "dst" : 'nantes_uni_2020_2026_PUI'
            })


        with _make_progress() as pbar:

            task = pbar.add_task("Parsing records …", total=None)

            for record in self._iter_records():
                ok = self._process_record(record, nodes, edges, predicates)
                if not ok:
                    skipped += 1
                pbar.advance(task)

        if skipped:
            self._host.send_warn_event(
                progress=None,
                message=f"[Stage 1 · Parse] Skipped {skipped} malformed / incomplete records",
            )

        for ntype, store in nodes.items():
            self._host.send_info_event(
                progress=None,
                message=f"[Stage 1 · Parse] Node type '{ntype}' — {len(store):,} unique nodes",
            )
        for etype, store in edges.items():
            self._host.send_debug_event(
                progress=None,
                message=f"[Stage 1 · Parse] Edge type '{etype}' — {len(store):,} edges",
            )

        self._host.send_info_event(
            progress=None,
            message=f"[Stage 1 · Parse] Distinct predicates: {len(predicates)}",
        )
        self._host.send_info_event(
            progress=None,
            message="[Stage 1 · Parse] Completed successfully",
        )

        return RawGraph(nodes=dict(nodes), edges=dict(edges), predicates=predicates)

    # ------------------------------------------------------------------

    def _iter_records(self):
        with open(self.data_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        self._host.send_warn_event(
                            progress=None,
                            message=f"[Stage 1 · Parse] JSON decode error — skipping line: {exc}",
                        )

    def _process_record(
        self,
        record: Dict[str, Any],
        nodes: Dict[str, NodeStore],
        edges: Dict[str, EdgeStore],
        predicates: Dict[str, int],
    ) -> bool:
        """Returns False if the record is structurally invalid."""
        head_text = record.get("head")
        tail_text = record.get("tail")
        predicate  = record.get("predicate", "")

        if not head_text or not tail_text:
            return False

        sentence = record.get("sentence")
        document = sentence["document"] if sentence else None

        # ── Article + metadata ───────────────────────────────────────
        if document:
            doc_id = document["id"]

            nodes["Article"].add(doc_id, {
                "id": doc_id,
                "title": document["title"],
                "abstract": document.get("abstract", ""),
                "date": document.get("date", ""),
                "type": "Article",
            })

            for author in document.get("authors", []):
                a_id = _slugify(author)
                nodes["Author"].add(a_id, {"id": a_id, "name": author, "type": "Author"})
                edges["Article_WrittenBy_Author"].add(
                    f"{doc_id}-{a_id}",
                    {"id": f"{doc_id}-{a_id}", "src": doc_id, "dst": a_id},
                )

            for venue in document.get("venues", []):
                v_id = _slugify(venue)
                nodes["Venue"].add(v_id, {"id": v_id, "name": venue, "type": "Venue"})
                edges["Article_BelongsTo_Venue"].add(
                    f"{doc_id}-{v_id}",
                    {"id": f"{doc_id}-{v_id}", "src": doc_id, "dst": v_id},
                )

            for team in document.get("teams", []):
                lab_id = _slugify(team)
                nodes["Lab"].add(lab_id, {"id": lab_id, "name": team, "type": "Lab"})
                edges["Article_BelongsTo_Lab"].add(
                    f"{doc_id}-{lab_id}",
                    {"id": f"{doc_id}-{lab_id}", "src": doc_id, "dst": lab_id},
                )

            for label in document.get("labels", []):
                sd_id = _slugify(label)
                nodes["SubDomain"].add(sd_id, {"id": sd_id, "name": label, "type": "SubDomain"})
                edges["Article_BelongsTo_SubDomain"].add(
                    f"{doc_id}-{sd_id}",
                    {"id": f"{doc_id}-{sd_id}", "src": doc_id, "dst": sd_id},
                )

            for kw in document.get("keywords", []):
                kw_id = _slugify(kw)
                nodes["Keyword"].add(kw_id, {"id": kw_id, "name": kw, "type": "Keyword"})
                edges["Article_HasKeyword_Keyword"].add(
                    f"{doc_id}-{kw_id}",
                    {"id": f"{doc_id}-{kw_id}", "src": doc_id, "dst": kw_id},
                )

        # ── Entities ─────────────────────────────────────────────────
        for entity_text in [head_text, tail_text]:
            e_id = _slugify(entity_text)
            nodes["Entity"].add(e_id, {"id": e_id, "name": entity_text, "type": "Entity"})

            if document:
                edges["Entity_AppearsIn_Article"].add(
                    f"{e_id}-{document['id']}",
                    {"id": f"{e_id}-{document['id']}", "src": e_id, "dst": document["id"]},
                )

        # ── Triple ───────────────────────────────────────────────────
        h_id      = _slugify(head_text)
        t_id      = _slugify(tail_text)
        p_id      = _slugify(predicate)
        triple_id = f"{h_id}-{p_id}-{t_id}"

        if predicate not in predicates:
            predicates[predicate] = len(predicates)

        edges["Entity_HasRelationWith_Entity"].add(
            triple_id,
            {
                "id": triple_id,
                "src": h_id,
                "dst": t_id,
                "predicate": predicate,
                "predicate_group": record.get("predicate_group", predicate),
                "articles": [document["id"]] if document else [],
            },
        )

        return True


# ---------------------------------------------------------------------------
# Stage 2 – Filter
# ---------------------------------------------------------------------------

class GraphFilter:
    """
    Configurable pruning pipeline applied to a RawGraph.

    Steps (in order):
      1. Drop articles with no subdomain         (optional)
      2. Drop entities appearing in < N articles
      3. Keep largest / >= N-node components     (optional)
      4. Remove all isolated nodes               (optional)
    Each step cascades dangling-edge removal automatically.
    """

    def __init__(self, config: Config, host: "FeatureExtractor") -> None:
        self.cfg = config
        self._host = host

    # ------------------------------------------------------------------

    def filter(self, raw: RawGraph) -> RawGraph:

        self._host.send_info_event(
            progress=None,
            message="[Stage 2 · Filter] Starting — applying graph pruning rules",
        )

        nodes_counts = { key : len(value._data) for key, value in raw.nodes.items() }
        stop = False
        attempt = 0

        while not stop:

            self._host.send_debug_event(
                progress=None,
                message=(
                    f"[Stage 2 · Filter :: Attempt {attempt}] Config: "
                    f"require_subdomain={self.cfg.require_subdomain}, "
                    f"min_entity_articles={self.cfg.min_entity_articles}, "
                    f"largest_component_only={self.cfg.largest_component_only}, "
                    f"min_component_size={self.cfg.min_component_size}, "
                    f"remove_isolated={self.cfg.remove_isolated}"
                ),
            )

            if self.cfg.require_subdomain:
                raw = self._drop_articles_without_subdomain(raw)

            if self.cfg.min_entity_articles > 1:
                raw = self._drop_sparse_entities(raw)

            if self.cfg.largest_component_only or self.cfg.min_component_size > 1:
                raw = self._apply_component_filter(raw)

            if self.cfg.remove_isolated:
                raw = self._drop_isolated_nodes(raw)

            for ntype, store in raw.nodes.items():

                self._host.send_info_event(
                    progress=None,
                    message=f"[Stage 2 · Filter] Remaining '{ntype}' nodes: {len(store):,}",
                )

            self._host.send_info_event(
                progress=None,
                message="[Stage 2 · Filter] Completed successfully",
            )

            new_nodes_counts = { key : len(value._data) for key, value in raw.nodes.items() }

            stop = all([count == nodes_counts[key] for key, count in new_nodes_counts.items()])
            nodes_counts = new_nodes_counts
            attempt += 1
        
        return raw

    # ------------------------------------------------------------------

    def _drop_articles_without_subdomain(self, raw: RawGraph) -> RawGraph:
        self._host.send_debug_event(
            progress=None,
            message="[Stage 2 · Filter] Dropping articles with no associated subdomain",
        )
        keep: Set[str] = {
            e["src"]
            for e in raw.edges.get("Article_BelongsTo_SubDomain", EdgeStore()).as_list()
        }
        before = len(raw.nodes.get("Article", NodeStore()))
        raw = self._retain_articles(raw, keep)
        dropped = before - len(raw.nodes.get("Article", NodeStore()))
        self._host.send_info_event(
            progress=None,
            message=f"[Stage 2 · Filter] Dropped {dropped} articles (no subdomain label)",
        )
        return raw

    def _drop_sparse_entities(self, raw: RawGraph) -> RawGraph:

        self._host.send_debug_event(
            progress=None,
            message=(
                f"[Stage 2 · Filter] Dropping entities appearing in fewer than "
                f"{self.cfg.min_entity_articles} article(s)"
            ),
        )

        counts: Dict[str, int] = defaultdict(int)

        for edge in raw.edges.get("Entity_AppearsIn_Article", EdgeStore()).as_list():
            counts[edge["src"]] += 1

        keep: Set[str] = {eid for eid, cnt in counts.items() if cnt >= self.cfg.min_entity_articles}
        before = len(raw.nodes.get("Entity", NodeStore()))
        raw = self._retain_entities(raw, keep)
        dropped = before - len(raw.nodes.get("Entity", NodeStore()))
        self._host.send_info_event(
            progress=None,
            message=f"[Stage 2 · Filter] Dropped {dropped} sparse entities",
        )
        return raw

    def _apply_component_filter(self, raw: RawGraph) -> RawGraph:
        self._host.send_debug_event(
            progress=None,
            message="[Stage 2 · Filter] Computing connected components on entity subgraph",
        )
        G = nx.Graph()
        for edge in raw.edges.get("Entity_HasRelationWith_Entity", EdgeStore()).as_list():
            G.add_edge(edge["src"], edge["dst"])

        components = sorted(nx.connected_components(G), key=len, reverse=True)
        total = len(components)

        if self.cfg.largest_component_only:
            keep_components = components[:1]
            reason = "largest component only"
        else:
            keep_components = [c for c in components if len(c) >= self.cfg.min_component_size]
            reason = f"component size >= {self.cfg.min_component_size}"

        keep_nodes: Set[str] = set().union(*keep_components) if keep_components else set()

        self._host.send_info_event(
            progress=None,
            message=(
                f"[Stage 2 · Filter] Components: {total} total → "
                f"keeping {len(keep_components)} ({reason}), "
                f"covering {len(keep_nodes):,} entity nodes"
            ),
        )

        before = len(raw.nodes.get("Entity", NodeStore()))
        raw = self._retain_entities(raw, keep_nodes)
        dropped = before - len(raw.nodes.get("Entity", NodeStore()))
        self._host.send_info_event(
            progress=None,
            message=f"[Stage 2 · Filter] Dropped {dropped} entities (outside kept components)",
        )
        return raw

    def _drop_isolated_nodes(self, raw: RawGraph) -> RawGraph:
        self._host.send_debug_event(
            progress=None,
            message="[Stage 2 · Filter] Removing isolated nodes (no edges of any type)",
        )
        connected: Dict[str, Set[str]] = defaultdict(set)
        for etype, store in raw.edges.items():

            if etype == 'Entity_HasRelationWith_Entity':
                continue

            src_type, _, dst_type = etype.split("_", 2)

            for edge in store.as_list():
                if src_type != 'Article':
                    connected[src_type].add(edge["src"])
                connected[dst_type].add(edge["dst"])

        for ntype, nstore in raw.nodes.items():
            before = len(nstore)
            keep = connected.get(ntype, set())
            nstore._data = {k: v for k, v in nstore._data.items() if k in keep}
            dropped = before - len(nstore)
            if dropped:
                self._host.send_info_event(
                    progress=None,
                    message=f"[Stage 2 · Filter] Removed {dropped} isolated '{ntype}' nodes",
                )
        return raw

    # ------------------------------------------------------------------
    # Cascade helpers
    # ------------------------------------------------------------------

    def _retain_articles(self, raw: RawGraph, keep_ids: Set[str]) -> RawGraph:
        if "Article" in raw.nodes:
            raw.nodes["Article"]._data = {
                k: v for k, v in raw.nodes["Article"]._data.items() if k in keep_ids
            }
        return self._cascade_edge_filter(raw)

    def _retain_entities(self, raw: RawGraph, keep_ids: Set[str]) -> RawGraph:
        if "Entity" in raw.nodes:
            raw.nodes["Entity"]._data = {
                k: v for k, v in raw.nodes["Entity"]._data.items() if k in keep_ids
            }
        return self._cascade_edge_filter(raw)

    def _cascade_edge_filter(self, raw: RawGraph) -> RawGraph:
        """Remove edges whose endpoints no longer exist in any node store."""
        valid: Dict[str, Set[str]] = {
            ntype: set(store.ids()) for ntype, store in raw.nodes.items()
        }
        for etype, estore in raw.edges.items():
            src_type, _, dst_type = etype.split("_", 2)
            src_valid = valid.get(src_type, set())
            dst_valid = valid.get(dst_type, set())
            before = len(estore)
            estore._data = {
                eid: ev
                for eid, ev in estore._data.items()
                if ev["src"] in src_valid and ev["dst"] in dst_valid
            }
            removed = before - len(estore)
            if removed:
                self._host.send_debug_event(
                    progress=None,
                    message=(
                        f"[Stage 2 · Filter] Cascade: pruned {removed} dangling edges "
                        f"from '{etype}'"
                    ),
                )
        return raw


# ---------------------------------------------------------------------------
# Stage 3 – Embed
# ---------------------------------------------------------------------------

class Embedder:
    """
    Produces SentenceTransformer embeddings for:
      • Entity nodes  – by entity name
      • Article nodes – by abstract (query + passage variants for e5 models)
      • Predicates    – by predicate text
    """

    def __init__(self, config: Config, host: "FeatureExtractor") -> None:
        self.cfg = config
        self._host = host

    # ------------------------------------------------------------------

    def embed(self, raw: RawGraph) -> Dict[str, torch.Tensor]:

        self._host.send_info_event(
            progress=None,
            message="[Stage 3 · Embed] Starting — loading SentenceTransformer model",
        )
        
        self._host.send_debug_event(
            progress=None,
            message=(
                f"[Stage 3 · Embed] Model: {self.cfg.model_name}  "
                f"device: {self.cfg.device}  batch_size: {self.cfg.batch_size}"
            ),
        )

        model = SentenceTransformer(hf_model_dir(self.cfg.model_name), device=self.cfg.device)

        n_entities   = len(raw.nodes["Entity"])
        n_articles   = len(raw.nodes.get("Article", NodeStore()))
        n_predicates = len(raw.predicates)

        self._host.send_info_event(
            progress=None,
            message=(
                f"[Stage 3 · Embed] Encoding {n_entities:,} entities, "
                f"{n_articles:,} articles, {n_predicates} predicates"
            ),
        )

        entity_names    = [n["name"] for n in raw.nodes["Entity"].as_list()]
        article_texts   = [n["abstract"] for n in raw.nodes.get("Article", NodeStore()).as_list()]
        predicate_texts = list(raw.predicates.keys())

        emb: Dict[str, torch.Tensor] = {}

        self._host.send_debug_event(progress=None, message="[Stage 3 · Embed] Encoding entity names …")
        emb["Entity"] = self._encode(model, self._e5(entity_names, "query"), "Entities")
        self._host.send_debug_event(
            progress=None,
            message=f"[Stage 3 · Embed] Entity embeddings shape: {list(emb['Entity'].shape)}",
        )

        self._host.send_debug_event(progress=None, message="[Stage 3 · Embed] Encoding article abstracts (query) …")
        emb["Article_query"] = self._encode(model, self._e5(article_texts, "query"), "Articles (query)")

        is_e5 = "e5" in self.cfg.model_name.lower()
        self._host.send_debug_event(
            progress=None,
            message=(
                f"[Stage 3 · Embed] Encoding article abstracts (passage) — "
                f"{'asymmetric e5 mode' if is_e5 else 'reusing query embeddings'}"
            ),
        )
        emb["Article_passage"] = (
            self._encode(model, self._e5(article_texts, "passage"), "Articles (passage)")
            if is_e5
            else emb["Article_query"]
        )

        self._host.send_debug_event(progress=None, message="[Stage 3 · Embed] Encoding predicates …")
        emb["Predicate"] = self._encode(model, self._e5(predicate_texts, "query"), "Predicates")

        dim = emb["Entity"].shape[-1]
        self._host.send_info_event(
            progress=None,
            message=f"[Stage 3 · Embed] Embedding dimension: {dim}",
        )
        self._host.send_info_event(
            progress=None,
            message="[Stage 3 · Embed] Completed successfully",
        )
        return emb

    # ------------------------------------------------------------------

    def _encode(self, model: SentenceTransformer, sentences: List[str], label: str) -> torch.Tensor:
        with _make_progress() as pbar:
            pbar.add_task(f"Encoding {label} …", total=None)
            return model.encode(
                sentences=sentences,
                batch_size=self.cfg.batch_size,
                device=self.cfg.device,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
            ).cpu()

    def _e5(self, texts: List[str], prefix: str) -> List[str]:
        if "e5" in self.cfg.model_name.lower():
            return [f"{prefix}: {t}" for t in texts]
        return texts


# ---------------------------------------------------------------------------
# Stage 4 – Build tensors + score
# ---------------------------------------------------------------------------

class TensorGraphBuilder:
    """
    Converts RawGraph + embeddings into index-based PyTorch tensors and
    computes per-entity scores: generality, relevancy, degrees, PageRank,
    specificity-weighted PageRank, and component IDs.

    Output shape summary
    --------------------
    G["edges"][(src, rel, dst)]              → LongTensor [2, E]  (or [3, E] for triples)
    G["edges"]["Triple_ExtractedFrom_Article"] → LongTensor [2, T]
    G["edges"][(Article, ShareAuthor, Article)]  → LongTensor [2, X]
    G["edges"][(Article, ShareKeyword, Article)] → LongTensor [2, Y]
    """

    def __init__(self, config: Config, host: "FeatureExtractor") -> None:
        self.cfg = config
        self._host = host

    # ------------------------------------------------------------------

    def build(self, raw: RawGraph, emb: Dict[str, torch.Tensor]) -> Dict[str, Any]:

        self._host.send_info_event(
            progress=None,
            message="[Stage 4 · Build] Starting — converting graph to index tensors",
        )

        G: Dict[str, Any] = {"nodes": {}, "edges": {}, "predicates": {}, "labels": []}
        id2idx = {ntype: store.id2idx() for ntype, store in raw.nodes.items()}

        # ── Node tensors ─────────────────────────────────────────────
        G["nodes"]["Entity"] = {
            "id": raw.nodes["Entity"].ids(),
            "embeddings": emb["Entity"],
        }
        G["nodes"]["Article"] = {
            "id": raw.nodes.get("Article", NodeStore()).ids(),
            "embeddings": emb["Article_query"],
        }
        G["predicates"] = {
            "id": list(raw.predicates.keys()),
            "embeddings": emb["Predicate"],
        }
        G["labels"] = {
            'domains': [n["id"] for n in raw.nodes.get("SubDomain", NodeStore()).as_list()],
            'labs': [n["id"] for n in raw.nodes.get("Lab", NodeStore()).as_list()],
        }

        self._host.send_debug_event(
            progress=None,
            message=(
                f"[Stage 4 · Build] Node counts — "
                f"Entity: {len(G['nodes']['Entity']['id']):,}  "
                f"Article: {len(G['nodes']['Article']['id']):,}  "
                f"SubDomain labels: {len(G['labels']['domains'])} "
                f"Labs labels: {len(G['labels']['labs'])}"
            ),
        )

        # ── Edge tensors ─────────────────────────────────────────────
        self._host.send_debug_event(
            progress=None,
            message="[Stage 4 · Build] Building edge index tensors …",
        )

        triple_article_pairs: List[Tuple[int, int]] = []

        for etype, estore in raw.edges.items():
            
            src_type, rel, dst_type = etype.split("_", 2)
            src_map = id2idx.get(src_type, {})
            dst_map = id2idx.get(dst_type, {})
            rows: List[List[int]] = []

            for triple_idx, edge in enumerate(estore.as_list()):
                src_i = src_map.get(edge["src"])
                dst_i = dst_map.get(edge["dst"])
                if src_i is None or dst_i is None:
                    continue

                if etype == "Entity_HasRelationWith_Entity":
                    pred_i = raw.predicates[edge["predicate"]]
                    rows.append([src_i, dst_i, pred_i])
                    art_map = id2idx.get("Article", {})
                    for art_id in edge.get("articles", []):
                        art_i = art_map.get(art_id)
                        if art_i is not None:
                            triple_article_pairs.append((triple_idx, art_i))
                else:
                    rows.append([src_i, dst_i])

            if rows:
                tensor = torch.tensor(rows, dtype=torch.long).t().contiguous()
                G["edges"][(src_type, rel, dst_type)] = tensor
                self._host.send_debug_event(
                    progress=None,
                    message=(
                        f"[Stage 4 · Build] Edge ({src_type}, {rel}, {dst_type}): "
                        f"{tensor.shape[1]:,} entries"
                    ),
                )

        if triple_article_pairs:
            ext = torch.tensor(triple_article_pairs, dtype=torch.long).t().contiguous()
            G["edges"]["Triple_ExtractedFrom_Article"] = ext
            self._host.send_debug_event(
                progress=None,
                message=(
                    f"[Stage 4 · Build] Triple_ExtractedFrom_Article: "
                    f"{ext.shape[1]:,} (triple, article) pairs"
                ),
            )

        # ── Derived article–article edges ────────────────────────────
        self._host.send_debug_event(
            progress=None,
            message="[Stage 4 · Build] Deriving Article–ShareAuthor / ShareKeyword edges …",
        )
        G["edges"].update(self._article_cooccurrence(raw, id2idx))

        # ── Scores ───────────────────────────────────────────────────
        self._compute_scores(G, raw, emb, id2idx)

        self._host.send_info_event(
            progress=None,
            message="[Stage 4 · Build] Completed successfully",
        )
        return G

    # ------------------------------------------------------------------
    # Article co-occurrence
    # ------------------------------------------------------------------

    def _article_cooccurrence(
        self,
        raw: RawGraph,
        id2idx: Dict[str, Dict[str, int]],
    ) -> Dict[Any, torch.Tensor]:
        extras: Dict[Any, torch.Tensor] = {}
        art_map = id2idx.get("Article", {})

        for rel_name, edge_type in [
            ("ShareAuthor",  "Article_WrittenBy_Author"),
            ("ShareKeyword", "Article_HasKeyword_Keyword"),
        ]:
            bucket: Dict[str, Set[int]] = defaultdict(set)
            for edge in raw.edges.get(edge_type, EdgeStore()).as_list():
                a_i = art_map.get(edge["src"])
                if a_i is not None:
                    bucket[edge["dst"]].add(a_i)

            pairs: List[Tuple[int, int]] = []
            for arts in bucket.values():
                arts_list = sorted(arts)
                for i in range(len(arts_list)):
                    for j in range(i + 1, len(arts_list)):
                        pairs.append((arts_list[i], arts_list[j]))
                        pairs.append((arts_list[j], arts_list[i]))

            if pairs:
                t   = torch.tensor(pairs, dtype=torch.long).t().contiguous()
                key = ("Article", rel_name, "Article")
                extras[key] = t
                self._host.send_debug_event(
                    progress=None,
                    message=f"[Stage 4 · Build] Edge {key}: {t.shape[1]:,} edges",
                )

        return extras

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_scores(
        self,
        G: Dict[str, Any],
        raw: RawGraph,
        emb: Dict[str, torch.Tensor],
        id2idx: Dict[str, Dict[str, int]],
    ) -> None:
        self._host.send_info_event(
            progress=None,
            message=(
                "[Stage 4 · Score] Computing per-entity scores: "
                "degree, PageRank, generality, relevancy"
            ),
        )

        n_entities  = len(raw.nodes["Entity"])
        appears     = G["edges"].get(("Entity", "AppearsIn", "Article"))
        triple_edge = G["edges"].get(("Entity", "HasRelationWith", "Entity"))

        # ── Degree ───────────────────────────────────────────────────
        self._host.send_debug_event(progress=None, message="[Stage 4 · Score] Computing in/out degree …")
        in_deg  = torch.zeros(n_entities, dtype=torch.long)
        out_deg = torch.zeros(n_entities, dtype=torch.long)
        if triple_edge is not None:
            for d in triple_edge[1]:
                in_deg[d] += 1
            for s in triple_edge[0]:
                out_deg[s] += 1
        degree = in_deg + out_deg
        self._host.send_debug_event(
            progress=None,
            message=(
                f"[Stage 4 · Score] Degree stats — "
                f"max: {degree.max().item()}, "
                f"mean: {degree.float().mean().item():.2f}, "
                f"zero-degree nodes: {(degree == 0).sum().item()}"
            ),
        )

        # ── PageRank ─────────────────────────────────────────────────
        self._host.send_debug_event(progress=None, message="[Stage 4 · Score] Running PageRank (alpha=0.85) …")
        nx_G = nx.DiGraph()
        nx_G.add_nodes_from(range(n_entities))
        if triple_edge is not None:
            for s, d in zip(triple_edge[0].tolist(), triple_edge[1].tolist()):
                nx_G.add_edge(s, d)

        pr_dict  = nx.pagerank(nx_G, alpha=0.85, max_iter=500)
        pr_arr   = np.array([pr_dict.get(i, 0.0) for i in range(n_entities)], dtype=np.float32)
        pagerank = torch.from_numpy(pr_arr)
        self._host.send_debug_event(
            progress=None,
            message=(
                f"[Stage 4 · Score] PageRank — "
                f"max: {pagerank.max().item():.6f}, "
                f"mean: {pagerank.mean().item():.6f}"
            ),
        )

        # specificity-weighted PageRank  =  PR * log((N+1) / (degree+1))
        deg_np        = degree.numpy().astype(np.float32)
        specificity_pr = pagerank * torch.from_numpy(
            np.log((n_entities + 1.0) / (deg_np + 1.0)).astype(np.float32)
        )

        # ── Component IDs ────────────────────────────────────────────
        self._host.send_debug_event(progress=None, message="[Stage 4 · Score] Assigning component IDs …")
        nx_UG    = nx_G.to_undirected()
        comp_arr = np.zeros(n_entities, dtype=np.int32)
        n_comps  = 0
        for comp_id, comp_nodes in enumerate(nx.connected_components(nx_UG)):
            for node in comp_nodes:
                comp_arr[node] = comp_id
            n_comps = comp_id + 1
        component_id = torch.from_numpy(comp_arr)
        self._host.send_info_event(
            progress=None,
            message=f"[Stage 4 · Score] Connected components in final graph: {n_comps}",
        )

        # ── Generality + Relevancy ────────────────────────────────────
        self._host.send_debug_event(
            progress=None,
            message=(
                "[Stage 4 · Score] Computing generality "
                "(mean cosine dissimilarity across co-article embeddings) …"
            ),
        )
        self._host.send_debug_event(
            progress=None,
            message=(
                "[Stage 4 · Score] Computing relevancy "
                "(entity–article cosine dissimilarity) …"
            ),
        )

        article_query_emb   = G["nodes"]["Article"]["embeddings"]
        article_passage_emb = emb["Article_passage"]
        entity_emb          = G["nodes"]["Entity"]["embeddings"]

        generality_scores = torch.zeros(n_entities)
        relevancy_scores  = torch.zeros(n_entities)

        with _make_progress() as pbar:
            task_g = pbar.add_task("Generality …", total=n_entities)
            task_r = pbar.add_task("Relevancy …",  total=n_entities)

            for e_idx in range(n_entities):
                art_indices = self._articles_for_entity(e_idx, appears)

                if len(art_indices) > 0:
                    a_emb = article_query_emb[art_indices]
                    generality_scores[e_idx] = 1.0 - (a_emb @ a_emb.T).mean().item()
                pbar.advance(task_g)

                if len(art_indices) > 0:
                    e_vec = entity_emb[e_idx].unsqueeze(0)
                    p_emb = article_passage_emb[art_indices]
                    relevancy_scores[e_idx] = 1.0 - (e_vec @ p_emb.T).mean().item()
                pbar.advance(task_r)

        def _minmax(t: torch.Tensor) -> torch.Tensor:
            mn, mx = t.min(), t.max()
            return (t - mn) / (mx - mn + 1e-8)

        G["nodes"]["Entity"].update({
            "in_degree":                     in_deg,
            "out_degree":                    out_deg,
            "degree":                        degree,
            "pagerank":                      pagerank,
            "specificity_weighted_pagerank": _minmax(specificity_pr),
            "component_id":                  component_id,
            "generality_score":              _minmax(generality_scores),
            "relevancy_score":               _minmax(relevancy_scores),
        })

        self._host.send_info_event(
            progress=None,
            message=(
                f"[Stage 4 · Score] Scores attached — "
                f"avg generality: {generality_scores.mean().item():.4f}, "
                f"avg relevancy: {relevancy_scores.mean().item():.4f}"
            ),
        )

    @staticmethod
    def _articles_for_entity(e_idx: int, appears: Optional[torch.Tensor]) -> torch.Tensor:
        if appears is None:
            return torch.tensor([], dtype=torch.long)
        mask = appears[0] == e_idx
        return torch.unique(appears[1][mask])


# ---------------------------------------------------------------------------
# Stage 5 – Save
# ---------------------------------------------------------------------------

class GraphSerializer:
    """
    Writes three files to ``out_dir``:

    graph.pt
        Full tensor graph — embeddings, edge indices, per-node scores.

    metadata.json
        Human-readable node/edge tables with scalar scores inlined and
        ``_emb_index`` cross-linking each node to its tensor row.

    node_emb_index.json
        Compact {node_id → tensor_row} map for fast point-lookups
        without parsing the full metadata blob.
    """

    def __init__(self, out_dir: Path, host: "FeatureExtractor") -> None:
        self.out_dir = out_dir
        self._host = host
        out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def save(self, raw: RawGraph, G: Dict[str, Any]) -> None:
        self._host.send_info_event(
            progress=None,
            message=f"[Stage 5 · Save] Starting — output directory: {self.out_dir}",
        )

        # ── graph.pt ─────────────────────────────────────────────────
        pt_path = self.out_dir / "graph.pt"
        self._host.send_debug_event(progress=None, message=f"[Stage 5 · Save] Writing {pt_path} …")
        torch.save(G, pt_path)
        self._host.send_info_event(
            progress=None,
            message=f"[Stage 5 · Save] graph.pt written — {pt_path.stat().st_size / 1e6:.1f} MB",
        )

        # ── metadata.json ─────────────────────────────────────────────
        self._host.send_debug_event(
            progress=None,
            message="[Stage 5 · Save] Building metadata (enriching nodes with scores) …",
        )
        meta = self._build_metadata(raw, G)
        meta_path = self.out_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)
        self._host.send_info_event(
            progress=None,
            message=f"[Stage 5 · Save] metadata.json written — {meta_path.stat().st_size / 1e6:.1f} MB",
        )

        # ── node_emb_index.json ───────────────────────────────────────
        idx_path = self.out_dir / "node_emb_index.json"
        idx_map = {
            ntype: {nid: i for i, nid in enumerate(G["nodes"][ntype]["id"])}
            for ntype in ("Entity", "Article")
            if ntype in G["nodes"]
        }
        with open(idx_path, "w", encoding="utf-8") as fh:
            json.dump(idx_map, fh, ensure_ascii=False)
        self._host.send_info_event(
            progress=None,
            message=(
                f"[Stage 5 · Save] node_emb_index.json written — "
                f"Entity: {len(idx_map.get('Entity', {})):,}, "
                f"Article: {len(idx_map.get('Article', {})):,}"
            ),
        )

        self._host.send_info_event(
            progress=None,
            message="[Stage 5 · Save] All files written successfully",
        )

    # ------------------------------------------------------------------

    def _build_metadata(self, raw: RawGraph, G: Dict[str, Any]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"nodes": {}, "edges": {}}

        for ntype, store in raw.nodes.items():
            meta["nodes"][ntype] = store.as_list()

        score_keys = [
            "in_degree", "out_degree", "degree",
            "pagerank", "specificity_weighted_pagerank", "component_id",
            "generality_score", "relevancy_score",
        ]
        ent_id2idx = {nid: i for i, nid in enumerate(G["nodes"]["Entity"]["id"])}
        for node in meta["nodes"].get("Entity", []):
            idx = ent_id2idx.get(node["id"])
            if idx is not None:
                node["_emb_index"] = idx
                for key in score_keys:
                    t = G["nodes"]["Entity"].get(key)
                    if t is not None:
                        node[key] = t[idx].item()

        art_id2idx = {nid: i for i, nid in enumerate(G["nodes"]["Article"]["id"])}
        for node in meta["nodes"].get("Article", []):
            idx = art_id2idx.get(node["id"])
            if idx is not None:
                node["_emb_index"] = idx

        for etype, store in raw.edges.items():
            meta["edges"][etype] = store.as_list()

        return meta

@cli.register_stage(
    name="feature_extraction",
    description=(
        "Transforms raw JSONL triples into an embedded, annotated knowledge graph "
        "ready for GNN training. Includes parsing, filtering, sentence-bert embedding, "
        "PageRank scoring, and dual-format output (graph.pt + metadata.json)."
    ),
    add_from_conf=False,
    add_save_to=False,
)
class FeatureExtractor(BaseCommand):
    """
    Orchestrates all five pipeline stages and routes every ``send_*_event``
    call through the project's handler chain (ConsoleLogger by default).

    Each stage class receives ``host=self`` and uses:
      - self._host.send_info_event    for stage milestones and summaries
      - self._host.send_debug_event   for per-item / diagnostic details
      - self._host.send_warn_event for non-fatal anomalies
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------

    def __iter__(self):

        cfg = self.config

        data_path = Path(KGS_DIR) / cfg.corpus / "entities_mapping" / "data.jsonl"

        self.send_info_event(
            progress=None,
            message="===== Knowledge Graph Pipeline starting =====",
        )
        self.send_info_event(
            progress=None,
            message=(
                f"Corpus: {cfg.corpus}  |  Model: {cfg.model_name}  |  "
                f"Device: {cfg.device}  |  Batch size: {cfg.batch_size}"
            ),
        )
        self.send_debug_event(
            progress=None,
            message=f"Full config: {cfg.model_dump()}",
        )

        # ── Stage 1: Parse ──────────────────────────────────────────
        self.send_info_event(progress=None, message="---------- Stage 1 · Parse ----------")
        raw = GraphParser(cfg.domains, data_path, host=self).parse()

        # ── Stage 2: Filter ─────────────────────────────────────────
        self.send_info_event(progress=None, message="---------- Stage 2 · Filter ----------")
        raw = GraphFilter(cfg, host=self).filter(raw)

        # ── Stage 3: Embed ──────────────────────────────────────────
        self.send_info_event(progress=None, message="---------- Stage 3 · Embed ----------")
        emb = Embedder(cfg, host=self).embed(raw)

        # ── Stage 4: Build tensors + score ───────────────────────────
        self.send_info_event(progress=None, message="---------- Stage 4 · Build & Score ----------")
        G = TensorGraphBuilder(cfg, host=self).build(raw, emb)

        # ── Stage 5: Save ────────────────────────────────────────────
        self.send_info_event(progress=None, message="---------- Stage 5 · Save ----------")
        graph_id = cfg.model_name.split("/")[-1]
        out_dir  = Path(FEATURES_DIR) / cfg.corpus / graph_id
        GraphSerializer(out_dir, host=self).save(raw, G)

        self.send_info_event(
            progress=None,
            message="===== Knowledge Graph Pipeline completed =====",
        )

        yield from []

    @classmethod
    def from_config(cls, config: Config) -> "FeatureExtractor":
        fe = cls(config=config)
        fe.add_handler(ConsoleLogger("FeatureExtraction"))
        return fe