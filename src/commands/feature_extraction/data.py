import torch
import copy
from typing import Callable, Dict, Tuple, Type, List
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType, EdgeType

EdgeType = Tuple[str, str, str]
Generator = Callable[[HeteroData], Dict[str, Tensor]] 

def _parse_key(key) -> EdgeType:

    if isinstance(key, str):
        key = tuple(key.split("__"))
    if not (isinstance(key, tuple) and len(key) == 3):
        raise ValueError(f"Invalid edge type {key!r}")
    return key


def _parse_key(key) -> EdgeType:

    if isinstance(key, str):
        key = tuple(key.split("__"))
    if not (isinstance(key, tuple) and len(key) == 3):
        raise ValueError(f"Invalid edge type {key!r}")
    
    return key


def make_lazy_heterodata(
    relations: Dict[EdgeType, Generator] = None,
    name: str = "LazyHeteroData",
    **kwargs: Generator,
) -> Type[HeteroData]:

    generators = {_parse_key(k): fn for k, fn in {**(relations or {}), **kwargs}.items()}

    # Per-instance state in __dict__ (HeteroData.__setattr__ would put it in the global store)
    def _removed(self) -> set:
        return self.__dict__.setdefault("_lazy_removed", set())

    def _lazy_active(self) -> List[EdgeType]:
        removed = self._removed()
        return [et for et in generators if et not in removed]

    def _is_pending(self, edge_type) -> bool:
        """Lazy, not deleted, not generated yet."""
        return (edge_type in generators
                and edge_type not in self._removed()
                and "edge_index" not in self._edge_store_dict.get(edge_type, {}))

    @property
    def edge_types(self) -> List[EdgeType]:
        types = list(self._edge_store_dict.keys())
        return types + [et for et in self._lazy_active() if et not in self._edge_store_dict]

    def __getitem__(self, *args):
        # Canonicalise first so data['HASRELATIONWITH'] / data['Entity', 'Entity'] also trigger it
        key = self._to_canonical(*args)
        if isinstance(key, tuple) and self._is_pending(key):
            self._materialize(key)
        return HeteroData.__getitem__(self, key)

    def __delitem__(self, *args):
        key = self._to_canonical(*args)
        if key in generators:
            self._removed().add(key)      # stop listing / regenerating it
            if key not in self._edge_store_dict:
                return                     # never generated: nothing stored
        HeteroData.__delitem__(self, key)

    def __copy__(self):
        out = HeteroData.__copy__(self)
        out.__dict__["_lazy_removed"] = set(self._removed())  # don't share with original
        out.__dict__.pop("_lazy_building", None)
        return out

    def _materialize(self, edge_type):
        if not self._is_pending(edge_type):
            return
        building = self.__dict__.setdefault("_lazy_building", set())
        if edge_type in building:
            raise RecursionError(f"Cyclic lazy relation at {edge_type}")
        building.add(edge_type)
        try:
            attrs = generators[edge_type](self)
        finally:
            building.discard(edge_type)
        store = HeteroData.__getitem__(self, edge_type)
        for attr, value in attrs.items():
            store[attr] = value

    def materialize(self, *edge_types):
        """Generate all (or the given) lazy relations."""
        for et in (edge_types or self._lazy_active()):
            self._materialize(_parse_key(et))
        return self

    def subgraph(self, subset_dict: Dict[NodeType, Tensor]):
        """Subgraph of the base relations; lazy ones stay listed and are
        regenerated from the subgraph's own data on first access."""
        tmp = copy.copy(self)
        for et in generators:
            del tmp[et]                    # drop stored + hide from edge_types
        out = HeteroData.subgraph(tmp, subset_dict)
        out.__dict__["_lazy_removed"] = set(self._removed())
        return out

    return type(name, (HeteroData,), {
        "_removed": _removed,
        "_lazy_active": _lazy_active,
        "_is_pending": _is_pending,
        "edge_types": edge_types,
        "__getitem__": __getitem__,
        "__delitem__": __delitem__,
        "__copy__": __copy__,
        "_materialize": _materialize,
        "materialize": materialize,
        "subgraph": subgraph,
    })


# ---------------- Use case ----------------

def _num_nodes(data, nt):
    store = HeteroData.__getitem__(data, nt)
    return store.node_id.size(0) if "node_id" in store else store.num_nodes


def _triple_lookup(data, rel, dst):
    """For each Triple, index of its dst node via rel (-1 if missing)."""
    ei = data["Triple", rel, dst].edge_index
    out = torch.full((_num_nodes(data, "Triple"),), -1, dtype=torch.long, device=ei.device)
    out[ei[0]] = ei[1]
    return out


def entity_relation_entity(data : HeteroData) -> Dict[str, Tensor]:

    head = _triple_lookup(data, "HAS-HEAD", "Entity")
    tail = _triple_lookup(data, "HAS-TAIL", "Entity")
    pred = _triple_lookup(data, "HAS-PREDICATE", "Predicate")
    article = _triple_lookup(data, "BELONGSTO", "Article")

    valid = (head >= 0) & (tail >= 0) & (pred >= 0) & (article >= 0)

    edge_attr = None

    if 'x' in data['Predicate']:
        edge_attr = data["Predicate"].x[pred[valid]]

    return {
        "edge_index": torch.stack([head[valid], tail[valid]], dim=0),
        "edge_attr": edge_attr,
        "article_id" : article[valid],
        "edge_id" : torch.arange(valid.sum())
    }


def entity_appears_in_article(data : HeteroData) -> Dict[str, Tensor]:

    n_ent, n_tri, n_art = (_num_nodes(data, t) for t in ("Entity", "Triple", "Article"))

    h = data["Triple", "HAS-HEAD", "Entity"].edge_index
    t = data["Triple", "HAS-TAIL", "Entity"].edge_index
    b = data["Triple", "BELONGSTO", "Article"].edge_index

    et = torch.cat([h, t], dim=1).flip(0)  # Entity -> Triple

    E2T = torch.sparse_coo_tensor(et, torch.ones(et.size(1)), (n_ent, n_tri)).coalesce()
    T2A = torch.sparse_coo_tensor(b, torch.ones(b.size(1)), (n_tri, n_art)).coalesce()
    E2A = torch.sparse.mm(E2T, T2A).coalesce()

    return {
        "edge_index": E2A.indices(),
        "edge_id" : torch.arange(n_tri)
    }


KGData = make_lazy_heterodata({
    ("Entity", "HASRELATIONWITH", "Entity"): entity_relation_entity,
    ("Entity", "APPEARIN", "Article"): entity_appears_in_article,
}, name="KGData")