import asyncio
import torch
from torch_scatter import scatter
from typing import List, TypedDict, Iterable, Hashable, Optional, Dict, Any, Tuple, Callable
from abc import ABC, abstractmethod
from torch import Tensor
from src.shared.registry import registry
from returns.result import ResultE
from returns.pipeline import is_successful
from src.methods.embedders import Embedder, EmbedderRegistry
from src.shared.platforms import LLM


class Out(TypedDict):
    label : str
    score : float

class ClassificationOutput(TypedDict):
    id : Hashable
    content : str
    labels : List[Out]

class Label(TypedDict):
    name : str
    description : Optional[str]
    alt_names : Optional[List[str]]

class Document(TypedDict):
    id : Hashable
    content : str
    label : Label

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Hashable, Iterable, Iterator, List, Optional, Tuple, TypedDict

from returns.result import Failure, ResultE, Success


class Out(TypedDict):
    label: str
    score: float


class ClassificationOutput(TypedDict):
    id: Hashable
    content: str
    labels: List[Out]


class Label(TypedDict):
    name: str
    description: Optional[str]
    alt_names: Optional[List[str]]


class Document(TypedDict):
    id: Hashable
    content: str
    label: Label


Pair = Tuple[Document, Label]


class ZeroShotClassifier(ABC):

    def __init__(self, batch_size : int) -> None:
        super().__init__()
        self.batch_size = batch_size

    @abstractmethod
    def classify_pair(self, inputs: List[Pair]) -> List[ResultE[Out]]:
        ...

    def classify(
        self,
        docs: Iterable[Document],
        labels: List[Label],
    ) -> List[ResultE[ClassificationOutput]]:
        
        docs = list(docs)

        if not labels:
            return [Success(self._make_output(doc, [])) for doc in docs]

        pairs = self._build_pairs(docs, labels)
        results = self._score_pairs(pairs)
        return self._aggregate(results)

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _build_pairs(docs: List[Document], labels: List[Label]) -> List[Pair]:
        return [(doc, label) for doc in docs for label in labels]

    def _iter_batches(self, pairs: List[Pair]) -> Iterator[List[Pair]]:

        step = self.batch_size or max(len(pairs), 1)

        for start in range(0, len(pairs), step):
            yield pairs[start : start + step]

    def _score_pairs(
        self, 
        pairs: List[Pair], 
    ) -> List[Tuple[Document, ResultE[Out]]]:

        output = []
        
        for batch in self._iter_batches(pairs):

            outs = self.classify_pair(batch)

            for (doc, _), out in zip(batch, outs, strict=True):
                output.append((doc, out))

        return output

    def _aggregate(
        self,
        results: List[Tuple[Document, ResultE[Out]]],
    ) -> List[ResultE[ClassificationOutput]]:
        
        docs: Dict[Hashable, Document] = {}
        scores: Dict[Hashable, List[Out]] = {}
        errors: Dict[Hashable, Exception] = {}

        for doc, out in results:

            key = doc["id"]
            docs.setdefault(key, doc)
            scores.setdefault(key, [])

            if key in errors:
                continue

            if is_successful(out):
                scores[key].append(out.unwrap())
            else:
                errors[key] = out.failure()

        return [
            Failure(errors[key]) if key in errors
            else Success(self._make_output(docs[key], scores[key]))
            for key in docs
        ]

    @staticmethod
    def _make_output(doc: Document, labels: List[Out]) -> ClassificationOutput:
        return ClassificationOutput(id=doc["id"], content=doc["content"], labels=labels)

ZeroShotClassifierRegistry = registry(ZeroShotClassifier)

@ZeroShotClassifierRegistry.register("embedder:{model_name}")
class EmbeddingBasedZeroShotClassifier(ZeroShotClassifier):

    _DEFAULT_EMB_LABELS_BATCH_SIZE: int = 8

    def __init__(self, model_name: str, device: int, batch_size: int) -> None:

        super().__init__(batch_size=batch_size)

        self.model_name = model_name
        self.device = device
        self.embedder = EmbedderRegistry.create(model_name, device=device)

        # label name -> (n_sentences, dim) embeddings of its name/description/alt names
        self._label_cache: Dict[str, Tensor] = {}

    @staticmethod
    def update_kwargs(kwargs: Dict[str, Any], others: Dict[str, Any]) -> Dict[str, Any]:
        return {**kwargs, **others}

    # ------------------------------------------------------------------ embedder

    def _query(self, sentences: List[str], **kwargs) -> Tensor:
        return self.embedder.query(sentences, **kwargs)

    def _passage(self, sentences: List[str], **kwargs) -> Tensor:
        return self.embedder.passage(sentences, **kwargs)

    # ---------------------------------------------------------- label embeddings

    @staticmethod
    def _label_sentences(label: Label) -> List[str]:

        sentences = [label["name"]]

        if label.get("description") is not None:
            sentences.append(label["description"])

        sentences.extend(label.get("alt_names") or [])

        return sentences

    def _embed_missing_labels(self, labels: List[Label]) -> None:

        missing = [label for label in labels if label["name"] not in self._label_cache]

        if not missing:
            return

        per_label = [self._label_sentences(label) for label in missing]

        embeddings = self._query(
            [s for sentences in per_label for s in sentences],
            batch_size=self._DEFAULT_EMB_LABELS_BATCH_SIZE,
        )

        chunks = torch.split(embeddings, [len(s) for s in per_label])

        for label, chunk in zip(missing, chunks):
            self._label_cache[label["name"]] = chunk

    def _get_label_embeddings(self, names: List[str]) -> Tuple[Tensor, Tensor]:

        chunks = [self._label_cache[name] for name in names]

        embeddings = torch.cat(chunks, dim=0)

        indices = torch.cat([
            torch.full((len(chunk),), pos, dtype=torch.long)
            for pos, chunk in enumerate(chunks)
        ])

        return embeddings, indices

    # ------------------------------------------------------------ classification

    def classify_pair(self, inputs: List[Pair]) -> List[ResultE[Out]]:

        if not inputs:
            return []

        docs, labels = zip(*inputs)

        # Embed Labels & Retriev from cache
        self._embed_missing_labels(list(labels))
        label_embeddings, pair_indices = self._get_label_embeddings(
            [label["name"] for label in labels]
        )

        # Document embeddings, one row per pair
        doc_embeddings = self._passage([doc["content"] for doc in docs])

        label_embeddings = label_embeddings.to(doc_embeddings.device)
        pair_indices = pair_indices.to(doc_embeddings.device)

        # Similarity of each label sentence with its own pair's document
        sims = (doc_embeddings[pair_indices] * label_embeddings).sum(dim=-1)

        # Max over each label's sentences -> one score per pair
        scores = scatter(
            src=sims,
            index=pair_indices,
            dim=0,
            dim_size=len(inputs),
            reduce="max",
        ).tolist()

        return [
            Success(Out(label=label["name"], score=score))
            for label, score in zip(labels, scores)
        ]

@ZeroShotClassifierRegistry.register("llm:{llm}:{prompt_name}")
class LLMBasedZeroShotClassifier(ZeroShotClassifier):

    _PROMPTS = {
        "levels": "DomainRelevanceLevels",
        "binary": "DomainRelevanceBinary",
        "score": "DomainRelevanceScore",
    }

    @staticmethod
    def update_kwargs(kwargs: Dict[str, Any], others: Dict[str, Any]) -> Dict[str, Any]:
        llm = LLM(name=others.pop("llm"))
        return {**kwargs, **others, "llm": llm}

    def __init__(self, llm: LLM, prompt_name: str, batch_size: int) -> None:

        super().__init__(batch_size=batch_size)

        if prompt_name not in self._PROMPTS:

            raise ValueError(
                f"Unknown prompt_name {prompt_name!r}, expected one of {list(self._PROMPTS)}"
            )

        self.llm = llm
        self.prompt_name = prompt_name

    # ------------------------------------------------------------ classification

    async def _classify_one(self, doc: Document, label: Label) -> ResultE[Out]:

        """Ask the LLM how relevant `doc` is to `label`; LLM errors come back as Failures."""
        result = await self.llm.make_async_request(
            prompt_name=self._PROMPTS[self.prompt_name],
            input={
                "content": doc["content"],
                "domain_name": label["name"],
                "domain_definition": label.get("description") or "NOT PROVIDED RELY on `domain name`.",
            },
            lang="en",
            validate_input=True,
            validate_output=True,
        )

        if not is_successful(result):
            return result

        try:
            norm = 10 if self.prompt_name == "score" else 1
            return Success(Out(label=label["name"], score=result.unwrap()["score"] / norm))
        except Exception as e:
            return Failure(e)

    async def _classify_pair_async(self, inputs: List[Pair]) -> List[ResultE[Out]]:
        return await asyncio.gather(*[self._classify_one(doc, label) for doc, label in inputs])

    def classify_pair(self, inputs: List[Pair]) -> List[ResultE[Out]]:

        if not inputs:
            return []
        
        return asyncio.run(self._classify_pair_async(inputs))

if __name__ == '__main__':

    labels = [
        Label(name="machine learning", description=None, alt_names=None),
        Label(name="computer science", description=None, alt_names=None)
    ]

    examples = [
        Document(id=0, content="This about machine learning."),
        Document(id=1, content="This about computer science.")
    ]

    classifier = ZeroShotClassifierRegistry.create(
        "embedder:sbert:sentence-transformers/all-MiniLM-L6-v2",
        device=-1,
        batch_size=8
    )

    print(classifier.classify(examples, labels=labels))

    classifier = ZeroShotClassifierRegistry.create(
        "llm:deepseek-flash:score",
        device=-1,
        batch_size=8
    )

    print(classifier.classify(examples, labels=labels))