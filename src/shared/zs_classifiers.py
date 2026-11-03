import torch
import asyncio
from torch_scatter import scatter
from typing import List, TypedDict, Iterable, Hashable, Optional, Dict
from abc import ABC, abstractmethod
from torch import Tensor
from warnings import warn
from returns.result import ResultE
from returns.pipeline import is_successful
from src.shared.embedders import Embedder
from src.shared.platforms import LLM

class Document(TypedDict):
    id : Hashable
    content : str

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

class ZeroShotClassifier(ABC):

    @abstractmethod
    def classify(self, docs : Iterable[Document]) -> List[ResultE[ClassificationOutput]]:
        ... 

    @staticmethod
    def from_name(name : str, **kwargs) -> 'ZeroShotClassifier':
        
        type_, model = name.split(':')

        if type_ == 'sbert':
            cls_ = EmbeddingBasedZeroShotClassifier
            kwargs = {  **kwargs, 'model_name' : model }
        elif type_ == 'llm':
            cls_ = LLMBasedZeroShotClassifier
            kwargs = { **kwargs, 'llm' : LLM(name=model) }
        else:
            raise Exception(f"No class coresponding to {name}.")
        
        args = cls_.__init__.__annotations__.copy()
        args.pop('return', None)
        args = args.keys()

        args2remove = []

        for arg in kwargs:

            if arg not in args:
                warn(f"Argument \"{arg}\" is not being used for class {cls_}")
                args2remove.append(arg)

        for arg in args2remove:
            kwargs.pop(arg)

        return cls_(**kwargs)

class EmbeddingBasedZeroShotClassifier(ZeroShotClassifier):

    _DEFAULT_EMB_LABELS_BATCH_SIZE : int = 8

    def __init__(self, 
        model_name : str, 
        device : str,
        labels : Optional[List[Label]],
        model_kwargs : Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.labels = labels

        self.embedder = Embedder.from_name(model_name=model_name, device=device, **(model_kwargs or {}))

        self._labels_embeddings : Tensor = None
        self._labels_indices : Tensor = None

    def _query(self, sentences : List[str], **kwargs) -> Tensor:
        return self.embedder.query(sentences, **kwargs)

    def _passage(self, sentences : List[str], **kwargs) -> Tensor:
        return self.embedder.passage(sentences, **kwargs)

    def _embed_labels(self, labels : List[Label]) -> None:

        if self._labels_embeddings is not None:
            return

        sentences = []
        indices = []

        for idx, label in enumerate(labels):

            alt_names = label.get('alt_names', None)
            description = label.get('description', None)

            sentences.append(label['name'])
            indices.append(idx)

            if description is not None:
                sentences.append(description)
                indices.append(idx)

            if alt_names is not None:
                sentences.extend(alt_names)
                indices.extend([idx] * len(alt_names))
            
        self._labels_embeddings = self._query(
            sentences,
            device=self.device, 
            batch_size=self._DEFAULT_EMB_LABELS_BATCH_SIZE,
        )

        self._labels_indices = torch.tensor(indices).long()

    def classify(self, docs : Iterable[Document]) -> List[ResultE[ClassificationOutput]]:

        if not isinstance(docs, list):
            docs = list(docs)

        ### Embed labels
        self._embed_labels(self.labels)
        query_embeddings = self._labels_embeddings
        query_indices = self._labels_indices

        ### Embed documents
        document_embeddings = self._passage(
            [doc['content'] for doc in docs], 
            device=self.device, 
        )

        ### Compute dot product
        S = document_embeddings @ query_embeddings.T

        ### Max Aggregation
        S = scatter(
            src=S, 
            index=query_indices.to(self.device), 
            dim=1, 
            dim_size=len(self.labels),
            reduce='max'
        )

        results = []

        for doc, scores in zip(docs, S):

            labels = []

            for idx, score in enumerate(scores):

                label_name = self.labels[idx]['name']

                labels.append({
                    "label" : label_name,
                    "score" : score.item()
                })

            results.append({
                **doc,
                "labels" : labels
            })

        return [ResultE.from_value(r) for r in results]
    
class LLMBasedZeroShotClassifier(ZeroShotClassifier):

    _PROMPTS = {
        "levels": "DomainRelevanceLevels",
        "binary": "DomainRelevanceBinary",
        "score": "DomainRelevanceScore"
    }

    def __init__(self, 
        llm : LLM,
        prompt_name : str,
        labels : List[Label],
    ) -> None:
        super().__init__()

        self.llm = llm
        self.prompt_name = prompt_name
        self.labels = labels

    async def _classify_per_domain(self, doc : str, label : Label) -> ResultE[Out]:

        result = await self.llm.make_async_request(
            prompt_name=self._PROMPTS[self.prompt_name],
            input={
                "content" : doc,
                "domain_name" : label['name'],
                "domain_definition" : label['description']
            },
            lang='en',
            validate_input=True,
            validate_output=True
        )

        if not is_successful(result):
            return result
        
        result = result.unwrap()

        try:
            norm = 10 if self.prompt_name == 'score' else 1
            return ResultE.from_value(Out(label=label['name'], score=result['score'] / norm))
        except Exception as e:
            return ResultE.from_failure(e)
    
    async def classify_all(self, doc : Document, labels : List[Label]) -> ResultE[ClassificationOutput]:
                    
        results = await asyncio.gather(*[self._classify_per_domain(doc['content'], label) for label in labels])
        
        unwrapped = []

        for r in results:

            if not is_successful(r):
                return r
            
            unwrapped.append(r.unwrap())

        return ResultE.from_value(ClassificationOutput(
            id=doc['id'],
            content=doc['content'],
            labels=unwrapped
        ))        
    
    async def classify_async(self, docs : Iterable[Document]) -> List[ResultE[ClassificationOutput]]:
        return await asyncio.gather(*[self.classify_all(doc, self.labels) for doc in docs])
    
    def classify(self, docs : Iterable[Document]) -> List[ResultE[ClassificationOutput]]:
        return asyncio.run(self.classify_async(docs))