import torch
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Literal, Iterator, Any
from definitions import hf_model_dir
from torch import Tensor
from src.shared.registry import registry

EmbedKey = Literal['query', 'passage']

class Embedder(ABC):

    def update_kwargs(kwargs : Dict[str, Any], others : Dict[str, Any]) -> Dict[str, Any]:

        model = hf_model_dir(others.pop('name'))
                
        return { 
            **kwargs, 
            'model' : model,
            **others
        }

    def __init__(self, 
        model : str, 
        device : int = -1
    ) -> None:
        self.model = model
        self.device = device
        self.sbert = None
        self.loaded = False

    def load(self) -> None:

        if self.loaded:
            return

        self.sbert = SentenceTransformer(
            model_name_or_path=self.model, 
            device='cpu' if self.device < 0 else f'cuda:{self.device}'
        )

        self.loaded = True

    def _base_encode(self, sentences : List[str], **kwargs) -> Tensor:

        self.load()

        return self.sbert.encode(
            sentences=sentences,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
            **kwargs
        )

    @abstractmethod
    def query(self, sentences : List[str], **kwargs) -> Tensor:
        ...

    @abstractmethod
    def passage(self, sentences : List[str], **kwargs) -> Tensor:
        ...
    
    def extract(
        self,
        sentences : List[str],
        **kwargs
    ) -> Dict[EmbedKey, Tensor]:
        
        q = self.query(sentences, **kwargs)

        if self.is_basic():
            p = q
        else:
            p = self.passage(sentences, **kwargs)

        return {
            'query' : q,
            'passage' : p
        }

    def extract_query(self, data : Iterator[List[str]]) -> Tensor:

        M = []

        for batch in data:
            B = self.query(batch)
            M.append(B)

        M = torch.cat(M, dim=0)

        return M

    def extract_passage(self, data : Iterator[List[str]]) -> Tensor:
    
        M = []

        for batch in data:
            B = self.passage(batch)
            M.append(B)

        M = torch.cat(M, dim=0)

        return M
    
    def is_basic(self) -> bool:
        return self.__class__ == SbertEmbedder

EmbedderRegistry = registry(Embedder)

@EmbedderRegistry.register("sbert:{name}")
class SbertEmbedder(Embedder):

    def query(self, sentences : List[str], **kwargs) -> Tensor:
        return self._base_encode(sentences=sentences, **kwargs)

    def passage(self, sentences : List[str], **kwargs) -> Tensor:
        return self._base_encode(sentences=sentences, **kwargs)

@EmbedderRegistry.register("qwen:{name}")
class QwenEmbedder(Embedder):

    def query(self, sentences : List[str], **kwargs) -> Tensor:
        return self._base_encode(sentences=sentences, prompt_name="query", **kwargs)
    
    def passage(self, sentences : List[str], **kwargs) -> Tensor:
        return self._base_encode(sentences=sentences, **kwargs)

@EmbedderRegistry.register("e5:{name}")
class E5Embedder(Embedder):

    def query(self, sentences : List[str], **kwargs) -> Tensor:

        return self._base_encode(
            sentences=[f"query: {sent}" for sent in sentences],
            **kwargs
        )
    
    def passage(self, sentences : List[str], **kwargs) -> Tensor:
        
        return self._base_encode(
            sentences=[f"passage: {sent}" for sent in sentences],
            **kwargs
        )

if __name__ == '__main__':

    examples = [
        "computer sience",
        "machine learning"
    ]

    embedder = EmbedderRegistry.create("sbert:sentence-transformers/all-MiniLM-L6-v2", device=0)
    print(embedder.query(examples).shape, embedder.query(examples).device)