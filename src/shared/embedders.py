import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Literal, Iterator
from torch import Tensor

EmbedKey = Literal['query', 'passage']

class Embedder:

    def __init__(self, model : SentenceTransformer) -> None:
        self.model = model

    def _base_encode(self, sentences : List[str], **kwargs) -> Tensor:

        return self.model.encode(
            sentences=sentences,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
            **kwargs
        )

    def query(self, sentences : List[str], **kwargs) -> Tensor:
        return self._base_encode(sentences=sentences, **kwargs)

    def passage(self, sentences : List[str], **kwargs) -> Tensor:
        return self._base_encode(sentences=sentences, **kwargs)
    
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
        return self.__class__ == Embedder
    
    @staticmethod
    def from_name(model_name : str, device : str) -> 'Embedder':

        model = SentenceTransformer(model_name, device=device)

        if 'e5' in model_name.lower():
            return E5Embedder(model)
        
        if 'qwen' in model_name.lower():
            return QwenEmbedder(model)
        
        return Embedder(model)

class QwenEmbedder(Embedder):

    def query(self, sentences : List[str], **kwargs) -> Tensor:
        return self._base_encode(sentences=sentences, prompt_name="query", **kwargs)
    
    def passage(self, sentences : List[str], **kwargs) -> Tensor:
        return self._base_encode(sentences=sentences, **kwargs)
    
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