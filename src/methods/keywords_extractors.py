import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Iterator, Any, Tuple, Optional
from definitions import hf_model_dir
from src.shared.registry import registry

class KeyphraseExtractor(ABC):

    def update_kwargs(kwargs : Dict[str, Any], others : Dict[str, Any]) -> Dict[str, Any]:

        model = hf_model_dir(others.pop('name'))

        return {
            **kwargs,
            'model' : model,
            **others
        }

    def __init__(self,
        model : str,
        device : int = -1,
        batch_size : Optional[int] = None
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.loaded = False

    @property
    def torch_device(self) -> str:
        if isinstance(self.device, int):
            return 'cpu' if self.device < 0 else f'cuda:{self.device}'
        return str(self.device)  # also accept 'cuda', 'cuda:0', torch.device(...) like the old class

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def extract(self, docs : List[str], **kwargs) -> List[List[str]]:
        """Takes a list of documents, returns one list of keyphrases per document."""
        ...

    def extract_all(self, data : Iterator[List[str]], **kwargs) -> List[List[str]]:

        K = []

        for batch in data:
            K.extend(self.extract(batch, **kwargs))

        return K


KeyphraseExtractorRegistry = registry(KeyphraseExtractor)


@KeyphraseExtractorRegistry.register("kpt:{name}")
class KeyPhraseTransformerExtractor(KeyphraseExtractor):

    generation_kwargs : Dict[str, Any] = dict(
        num_beams=2,
        max_length=512,
        repetition_penalty=2.5,
        length_penalty=1,
        early_stopping=True,
        top_p=0.95,
        top_k=50,
        num_return_sequences=1,
    )

    def __init__(self,
        model : str,
        device : int = -1,
        batch_size : Optional[int] = None,
        model_type : str = "t5",
        text_block_size : int = 64
    ) -> None:
        super().__init__(model=model, device=device, batch_size=batch_size)
        self.model_type = model_type
        self.text_block_size = text_block_size
        self.kpt = None

    def load(self) -> None:

        if self.loaded:
            return

        from keyphrasetransformer import KeyPhraseTransformer

        self.kpt = KeyPhraseTransformer(model_type=self.model_type, model_name=self.model)
        self.kpt.model.to(self.torch_device)
        self.kpt.model.eval()

        self.loaded = True

    @torch.inference_mode()
    def _generate(self, paragraphs : List[str]) -> List[List[str]]:
        """One generation per paragraph, wrapped in a list (the format kpt.process_outputs expects)."""

        preds = []
        step = self.batch_size or len(paragraphs)  # None -> all paragraphs in one generate call

        for i in range(0, len(paragraphs), step):

            chunk = paragraphs[i:i + step]

            inputs = self.kpt.tokenizer(
                chunk,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
            ).to(self.torch_device)

            generated_ids = self.kpt.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **self.generation_kwargs
            )

            decoded = self.kpt.tokenizer.batch_decode(
                generated_ids.cpu(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            preds.extend([d] for d in decoded)

        return preds

    def _split(self, docs : List[str], text_block_size : int) -> Tuple[List[int], List[str]]:

        ids, paragraphs = [], []

        for i, doc in enumerate(docs):

            paras = self.kpt.split_into_paragraphs(
                doc=doc, max_tokens_per_para=text_block_size
            )

            for par in paras:
                ids.append(i)
                paragraphs.append(par)

        return ids, paragraphs

    def extract(self, docs : List[str], text_block_size : int = None, **kwargs) -> List[List[str]]:

        self.load()

        if not docs:
            return []

        ids, paragraphs = self._split(docs, text_block_size or self.text_block_size)
        preds = self._generate(paragraphs) if paragraphs else []

        per_doc : List[List[List[str]]] = [[] for _ in docs]

        for idx, pred in zip(ids, preds):
            per_doc[idx].append(pred)

        return [
            self.kpt.filter_outputs(self.kpt.process_outputs(outputs), doc) if outputs else []
            for outputs, doc in zip(per_doc, docs)
        ]


if __name__ == '__main__':

    examples = [
        "Machine learning is a field of artificial intelligence that uses statistical techniques "
        "to give computer systems the ability to learn from data.",
        "Computer science is the study of computation, information, and automation."
    ]

    extractor = KeyphraseExtractorRegistry.create("kpt:snrspeaks/KeyPhraseTransformer", device=0)
    print(extractor.extract(examples))