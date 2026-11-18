import re
import torch
from src.shared.registry import registry
from abc import ABC, abstractmethod
from typing import TypedDict, List, Any
from definitions import hf_model_dir
from typing import Dict
from src.shared.platforms import LLM, RequestReturnType
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from returns.result import Result

class Acronym(TypedDict):
    acronym : str
    expansion : str
    score : float

class AcronymExtractor(ABC):

    def __init__(self):
        self.loaded = False

    @abstractmethod
    def supports_gpu(self) -> bool:
        ...

    @abstractmethod
    def base_load(self) -> None:
        ...

    @abstractmethod
    def base_extract(self, texts : List[str]) -> List[RequestReturnType[List[Acronym]]]:
        ...

    def load(self) -> None:

        if self.loaded:
            return

        self.base_load()
        self.loaded = True

    def extract(self, texts : List[str]) -> List[RequestReturnType[List[Acronym]]]:
        self.load()
        return self.base_extract(texts)

AcronymExtractorRegistry = registry(AcronymExtractor)

@AcronymExtractorRegistry.register("llm:{llm}")
class LLMAcronymExtractor(AcronymExtractor):

    def update_kwargs(kwargs : Dict[str, Any], others : Dict[str, Any]) -> Dict[str, Any]:
                    
        return { 
            **kwargs, 
            'llm' : LLM(others['llm'])
        }

    def __init__(self, llm : LLM):
        super().__init__()
        self.llm = llm

    def supports_gpu(self):
        return False

    def base_load(self):
        pass

    def base_extract(self, texts : List[str]) -> List[RequestReturnType[List[Acronym]]]:

        outputs = self.llm.batch_make_async_request(
            prompt_name='AcronymsExtraction',
            inputs=[
                { "text" : text }
                for text in texts
            ]
        )

        return [
            (out
                .map(lambda x : x['acronyms'])
                .map(lambda x : [{**item, 'score' : 1.0} for item in x])
            )
            for out in outputs
        ]

@AcronymExtractorRegistry.register("qa:{model_name}")
class QAAcronymExtractor(AcronymExtractor):

    def update_kwargs(kwargs : Dict[str, Any], others : Dict[str, Any]) -> Dict[str, Any]:
                
        return { 
            **kwargs, 
            **others
        }

    def __init__(self, model_name : str, device : int, batch_size : int = 1):

        super().__init__()

        self.batch_size = batch_size
        self.model_name = model_name
        self.device = device

        self.tokenizer = None
        self.model = None
        
        self.expansion_pattern = r"\([^()]*\b\w+\b[^()]*\b\w+\b[^()]*\)"

        self.base_acronym_patterns = [
            r"[A-Z0-9-]+s?",
            r"[A-Z]+[a-z-0-9-]+[A-Z]+s?",
            r"[a-z-0-9-][A-Z]+s?"
        ]

        self.acronym_patterns = [fr"\(\s*{pattern}\s*\)" for pattern in self.base_acronym_patterns]
        self.acronym_patterns.extend([fr"{pattern}\s*{self.expansion_pattern}" for pattern in self.base_acronym_patterns])

    def supports_gpu(self) -> bool:
        return True

    def base_load(self) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_dir(self.model_name))
        self.model = AutoModelForQuestionAnswering.from_pretrained(hf_model_dir(self.model_name))

    def is_valid_acronym(self, candidate : str) -> bool:
    
        candidate = candidate.strip().replace('-','')
            
        return  not candidate.isnumeric() \
            and len(candidate) > 1
    
    def get_acronym(self, candidate : str) -> str:

        if candidate.startswith('('):
            candidate = candidate.lstrip('(').rstrip(')')
        else:
            candidate = re.sub(self.expansion_pattern, "", candidate)

        return candidate.strip()

    def parse_acronyms_type1(self, sentence : str) -> List[str]:

        ### Preprocessing
        sentence = re.sub(r"\s+", " ", sentence)        

        ### Extract Acronyms
        candidates = set()
        
        for pattern in self.acronym_patterns:
            candidates.update(re.findall(pattern, sentence))

        ### Filter-out invalid acronyms
        candidates = list(
            filter(
                self.is_valid_acronym,
                map(self.get_acronym,candidates)
            )
        )

        return candidates

    def answer(self, pairs: list[tuple[str, str]]) -> list[dict]:
        
        results = []
        batch_size = self.batch_size

        for i in range(0, len(pairs), batch_size):

            batch = pairs[i:i + batch_size]
            questions = [q for q, _ in batch]
            contexts = [c for _, c in batch]

            inputs = self.tokenizer(
                questions,
                contexts,
                return_tensors="pt",
                truncation=True,
                padding=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Mask padding so it doesn't affect the softmax or the argmax
            pad_mask = ~inputs["attention_mask"].bool()
            start_logits = outputs.start_logits.masked_fill(pad_mask, float("-inf"))
            end_logits = outputs.end_logits.masked_fill(pad_mask, float("-inf"))

            start_probs = torch.softmax(start_logits, dim=-1)  # (B, L)
            end_probs = torch.softmax(end_logits, dim=-1)      # (B, L)

            start_idx = start_probs.argmax(dim=-1)  # (B,)
            end_idx = end_probs.argmax(dim=-1)      # (B,)

            scores = (
                start_probs.gather(1, start_idx.unsqueeze(1)).squeeze(1) *
                end_probs.gather(1, end_idx.unsqueeze(1)).squeeze(1)
            )

            for b in range(len(batch)):
                s = start_idx[b].item()
                e = end_idx[b].item()

                answer = self.tokenizer.decode(
                    inputs["input_ids"][b, s:e + 1],
                    skip_special_tokens=True
                )

                results.append({
                    "answer": answer,
                    "score": scores[b].item(),
                    "start": s,
                    "end": e
                })

        return results

    def question(self, acronym : str) -> str:
        return f"What is the expansion of the acronym \"{acronym}\" ?"

    def base_extract(self, texts : List[str]) -> List[RequestReturnType[List[Acronym]]]:

        candidates = [
            (i, text, self.question(candidate), candidate)
            for i, text in enumerate(texts)
            for candidate in self.parse_acronyms_type1(text)
        ]

        answers = self.answer([
            (question, text)
            for _, text, question, _ in candidates
        ])

        results = [[] for  _ in texts]

        for (i, _, _, candidate), answer in zip(candidates, answers):

            results[i].append({
                "acronym" : candidate,
                "expansion" : answer["answer"],
                "score" : answer['score']
            })

        results = [Result.from_value(r) for r in results]

        return results


if __name__ == '__main__':

    example = ["Machine Learning (ML) is used in Advanced Cancer Detection (ACD)"]
    extractor = AcronymExtractorRegistry.create("qa:timpal0l/mdeberta-v3-base-squad2", device=-1)

    print(extractor.__class__)
    print(extractor.extract(example))
