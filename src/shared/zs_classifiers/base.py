import torch
import asyncio
import json
from transformers import pipeline
from typing import List, TypedDict, Iterable, Hashable, Optional, Dict, Literal, Tuple
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from torch import Tensor
from warnings import warn
from src.helpers import make_openai_request
from returns.result import ResultE
from returns.pipeline import is_successful
from src.shared.env import LLM

class Document(TypedDict):
    id : Hashable
    content : str

class Out(TypedDict):
    label : str
    score : Dict[str, float]

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

        if type_ == 'qwen':
            cls_ = QwenZeroShotClassifier
            kwargs = {  **kwargs, 'model_name' : model }
        elif type_ == 'e5':
            cls_ = E5ZeroShotClassifier
            kwargs = {  **kwargs, 'model_name' : model }
        elif type_ == 'llm':
            cls_ = LLMBasedZeroShotClassifier
        else:
            raise Exception(f"No class coresponding to {name}.")
        
        args = cls_.__init__.__annotations__
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
        device : str = 'cuda' if torch.cuda.is_available() else 'cpu',
        labels : Optional[List[Label]] = None,
        use_descriptions : bool = False
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.labels = labels
        self.use_descriptions = use_descriptions

        print(self.use_descriptions)

        self.model = SentenceTransformer(model_name_or_path=self.model_name)
        self._labels_embeddings : Tensor = None

    @abstractmethod
    def _query(self, sentences : List[str], **kwargs) -> Tensor:
        ...

    @abstractmethod
    def _passage(self, sentences : List[str], **kwargs) -> Tensor:
        ...

    def _embed_labels(self, labels : List[Label]) -> Tuple[List[str], List[int]]:

        sentences = []
        indices = []
        names = []

        for idx, label in enumerate(labels):

            alt_names = label.get('alt_names', None)
            description = label.get('description', None)

            sentences.append(label['name'])
            indices.append(idx)
            names.append('name')

            if description is not None:
                sentences.append(description)
                indices.append(idx)
                names.append('definition')

            if alt_names is not None:
                sentences.extend(alt_names)
                names.extend(alt_names)
                indices.extend([idx] * len(alt_names))
            

        if self._labels_embeddings is None:

            self._labels_embeddings = self._query(
                sentences,
                device=self.device, 
                batch_size=self._DEFAULT_EMB_LABELS_BATCH_SIZE,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

        return names, indices

    def classify(self, 
        docs : Iterable[Document], 
    ) -> List[ResultE[ClassificationOutput]]:

        if not isinstance(docs, list):
            docs = list(docs)

        names, indices = self._embed_labels(self.labels)
        query_embeddings = self._labels_embeddings
        
        document_embeddings = self._passage(
            [doc['content'] for doc in docs], 
            device=self.device, 
            normalize_embeddings=True,
            convert_to_tensor=True
        )

        S = document_embeddings @ query_embeddings.T

        results = []

        for doc, scores in zip(docs, S):

            results.append({
                **doc,
                "labels" : [
                    {
                        "label" : self.labels[idx]['name'],
                        "score" : score.item(),
                        "name" : name
                    }
                    for idx, name, score in zip(indices, names, scores)
                ]
            })

        return [ResultE.from_value(r) for r in results]
    
class QwenZeroShotClassifier(EmbeddingBasedZeroShotClassifier):

    def _query(self, sentences : List[str], **kwargs) -> Tensor:

        return self.model.encode(
            sentences=sentences,
            prompt_name="query",
            **kwargs
        )
    
    def _passage(self, sentences : List[str], **kwargs) -> Tensor:
        
        return self.model.encode(
            sentences=sentences,
            **kwargs
        )
    
class E5ZeroShotClassifier(EmbeddingBasedZeroShotClassifier):

    def _query(self, sentences : List[str], **kwargs) -> Tensor:

        return self.model.encode(
            sentences=[f"query: {sent}" for sent in sentences],
            **kwargs
        )
    
    def _passage(self, sentences : List[str], **kwargs) -> Tensor:
        
        return self.model.encode(
            sentences=[f"passage: {sent}" for sent in sentences],
            **kwargs
        )
    
class HgZeroShotClassifier(ZeroShotClassifier):

    def __init__(self,
        model_name : str,
        labels : List[Label],
        device : str,
        use_descriptions : bool = True       
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.labels = labels
        self.device = device
        self.use_descriptions = self.use_descriptions

        self.pipe = pipeline(
            "zero-shot-classification", 
            model=self.model_name,
            device=device
        )

    def classify(self, docs : List[Document]) -> List[ClassificationOutput]:

        sequence_to_classify = [doc['content'] for doc in docs]

        candidate_labels = [
            (label['description'] if (label['description'] is not None and self.use_descriptions) else label['name'])
            for label in self.labels
        ]
        
        output = self.pipe(sequence_to_classify, candidate_labels, multi_label=True)
    
class LLMBasedZeroShotClassifier(ZeroShotClassifier):

    def __init__(self, 
        llm : LLM,
        prompt_name : str,
        labels : List[Label],
        mode : Literal['seperate', 'combined']
    ) -> None:
        super().__init__()

        self.llm = llm
        self.prompt_name = prompt_name
        self.labels = labels
        self.mode : Literal['seperate', 'combined'] = mode

    _PROMPTS = {
        "levels": (
            "You are an expert in scientific article classification and domain relevance assessment. "
            "Your task is to evaluate the degree of alignment between a research article and a target scientific domain. "
            "The article is defined by its title, abstract, and an optional list of keywords. "
            "The domain is defined by its name and definition. "
            "Analyze the article’s main objectives, methodology, terminology, application area, scientific context, "
            "and primary contributions to determine how strongly the article aligns with the specified domain.\n\n"
            "### Classification Levels:\n"
            "* **3 (Perfect Alignment):** The article’s core problems, objectives, and contributions are exactly what is expected in this domain.\n"
            "* **2 (Strong Alignment):** The article is highly relevant and clearly fits within this domain, although the focus may not be exclusively centered on it.\n"
            "* **1 (Weak Alignment):** The article has limited or indirect topical overlap with the domain but is not a primary fit.\n"
            "* **0 (No Alignment):** The article is completely unrelated to the domain.\n\n"
            "### Important Guidelines:\n"
            "- Use semantic understanding rather than keyword matching alone.\n"
            "- Consider synonyms, related concepts, interdisciplinary overlap, and implicit domain relevance.\n"
            "- Focus on the article’s primary scientific contribution rather than incidental mentions of domain-related terms.\n"
            "- Evaluate methodological and application-level relevance in addition to terminology overlap.\n"
            "- Be strict and consistent when distinguishing between weak and strong alignment.\n\n"
            "### Output Format:\n"
            "Return only one of the following numbers:\n"
            "0\n"
            "1\n"
            "2\n"
            "3\n\n"
            "Do not provide explanations, reasoning, confidence scores, or additional text."
        ),
        "binary": (
            "You are an expert in scientific article classification and domain relevance assessment. "
            "Your task is to evaluate the degree of alignment between a research article and a target scientific domain. "
            "The article is defined by its title, abstract, and an optional list of keywords. "
            "The domain is defined by its name and definition. "
            "Analyze the article’s main objectives, methodology, terminology, application area, scientific context, "
            "and primary contributions to determine whether the article aligns with the specified domain.\n\n"
            "### Classification Levels:\n"
            "* **1 (Aligned):** The article is relevant and fits within this domain (perfect, strong, or weak alignment), showing clear topical, methodological, or application-level overlap.\n"
            "* **0 (Not Aligned):** The article is completely unrelated to the domain.\n\n"
            "### Important Guidelines:\n"
            "- Use semantic understanding rather than keyword matching alone.\n"
            "- Consider synonyms, related concepts, interdisciplinary overlap, and implicit domain relevance.\n"
            "- Focus on the article’s primary scientific contribution rather than incidental mentions of domain-related terms.\n"
            "- Evaluate methodological and application-level relevance in addition to terminology overlap.\n"
            "- Be strict and consistent when distinguishing between aligned and not aligned articles.\n\n"
            "### Output Format:\n"
            "Return only one of the following numbers:\n"
            "0\n"
            "1\n\n"
            "Do not provide explanations, reasoning, confidence scores, or additional text."
        ),
        "score": (
            "You are an expert in scientific article classification and domain relevance assessment. "
            "Your task is to evaluate the degree of alignment between a research article and a target scientific domain. "
            "The article is defined by its title, abstract, and an optional list of keywords. "
            "The domain is defined by its name and definition. "
            "Analyze the article’s main objectives, methodology, terminology, application area, scientific context, "
            "and primary contributions to determine how strongly the article aligns with the specified domain.\n\n"
            "### Classification Levels:\n"
            "Assign an integer score from 0 to 10 based on the following framework:\n"
            "* **9-10 (Perfect Alignment):** The article’s core problems, objectives, and contributions are exactly what is expected in this domain.\n"
            "* **6-8 (Strong Alignment):** The article is highly relevant and clearly fits within this domain, although the focus may not be exclusively centered on it.\n"
            "* **3-5 (Moderate/Weak Alignment):** The article has limited, partial, or indirect topical overlap with the domain but is not a primary fit.\n"
            "* **1-2 (Marginal Alignment):** The article has almost no relevance, with only incidental or superficial mentions of domain-related concepts.\n"
            "* **0 (No Alignment):** The article is completely unrelated to the domain.\n\n"
            "### Important Guidelines:\n"
            "- Use semantic understanding rather than keyword matching alone.\n"
            "- Consider synonyms, related concepts, interdisciplinary overlap, and implicit domain relevance.\n"
            "- Focus on the article’s primary scientific contribution rather than incidental mentions of domain-related terms.\n"
            "- Evaluate methodological and application-level relevance in addition to terminology overlap.\n"
            "- Be strict, nuanced, and consistent when assigning the specific score within the 0-10 range.\n\n"
            "### Output Format:\n"
            "Return only a single integer number from 0 to 10 (e.g., 7).\n\n"
            "Do not provide explanations, reasoning, confidence scores, or additional text."
        )
    }

    def _get_prompt(self, domain : Label, article : str) -> str:

        return (
            f"### Article : \n {article} \n"

            "### Domain : \n"
            f"- Name : {domain['name']}\n"
            f"- Definition : {domain['description']}"
        )
    
    def format_labels_section(self, labels: List[Label]) -> str:
        """Helper function to format the input domains dynamically."""
        formatted = []
        for label in labels:
            desc = f" (Definition: {label['description']})" if label.get('description') else ""
            formatted.append(f"- **{label['name']}**{desc}")
        return "\n".join(formatted)


    def get_multi_label_prompts(self, labels: List[Label]) -> Dict[str, str]:

        domains_block = self.format_labels_section(labels)
        
        return {
            "levels": (
                "You are an expert in scientific article classification and domain relevance assessment.\n"
                "Your task is to evaluate the degree of alignment between a research article and a set of target scientific domains.\n"
                "The article is defined by its title, abstract, and an optional list of keywords.\n\n"
                
                "### Target Domains to Evaluate:\n"
                f"{domains_block}\n\n"
                
                "Analyze the article’s main objectives, methodology, terminology, application area, scientific context, "
                "and primary contributions to determine how strongly the article aligns with each specified domain.\n\n"
                
                "### Classification Levels for Each Domain:\n"
                "* **3 (Perfect Alignment):** The article’s core problems, objectives, and contributions are exactly what is expected in this domain.\n"
                "* **2 (Strong Alignment):** The article is highly relevant and clearly fits within this domain, although the focus may not be exclusively centered on it.\n"
                "* **1 (Weak Alignment):** The article has limited or indirect topical overlap with the domain but is not a primary fit.\n"
                "* **0 (No Alignment):** The article is completely unrelated to the domain.\n\n"
                
                "### Important Guidelines:\n"
                "- Use semantic understanding rather than keyword matching alone.\n"
                "- Consider synonyms, related concepts, interdisciplinary overlap, and implicit domain relevance.\n"
                "- Focus on the article’s primary scientific contribution rather than incidental mentions of domain-related terms.\n"
                "- Evaluate methodological and application-level relevance in addition to terminology overlap.\n"
                "- Be strict and consistent when distinguishing between weak and strong alignment.\n\n"
                
                "### Output Format:\n"
                "Return a valid JSON dictionary where the keys are the domain names exactly as provided, "
                "and the values are the corresponding integer scores (0, 1, 2, or 3).\n"
                "Example format:\n"
                "{\n"
                '  "Domain_A": 3,\n'
                '  "Domain_B": 0\n'
                "}\n\n"
                "Do not provide explanations, reasoning, confidence scores, markdown code blocks (like ```json), or additional text."
            ),
            
            "binary": (
                "You are an expert in scientific article classification and domain relevance assessment.\n"
                "Your task is to evaluate the degree of alignment between a research article and a set of target scientific domains.\n"
                "The article is defined by its title, abstract, and an optional list of keywords.\n\n"
                
                "### Target Domains to Evaluate:\n"
                f"{domains_block}\n\n"
                
                "Analyze the article’s main objectives, methodology, terminology, application area, scientific context, "
                "and primary contributions to determine whether the article aligns with each specified domain.\n\n"
                
                "### Classification Levels for Each Domain:\n"
                "* **1 (Aligned):** The article is relevant and fits within this domain (perfect, strong, or weak alignment), showing clear topical, methodological, or application-level overlap.\n"
                "* **0 (Not Aligned):** The article is completely unrelated to the domain.\n\n"
                
                "### Important Guidelines:\n"
                "- Use semantic understanding rather than keyword matching alone.\n"
                "- Consider synonyms, related concepts, interdisciplinary overlap, and implicit domain relevance.\n"
                "- Focus on the article’s primary scientific contribution rather than incidental mentions of domain-related terms.\n"
                "- Evaluate methodological and application-level relevance in addition to terminology overlap.\n"
                "- Be strict and consistent when distinguishing between aligned and not aligned articles.\n\n"
                
                "### Output Format:\n"
                "Return a valid JSON dictionary where the keys are the domain names exactly as provided, "
                "and the values are the corresponding binary integer scores (0 or 1).\n"
                "Example format:\n"
                "{\n"
                '  "Domain_A": 1,\n'
                '  "Domain_B": 0\n'
                "}\n\n"
                "Do not provide explanations, reasoning, confidence scores, markdown code blocks (like ```json), or additional text."
            ),
            
            "score": (
                "You are an expert in scientific article classification and domain relevance assessment.\n"
                "Your task is to evaluate the degree of alignment between a research article and a set of target scientific domains.\n"
                "The article is defined by its title, abstract, and an optional list of keywords.\n\n"
                
                "### Target Domains to Evaluate:\n"
                f"{domains_block}\n\n"
                
                "Analyze the article’s main objectives, methodology, terminology, application area, scientific context, "
                "and primary contributions to determine how strongly the article aligns with each specified domain.\n\n"
                
                "### Classification Levels for Each Domain:\n"
                "Assign an integer score from 0 to 10 based on the following framework:\n"
                "* **9-10 (Perfect Alignment):** The article’s core problems, objectives, and contributions are exactly what is expected in this domain.\n"
                "* **6-8 (Strong Alignment):** The article is highly relevant and clearly fits within this domain, although the focus may not be exclusively centered on it.\n"
                "* **3-5 (Moderate/Weak Alignment):** The article has limited, partial, or indirect topical overlap with the domain but is not a primary fit.\n"
                "* **1-2 (Marginal Alignment):** The article has almost no relevance, with only incidental or superficial mentions of domain-related concepts.\n"
                "* **0 (No Alignment):** The article is completely unrelated to the domain.\n\n"
                
                "### Important Guidelines:\n"
                "- Use semantic understanding rather than keyword matching alone.\n"
                "- Consider synonyms, related concepts, interdisciplinary overlap, and implicit domain relevance.\n"
                "- Focus on the article’s primary scientific contribution rather than incidental mentions of domain-related terms.\n"
                "- Evaluate methodological and application-level relevance in addition to terminology overlap.\n"
                "- Be strict, nuanced, and consistent when assigning the specific score within the 0-10 range.\n\n"
                
                "### Output Format:\n"
                "Return a valid JSON dictionary where the keys are the domain names exactly as provided, "
                "and the values are the corresponding integer scores from 0 to 10.\n"
                "Example format:\n"
                "{\n"
                '  "Domain_A": 8,\n'
                '  "Domain_B": 2\n'
                "}\n\n"
                "Do not provide explanations, reasoning, confidence scores, markdown code blocks (like ```json), or additional text."
            )
        }

    async def _classify_per_domain(self, doc : str, label : Label) -> ResultE[Out]:
        
        messages = [
            {
                "role" : "system",
                "content" : self._PROMPTS[self.prompt_name]
            },
            {
                "role" : "user",
                "content" : self._get_prompt(label, doc)
            }
        ]

        res = await make_openai_request(
            client=self.llm.async_client,
            model=self.llm.info.name,
            messages=messages
        )

        if not is_successful(res):
            return res
        
        res = res.unwrap()

        answer = res.choices[0].message.content
        score = float(answer.strip('"\'* ').lower())

        return ResultE.from_value(Out(label=label['name'], score=score))
    
    async def classify_combined(self, doc : Document, labels : List[Label]) -> ResultE[ClassificationOutput]:
        
        messages = [
            {
                "role" : "system",
                "content" : self.get_multi_label_prompts(labels)[self.prompt_name]
            },
            {
                "role" : "user",
                "content" : doc['content']
            }
        ]

        res = await make_openai_request(
            client=self.llm.async_client,
            model=self.llm.info.name,
            messages=messages
        )

        if not is_successful(res):
            return res
        
        res = res.unwrap()

        content = res.choices[0].message.content

        try:
            content = json.loads(content)
        except:
            return ResultE.from_failure(Exception("Invalid json response"))
        
        l = []
        
        for label in labels:

            if label['name'] in content:
                l.append(Out(label=label['name'], score=content[label['name']]))
            else:
                return ResultE.from_failure(Exception(f"Inexisistent label {label['name']}"))
            
        return ResultE.from_value(ClassificationOutput(
            id=doc['id'],
            content=doc['content'],
            labels=l
        ))
    
    async def classify_all(self, doc : Document, labels : List[Label]) -> ResultE[ClassificationOutput]:
            
        if self.mode == 'seperate':
        
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
        
        return await self.classify_combined(doc, labels)
        
    
    async def classify_async(self, docs : Iterable[Document]) -> List[ResultE[ClassificationOutput]]:
        return await asyncio.gather(*[self.classify_all(doc, self.labels) for doc in docs])
    
    def classify(self, docs : Iterable[Document]) -> List[ResultE[ClassificationOutput]]:
        return asyncio.run(self.classify_async(docs))

        