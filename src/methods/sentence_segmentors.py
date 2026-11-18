import nltk
import spacy
from abc import ABC, abstractmethod
from stanza import Pipeline
from typing import List
from src.shared.registry import registry

class SentenceSegmentor(ABC):

    def __init__(self):
        self.loaded = False

    @abstractmethod
    def base_load(self) -> None:
        ...

    @abstractmethod
    def base_segment(self, doc : str) -> str:
        ...

    def load(self) -> None:

        if self.loaded:
            return

        self.base_load()
        self.loaded = True

    def segment(self, doc : str) -> List[str]:
        self.load()
        return self.base_segment(doc)

SentenceSegmentorRegistry = registry(SentenceSegmentor)

@SentenceSegmentorRegistry.register("stanza")
class StanzaSegmentor(SentenceSegmentor):

    def __init__(self, use_gpu : bool):
        super().__init__()
        self.use_gpu = use_gpu

    def base_load(self):
        self.nlp = Pipeline(lang='en', processors='tokenize', use_gpu=self.use_gpu)

    def base_segment(self, doc : str) -> List[str]:

        doc = self.nlp(doc)
        
        return [
            sentence.text.strip()
            for sentence in doc.sentences
        ]

@SentenceSegmentorRegistry.register("nltk")
class NLTKSegmentor(SentenceSegmentor):

    def base_load(self):
        return

    def base_segment(self, doc : str) -> List[str]:
        return nltk.sent_tokenize(doc, language="english")

@SentenceSegmentorRegistry.register("spacy")
class SpacySegmentor(SentenceSegmentor):

    def __init__(self):
        super().__init__()
        self.nlp = None

    def base_load(self):
        self.nlp = spacy.load("en_core_web_sm")

    def base_segment(self, doc : str) -> List[str]:
        doc = self.nlp(doc)
        return [str(sent) for sent in doc.sents]

if __name__ == '__main__':

    example = "The light of the lava lamp deepens my experience of my environment. It transforms the room into a peaceful refuge."

    print(SentenceSegmentorRegistry.available())

    segmentor = SentenceSegmentorRegistry.create("stanza")
    print(segmentor.segment(example))

    segmentor = SentenceSegmentorRegistry.create("nltk")
    print(segmentor.segment(example))

    segmentor = SentenceSegmentorRegistry.create("spacy")
    print(segmentor.segment(example))
