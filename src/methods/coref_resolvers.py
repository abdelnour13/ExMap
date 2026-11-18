import spacy
import re
import coreferee
from src.shared.registry import registry
from src.rpc.interface import load_predictor_from_path, allennlp_predict
from abc import ABC, abstractmethod
from typing import Dict

class CorefResolver(ABC):

    def __init__(self):
        self.loaded = False

    @abstractmethod
    def supports_gpu(self) -> bool:
        ...

    @abstractmethod
    def base_load(self) -> None:
        ...

    @abstractmethod
    def base_resolve(self, doc : str) -> str:
        ...

    def load(self) -> None:

        if self.loaded:
            return

        self.base_load()
        self.loaded = True

    def resolve(self, doc : str) -> str:
        self.load()
        return self.base_resolve(doc)

CorefResolverRegistry = registry(CorefResolver)

@CorefResolverRegistry.register("spanbert-large")
class SpanBERTResolver(CorefResolver):

    MODELS = {
        "spanbert-large" : "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
    }

    def __init__(self, device : int):
        super().__init__()
        self.device = device
        self.model = "spanbert-large"

    def supports_gpu(self) -> bool:
        return True

    def base_load(self):
        load_predictor_from_path(self.MODELS[self.model], self.device)

    def base_resolve(self, doc : str) -> str:

        result : Dict = allennlp_predict(self.MODELS[self.model], self.device, doc)
        
        ### Resolve The Text
        tokens = result['document']
        clusters = result["clusters"]

        ### Build Replacements Dictionary
        replacements = {}

        for cluster in clusters:

            main_start, main_end = cluster[0]
            main_mention = " ".join(tokens[main_start: main_end + 1])

            for start, end in cluster[1:]:
                replacements[start] = (main_mention, end)

        ### Build New Lists Of Tokens
        new_tokens = []
        i = 0

        while i < len(tokens):
            if i in replacements:
                main_mention, end = replacements[i]
                new_tokens.append(main_mention)
                i = end + 1   # skip the whole span
            else:
                new_tokens.append(tokens[i])
                i += 1

        ### Join & Clean Tokenizations Side Effects
        resolved_text = " ".join(new_tokens)

        # Fix whitespace/tokenization issues
        resolved_text = re.sub(r"\s+", " ", resolved_text)
        resolved_text = re.sub(r"\s-", "-", resolved_text)
        resolved_text = re.sub(r"-\s", "-", resolved_text)

        return resolved_text

@CorefResolverRegistry.register("spacy")
class SpacyCorefResolver(CorefResolver):

    def __init__(self):
        super().__init__()
        self.nlp = None

    def supports_gpu(self) -> bool:
        return False

    def base_load(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.nlp.add_pipe("coreferee")

    def base_resolve(self, doc : str) -> str:

        doc = self.nlp(doc)
        replacements = {}

        for token in doc:

            if token._.coref_chains:
                resolved = doc._.coref_chains.resolve(token)

                if resolved:
                    replacements[token.i] = str(resolved[0])

        result = []

        for token in doc:

            if token.i in replacements:
                result.append(replacements[token.i] + token.whitespace_)
            else:
                result.append(token.text_with_ws)

        return "".join(result)


if __name__ == '__main__':

    example = "The light of the lava lamp deepens my experience of my environment, It transforms the room into a peaceful refuge."

    resolver = CorefResolverRegistry.create("spanbert-large", device=-1)
    print(resolver.resolve(example))

    resolver = resolver = CorefResolverRegistry.create("spacy", device=-1)
    print(resolver.resolve(example))
