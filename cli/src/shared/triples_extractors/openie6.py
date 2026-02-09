from .base import BaseTriplesExtractor, Triple, Result, List
from xmlrpc.client import ServerProxy

class OpenIE6TriplesExtractor(BaseTriplesExtractor):

    def __init__(self, server : ServerProxy) -> None:
        super().__init__()
        self.server = server

    def extract(self, texts : List[str]) -> List[Result[List[Triple], Exception]]:

        texts = [
            (text if text.endswith('.') else text + ' .')
            for text in texts
        ]

        triples = self.server.extract_triples(texts)

        triples_groups = [[] for _ in texts]

        for triple in triples:
            idx = triple['sent_idx']
            triple = Triple(head=triple['head'],predicate=triple['predicate'],tail=triple['tail'])
            triples_groups[idx].append(triple)

        triples_groups = [Result.from_value(group) for group in triples_groups]

        return triples_groups