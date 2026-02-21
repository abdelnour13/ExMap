import os
from abc import ABC, abstractmethod
from typing import Dict, Iterable

class Cache(ABC):

    @abstractmethod
    def check_exists(self, doc : Dict) -> bool:
        ...

    @abstractmethod
    def update(self, doc : Iterable[Dict]) -> None:
        ...

class TextfileCache(Cache):

    def __init__(self, checkpoint_filename : str):
        super().__init__()
        self.checkpoint_filename = os.path.join(checkpoint_filename, 'store.txt')

        if os.path.exists(self.checkpoint_filename):

            with open(self.checkpoint_filename, 'r') as f:
                self.ids = set(f.read().split('\n'))

        else:
            self.ids = set()

    def check_exists(self, doc : Dict) -> bool:
        return str(doc['id']) in self.ids
    
    def update(self, docs : Iterable[Dict]) -> None:

        ids = [id for id in map(lambda doc : str(doc['id']), docs) if id not in self.ids]

        if len(ids) > 0:
            
            self.ids.update(ids)

            dump = ''

            if os.path.exists(self.checkpoint_filename):
                dump = '\n'
            else:
                os.makedirs(os.path.dirname(self.checkpoint_filename))
            
            dump += '\n'.join(ids)
            
            with open(self.checkpoint_filename, 'a') as f:
                f.write(dump)