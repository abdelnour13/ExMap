import lmdb
import json
import os
import msgpack
from typing import Dict, Iterable, Optional
from abc import ABC, abstractmethod
from .cache import Cache

class BufferedWriter:

    def __init__(self, 
        writer : 'Writer', 
        buffer_size : int,
        cache : Optional[Cache] = None
    ) -> None:
        self.writer = writer
        self.buffer_size = buffer_size
        self.cache = cache
        self.buffer = []

    def write(self, docs : Iterable[Dict]) -> int:

        self.buffer.extend(docs)

        if len(self.buffer) >= self.buffer_size:

            # buff = self.buffer[:self.buffer_size]
            self.writer.write(self.buffer)
            n_saved = len(self.buffer)

            if self.cache is not None:
                self.cache.update(self.buffer)

            self.buffer = []

            self.buffer = self.buffer[self.buffer_size:]

            return n_saved

        return 0

    def flush(self) -> int:

        if len(self.buffer) > 0:

            self.writer.write(self.buffer)

            if self.cache is not None:
                self.cache.update(self.buffer)
        
        return len(self.buffer)

class Writer(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def write(self, docs : Iterable[Dict]) -> None:
        ...

    def buffered(self, buffer_size : int, cache : Optional[Cache] = None) -> BufferedWriter:
        return BufferedWriter(self, buffer_size, cache)

class FileWriter(Writer):

    def __init__(self, filename : str):
        self.filename = filename

    @staticmethod
    def from_filename(filename : str) -> 'FileWriter':
        
        extension = os.path.splitext(filename)[1].lower()[1:]

        if extension == "lmdb":
            return LMDBWriter(filename)
        elif extension == "jsonl":
            return JSONlWriter(filename)
        else:
            raise ValueError(f"Unsupported extention {extension}.")
        
class JSONlWriter(FileWriter):

    def write(self, chunk : Iterable[Dict]):

        dump = ""
        
        for doc in chunk:
            dump += f"{json.dumps(doc)}\n"

        with open(self.filename, 'a') as f:
            f.write(dump)

class LMDBWriter(FileWriter):

    def __init__(self, filename : str) -> None:
        super().__init__(filename)
        self.db = lmdb.open(filename, map_size = 1 << 30)

    def _get_num_records(self) -> int:

        n = None
        
        with self.db.begin() as trx:
            n = trx.stat()['entries']

        return n

    def write(self, chunk : Iterable[Dict]) -> None:

        n = self._get_num_records()
        
        with self.db.begin(write=True) as trx:
            for i,doc in enumerate(chunk):
                trx.put(str(n + i).encode("ascii"), msgpack.packb(doc))