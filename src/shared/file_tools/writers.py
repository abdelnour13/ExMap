import lmdb
import json
import os
import msgpack
from typing import Dict, Iterable

class FileWriter:

    def __init__(self, filename : str):
        self.filename = filename

    def write(self, doc : Dict) -> None:
        raise NotImplementedError
    
    def write_chunk(self, chunk : Iterable[Dict]) -> None:

        for doc in chunk:
            self.write(doc)

    @staticmethod
    def from_filename(filename : str) -> 'FileWriter':
        
        extension = os.path.splitext(filename)[1].lower()[1:]

        if extension == "lmdb":
            return LMDBWriter(filename)
        elif extension == "jsonl":
            return JSONlWriter(filename)
        else:
            raise ValueError(f"Unsupported extention {extension}")
        
class JSONlWriter(FileWriter):

    def write(self, doc : Dict) -> None:
        
        with open(self.filename, 'a') as f:
            f.write(json.dumps(doc) + "\n")

    def write_chunk(self, chunk : Iterable[Dict]):

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

    def write(self, doc : Dict) -> None:

        n = self._get_num_records()
        
        with self.db.begin(write=True) as trx:
            trx.put(str(n).encode("ascii"), msgpack.packb(doc))

    def write_chunk(self, chunk : Iterable[Dict]) -> None:

        n = self._get_num_records()
        
        with self.db.begin(write=True) as trx:
            for i,doc in enumerate(chunk):
                trx.put(str(n + i).encode("ascii"), msgpack.packb(doc))