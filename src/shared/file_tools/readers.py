import json
import ijson
import os
import lmdb
import msgpack
from typing import Generator, Tuple, Optional, List, Dict
from abc import ABC, abstractmethod
from .cache import Cache

ReaderOutput = Tuple[Dict, Optional[float]]
BatchedReaderOutput = Tuple[List[Dict], Optional[float]]

class BatchedReader:
    
    def __init__(self, 
        reader : 'Reader',
        batch_size : int
    ) -> None:
        self.reader = reader
        self.batch_size = batch_size

    def __iter__(self) -> Generator[BatchedReaderOutput, None, None]:
        
        batch = []
        total_progress = 0.0

        for record, progress in self.reader:

            batch.append(record)

            if progress is not None:
                total_progress = progress

            if len(batch) == self.batch_size:
                    
                yield batch, total_progress

                batch.clear()
                total_progress = 0.0

        if len(batch) > 0:
            yield batch, total_progress

class Reader(ABC):

    def __init__(self, cache : Optional[Cache] = None):
        super().__init__()
        self.cache = cache

    @abstractmethod
    def generate(self) -> Generator[ReaderOutput, None, None]:
        ...   

    def __iter__(self):
        
        if self.cache is not None:

            for doc, prog in self.generate():
                if not self.cache.check_exists(doc):
                    yield doc, prog

        else:
            yield from self.generate()

    def batched(self, batch_size : int) -> BatchedReader:
        return BatchedReader(self, batch_size)

class FileReader(Reader):

    def __init__(self, filename : str, cache : Optional[Cache] = None) -> None:
        super().__init__(cache)
        self.filename = filename

    @staticmethod
    def from_filename(filename : str, cache : Optional[Cache] = None) -> 'FileReader':
        
        extension = os.path.splitext(filename)[1].lower()[1:]

        if extension in ['txt', 'log']:
            return TextFileReader(filename, cache)
        elif extension in ['json']:
            return JSONFileReader(filename, cache)
        elif extension in ['jsonl']:
            return JSONlFileReader(filename, cache)
        elif extension in ['lmdb']:
            return LMDBReader(filename, cache)
        else:
            raise Exception(f"Extension {extension} is not supported")
        
class LMDBReader(FileReader):

    def __init__(self, filename : str, cache : Optional[Cache] = None):
        super().__init__(filename, cache)
        self.db = lmdb.open(filename)

    def __len__(self) -> int:

        n = 0

        with self.db.begin() as trx:
            n = trx.stat()['entries']

        return n
        
    def generate(self) -> Generator[ReaderOutput, None, None]:

        num_records = len(self)
        
        with self.db.begin() as trx:
            for key, value in trx.cursor():
                yield msgpack.unpackb(value), 1 / num_records
                
class TextFileReader(FileReader):

    def __init__(self, filename : str, cache : Optional[Cache] = None) -> None:
        super().__init__(filename, cache)
        self.bytes_read = 0.0

    def generate(self) -> Generator[ReaderOutput, None, None]:

        bytes = os.path.getsize(self.filename)

        with open(self.filename, 'r') as f:

            for line in f:

                self.bytes_read += len(line)
                record = { "content" : line.strip() }
                progress = self.bytes_read / bytes

                yield record, progress

class JSONFileReader(FileReader):

    def __init__(self, filename : str, cache : Optional[Cache] = None) -> None:
        super().__init__(filename, cache)

    def generate(self) -> Generator[ReaderOutput, None, None]:
        
        with open(self.filename, 'r') as f:

            for item in ijson.items(f, "item"):
                yield item, None

class JSONlFileReader(FileReader):

    def __init__(self, filename : str, cache : Optional[Cache] = None) -> None:
        super().__init__(filename, cache)
        self.bytes_read = 0.0

    def generate(self) -> Generator[ReaderOutput, None, None]:

        bytes = os.path.getsize(self.filename)
        
        with open(self.filename, "r") as f:

            for line in f:
                self.bytes_read += len(line)
                record = json.loads(line)
                progress = self.bytes_read / bytes
                yield record, progress