import json
import ijson
import os
import lmdb
import msgpack
from typing import Dict, Generator, Tuple, Optional, List, TypeVar, Callable, Any
from abc import ABC, abstractmethod

E = TypeVar('E')

MapFn = Callable[[Dict], Any]

ReaderOutput = Tuple[Any, Optional[float]]
BatchedReaderOutput = Tuple[List[Any], Optional[float]]

def batched(
    generator : Generator[ReaderOutput, None, None],
    batch_size : int
) -> Generator[BatchedReaderOutput, None, None]:
    
    batch = []
    total_progres = 0.0

    for record, progres in generator:

        batch.append(record)

        if progres is not None:
            total_progres = progres

        if len(batch) == batch_size:
                
            yield batch, total_progres

            batch.clear()
            total_progres = 0.0


    if len(batch) > 0:
        yield batch, total_progres

def map(gen : Generator[ReaderOutput, None, None], mapper : MapFn) -> Generator[ReaderOutput, None, None]:

    for doc, progress in gen:
        yield (mapper(doc), progress)

def flatten(gen :  Generator[BatchedReaderOutput, None, None]) -> Generator[BatchedReaderOutput, None, None]:

    p = 0

    for doc, progress in gen:

        for i,sub_doc in enumerate(doc):

            yield (sub_doc, p + (progress - p) * (i+1) / len(doc))

        p = progress

class Reader(ABC):

    @abstractmethod
    def __iter__(self) -> Generator[ReaderOutput, None, None]:
        ...   


class FileReader(Reader):

    def __init__(self, filename : str) -> None:
        self.filename = filename

    @abstractmethod
    def __iter__(self) -> Generator[ReaderOutput, None, None]:
        ...

    @staticmethod
    def from_filename(filename : str) -> 'FileReader':
        
        extension = os.path.splitext(filename)[1].lower()[1:]

        if extension in ['txt', 'log']:
            return TextFileReader(filename)
        elif extension in ['json']:
            return JSONFileReader(filename)
        elif extension in ['jsonl']:
            return JSONlFileReader(filename)
        elif extension in ['lmdb']:
            return LMDBReader(filename)
        else:
            raise Exception(f"Extension {extension} is not supported")
        
class LMDBReader(FileReader):

    def __init__(self, filename : str):
        super().__init__(filename)
        self.db = lmdb.open(filename)

    def __len__(self) -> int:

        n = 0

        with self.db.begin() as trx:
            n = trx.stat()['entries']

        return n
        
    def __iter__(self) -> Generator[ReaderOutput, None, None]:

        num_records = len(self)
        
        with self.db.begin() as trx:
            for key, value in trx.cursor():
                yield msgpack.unpackb(value), 1 / num_records
                
class TextFileReader(FileReader):

    def __iter__(self) -> Generator[ReaderOutput, None, None]:

        bytes = os.path.getsize(self.filename)
        bytes_read = 0

        with open(self.filename, 'r') as f:

            for line in f:

                bytes_read += len(line)
                record = { "content" : line.strip() }
                progress = bytes_read / bytes

                yield record, progress

class JSONFileReader(FileReader):

    def __iter__(self) -> Generator[ReaderOutput, None, None]:
        
        with open(self.filename, 'r') as f:

            for item in ijson.items(f, "item"):
                yield item, None

class JSONlFileReader(FileReader):

    def __init__(self, filename : str) -> str:
        super().__init__(filename)

    def __iter__(self) -> Generator[ReaderOutput, None, None]:

        bytes = os.path.getsize(self.filename)
        bytes_read = 0
        
        with open(self.filename, "r") as f:

            for line in f:
                bytes_read += len(line)
                record = json.loads(line)
                progress = bytes_read / bytes
                yield record, progress