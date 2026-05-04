import asyncio
import multiprocessing
from typing import Dict, Iterable, Union, List, Literal, Optional, Tuple, Generator,Self
from abc import abstractmethod
from returns.result import Result
from returns.pipeline import is_successful
from src.shared.file_tools import BufferedWriter, BatchedReader, Cache
from src.shared.handlers import Event, EventLevel
from .base_command import BaseCommand
from .config import OutputDirOption, InputFileOption, ConnectionOption

class StageConfig(OutputDirOption, InputFileOption, ConnectionOption):
    pass

TransformReturn = Union[Dict, List[Dict]]

class Stage(BaseCommand):

    def __init__(self,
        reader : BatchedReader,
        writer : BufferedWriter,
        cache : Optional[Cache] = None
    ) -> None:
        super().__init__()

        self.reader = reader
        self.writer = writer
        self.cache = cache
        self.parallelism_mode : Optional[Literal['async', 'multiprocessing']] = None

        self.total_processed = 0
        self.total_saved = 0
        self.total_skipped = 0
        self.total_errors = 0

    def set_parallelism_mode(self, mode : Optional[Literal['async', 'multiprocessing']]) -> Self:
        self.parallelism_mode = mode
        return self

    def skip(self, doc : Dict) -> bool:
        """
        Determines if a document should be skipped or not
        
        :param doc: The document to check.
        :type doc: Dict
        :return: a boolean indicating if the document should be skipped or not.
        :rtype: bool
        """
        return False

    @abstractmethod
    def transform(self, doc : Dict) -> TransformReturn:
        """
        Transforms a document, yielding one or many documents.
        
        :param doc: The document to transform
        :type doc: Dict
        :return: A list containing the transformed document(s)
        :rtype: Dict | List[Dict]
        """
        ...

    def safe_transform(self, doc : Dict) -> Result[TransformReturn, Exception]:

        try:
            return Result.from_value(self.transform(doc))
        except Exception as e:
            return Result.from_failure(e)
        
    def mp_safe_transform(self, doc : Dict):

        try:
            return { "value" : self.transform(doc) }
        except Exception as e:
            return { "error" : e }
        
    async def async_safe_transform(self, doc : Dict) -> Result[TransformReturn, Exception]:

        try:
            return Result.from_value(await self.transform(doc))
        except Exception as e:
            return Result.from_failure(e)

    async def async_transform_chunk(self, chunk : Iterable[Dict]) -> List[Result[TransformReturn, Exception]]:
        return await asyncio.gather(*[self.async_safe_transform(document) for document in chunk])
    
    def mp_transform_chunk(self, chunk : Iterable[Dict]) -> List[Result[TransformReturn, Exception]]:
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            
            return [
                Result.from_failure(result['error']) if "error" in result else Result.from_failure(result['value'])
                for result in pool.map(self.mp_safe_transform, chunk)
            ]

    def make_event(self,
        progress : Optional[float],
        level : EventLevel,
        message : str,
        task : str = "main",
        extra : Optional[Dict] = None
    ) -> Event:
        
        return super().make_event(
            progress=progress, 
            level=level, 
            message=message,
            task=task,
            extra={
                'total_processed' : self.total_processed,
                'total_saved' : self.total_saved,
                'total_skipped' : self.total_skipped,
                'total_errors' : self.total_errors,
                **(extra or {})
            }
        )

    def transform_chunk(self, chunk : Iterable[Dict]) -> Tuple[List[Dict], List[Dict], List[Exception]]:
        """
        Transforms a chunk of documents, bu default it runs a self.transform on
        each document individually but can be overriden for a specific task that 
        supports parallelism (Like deep learning models).
        
        :param chunk: An Iterable of documents.
        :type chunk: Iterable[Dict]
        :return: A list containing the transformed documents
        :rtype: List[Dict]
        """

        if self.parallelism_mode == 'async':
            transformed_chunk = asyncio.run(self.async_transform_chunk(chunk))
        elif self.parallelism_mode == 'multiprocessing':
            transformed_chunk = self.mp_transform_chunk(chunk)
        else:
            transformed_chunk = [self.safe_transform(document) for document in chunk]

        results = []
        errors = []
        docs = []

        for doc, result in zip(chunk, transformed_chunk):

            if is_successful(result):

                result = result.unwrap()

                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)

                docs.append(doc)

            else:
                error = result.failure()
                errors.append(error)

        return docs, results, errors

    def __iter__(self) -> Generator[Event, None, None]:

        ### Notify Job started
        yield self.make_info_event(
            progress=0.0,
            message=f"Job {self.__class__.__name__} has started."
        )
        
        for i,(chunk,progress) in enumerate(self.reader):

            docs, records, errors = self.transform_chunk(chunk)

            if self.cache is not None:
                self.cache.update(docs)
                
            self.total_processed += len(chunk)

            yield self.make_info_event(
                progress=progress,
                message=f"Chunk {i} processing completed."
            )

            ### Filter results :
            self.total_skipped += len(records)
            records = [item for item in records if not self.skip(item)]
            self.total_skipped -= len(records)

            if len(records) > 0:

                written_documents = self.writer.write(records)
                self.total_saved += written_documents

                if written_documents > 0:

                    yield self.make_info_event(
                        progress=progress,
                        message=f"{written_documents} documents written to disk."
                    )

            ### Handle errors :
            if len(errors) > 0:

                self.total_errors += len(errors)

                for error in errors:

                    yield self.make_error_event(
                        progress=progress,
                        message=f"An error has occurred during processing chunk {i}. error : {error}"
                    )
            
        self.total_saved += self.writer.flush()

        yield self.make_info_event(
            progress=1.0,
            message=f"Job {self.__class__.__name__} has been completed."
        )