import asyncio
import multiprocessing
import inspect
from typing import Dict, Iterable, Union, List, Type, Literal, Optional, Tuple, Generator,Self, TypeVar, Callable
from src.shared.handlers import SocketHandler, ConsoleLogger, FileLogger
from abc import abstractmethod
from src.shared.file_tools import BufferedWriter, BatchedReader
from src.shared.handlers import Event, EventLevel
from functools import wraps
from returns.result import Result
from returns.pipeline import is_successful
from .base_command import BaseCommand, OutputDirOption, InputFileOption, ConnectionOption

class StageConfig(OutputDirOption, InputFileOption, ConnectionOption):
    pass

T = TypeVar('T')

def add_handlers(cls : T):

    from_config = cls.__dict__['from_config'].__func__    

    @wraps(from_config)
    def new_from_config(cls, config : StageConfig):

        instance : 'Stage' = from_config(cls, config)

        if config.connection is not None:
            instance.add_handler(SocketHandler(host=config.connection.host, port=config.connection.port))

        instance.add_handler(ConsoleLogger(cls.__name__))
        instance.add_handler(FileLogger(name=cls.__name__, log_file=config.log_filename))

        return instance

    cls.from_config = classmethod(new_from_config)

    return cls

TransformReturn = Union[Dict, List[Dict]]

def safe(func : Callable) -> Callable:

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def __wraper__(*args, **kwargs):

            try:
                return Result.from_value(await func(*args,**kwargs))
            except Exception as e:
                return Result.from_failure(e)
            
    else:

        @wraps(func)
        def __wraper__(*args, **kwargs):

            try:
                return Result.from_value(func(*args,**kwargs))
            except Exception as e:
                return Result.from_failure(e)
            
    return func

class Stage(BaseCommand):

    def __init__(self,
        reader : BatchedReader,
        writer : BufferedWriter,
    ) -> None:
        super().__init__()

        self.reader = reader
        self.writer = writer
        self.parallelism_mode : Optional[Literal['async', 'multiprocessing']] = None

        self.total_processed = 0
        self.total_saved = 0
        self.total_skipped = 0

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
        
    async def async_safe_transform(self, doc : Dict) -> Result[TransformReturn, Exception]:

        try:
            return Result.from_value(await self.transform(doc))
        except Exception as e:
            return Result.from_failure(e)

    async def async_transform_chunk(self, chunk : Iterable[Dict]) -> List[Result[TransformReturn, Exception]]:
        return await asyncio.gather(*[self.async_safe_transform(document) for document in chunk])
    
    def mp_transform_chunk(self, chunk : Iterable[Dict]) -> List[Result[TransformReturn, Exception]]:
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            return pool.map(self.safe_transform, chunk)

    def make_event(self,
        progress : Optional[float],
        level : EventLevel,
        message : str
    ) -> Event:
        
        super().make_event(
            progress=progress, 
            level=level, 
            message=message,
            extra={
                'total_processed' : self.total_processed,
                'total_saved' : self.total_saved,
                'total_skipped' : self.total_skipped,
            }
        )

    def transform_chunk(self, chunk : Iterable[Dict]) -> Tuple[List[Dict], List[Exception]]:
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

        for result in transformed_chunk:

            if is_successful(result):

                result = result.unwrap()

                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)

            else:
                error = result.failure()
                errors.append(error)

        return results, errors

    def __iter__(self) -> Generator[Event, None, None]:

        ### Notify Job started
        yield self.make_info_event(
            progress=0.0,
            message=f"Job {self.__class__.__name__} has started."
        )
        
        for i,(chunk,progress) in enumerate(self.reader):

            records, errors = self.transform_chunk(chunk)
            self.total_processed += len(chunk)

            yield self.make_info_event(
                progress=progress,
                message=f"Chunk {i} processing completed."
            )

            ### Filter results :
            records = [item for item in records if not self.skip(item)]

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

                self.total_skipped += len(errors)

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