import asyncio
import multiprocessing
from typing import Dict, Iterable, Union, List, Literal, Optional, Self, TypeVar, Tuple
from src.shared.handlers import SocketHandler, ConsoleLogger, FileLogger
from abc import abstractmethod
from src.shared.file_tools import BufferedWriter, BatchedReader
from src.shared.handlers import EventLevel
from functools import wraps
from returns.result import Result
from returns.pipeline import is_successful
from .base_command import BaseCommand
from .config import OutputDirOption, InputFileOption, ConnectionOption

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
SafeTransformReturn = Result[TransformReturn, Exception]

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
        self.total_errors = 0

        self.chunk : List[Tuple[Dict, float | None]] = []
        self.max_concurent_tasks = reader.batch_size # child class can change this to depend on other factors
        self.results_queue = asyncio.Queue[Tuple[Dict, SafeTransformReturn, float | None]](0)
        self.lock = asyncio.Semaphore(self.max_concurent_tasks)

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

    def safe_transform(self, doc : Dict) -> SafeTransformReturn:

        try:
            return Result.from_value(self.transform(doc))
        except Exception as e:
            return Result.from_failure(e)
        
    async def async_safe_transform(self, doc : Dict) -> Result[TransformReturn, Exception]:

        try:
            return Result.from_value(await self.transform(doc)) # Should be async when overriden by child class
        except Exception as e:
            return Result.from_failure(e)

    def transform_chunk(self, chunk : List[Dict]) -> List[SafeTransformReturn]:
        """
        Transforms a chunk of documents, bu default it runs a self.transform on
        each document individually but can be overriden for a specific task that 
        supports parallelism (Like deep learning models).
        
        :param chunk: An Iterable of documents.
        :type chunk: Iterable[Dict]
        :return: A list containing the transformed documents
        :rtype: List[Dict]
        """

        return [self.safe_transform(document) for document in chunk]
    
    def mp_transform_chunk(self, chunk : Iterable[Dict]) -> List[SafeTransformReturn]:
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            return pool.map(self.safe_transform, chunk)
        
    async def async_transform_task(self, doc : Dict, progress : float | None) -> None:
        result = await self.async_safe_transform(doc)
        self.lock.release()
        await self.results_queue.put((doc, result, progress))

    async def add_to_queue(self, 
        doc : Dict, 
        progress : float | None,
    ) -> None:

        if self.parallelism_mode is None or self.parallelism_mode == 'multiprocessing':

            if len(self.chunk) < self.max_concurent_tasks:
                self.chunk.append((doc, progress))

            if len(self.chunk) == self.max_concurent_tasks:
                
                ### Custom or sequential (default)
                if self.parallelism_mode is None:
                    transformed = self.transform_chunk([doc for doc, _ in self.chunk])

                ### Using multiprocessing
                else:
                    transformed = self.mp_transform_chunk(self.chunk)

                for result, (doc, progress) in zip(transformed, self.chunk):
                    await self.results_queue.put((doc, result, progress))

                self.chunk.clear()

        ### asyncio
        else:
            # Seperate lock and release because i don't want to load more documents and create tasks until a slot is available.
            await self.lock.acquire()
            asyncio.create_task(self.async_transform_task(doc, progress))

    async def process_remaining(self) -> None:

        if len(self.chunk) > 0:

            if self.parallelism_mode is None or self.parallelism_mode == 'multiprocessing':
                    
                if self.parallelism_mode is None:
                    transformed = self.transform_chunk([doc for doc, _ in self.chunk])
                else:
                    transformed = self.mp_transform_chunk(self.chunk)

                for result, (doc, progress) in zip(transformed, self.chunk):
                    await self.results_queue.put((doc, result, progress))

                self.chunk.clear()

            ### asyncio
            else:
                for doc, progress in self.chunk:
                    await self.lock.acquire()
                    asyncio.create_task(self.async_transform_task(doc, progress))

    async def result_consumer(self) -> None:

        while True:

            doc, result, progress = await self.results_queue.get()
            self.total_processed += 1

            if is_successful(result):

                value = result.unwrap()

                # don't worry about this for now
                if self.skip(value):
                    self.total_skipped += 1
                else:
                    # TODO: diffrentiate between write many and write single
                    written_documents = self.writer.write(value if isinstance(value, list) else [value]) 
                    self.total_saved += written_documents

                    await self.emit_info_event(
                        progress=progress,
                        message=f"Document {doc['id']} processing completed."
                    )

                    if written_documents > 0:
                        
                        await self.emit_info_event(
                            progress=progress,
                            message=f"{written_documents} documents written to disk."
                        )

            else:

                error = result.failure()

                self.total_errors += 1

                await self.emit_error_event(
                    progress=progress,
                    message=f"An error has occurred during processing document {doc['id']}. error : {error}"
                )

            self.results_queue.task_done()
            
    async def generate(self) -> bool:

        consume_task = asyncio.create_task(self.result_consumer())

        try:

            await self.emit_info_event(
                progress=0.0,
                message=f"Job {self.__class__.__name__} has started."
            )
            
            for chunk, progress in self.reader:
                for doc in chunk:
                    await self.add_to_queue(doc, progress)

        except Exception as e:

            await self.emit_warn_event(
                progress=0.0,
                message=f"Shutting down due to an unexpected error waiting for remaining tasks to complete. error : {e}."
            )

        finally:

            # await self.process_remaining()

            await self.results_queue.join()
            consume_task.cancel()

            self.total_saved += self.writer.flush()

            await self.emit_info_event(
                progress=1.0,
                message=f"Job {self.__class__.__name__} has been completed."
            )

        return True
        
    async def emit_event(self,
        progress : Optional[float],
        level : EventLevel,
        message : str,
        extra : Optional[Dict] = None
    ) -> None:
        
        await super().emit_event(
            progress=progress, 
            level=level, 
            message=message,
            extra={
                'total_processed' : self.total_processed,
                'total_saved' : self.total_saved,
                'total_skipped' : self.total_skipped,
                'total_errors' : self.total_errors,
                **(extra or {})
            }
        )