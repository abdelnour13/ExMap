import asyncio
import multiprocessing
from typing import Generator, Dict, Iterable, Union, List, Type, Literal, Optional
from src.shared.handlers import SocketHandler, ConsoleLogger, FileLogger
from abc import abstractmethod
from src.shared.file_tools import BufferedWriter, BatchedReader
from src.shared.handlers import Event, EventLevel
from functools import wraps
from .base_command import BaseCommand, OutputDirOption, InputFileOption, ConnectionOption

class StageConfig(OutputDirOption, InputFileOption, ConnectionOption):
    pass

def add_handlers(cls : Type['Stage']):

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

    def set_parallelism_mode(self, mode : Optional[Literal['async', 'multiprocessing']]) -> None:
        self.parallelism_mode = mode

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
    def transform(self, doc : Dict) -> Union[Dict, List[Dict]]:
        """
        Transforms a document, yielding one or many documents.
        
        :param doc: The document to transform
        :type doc: Dict
        :return: A list containing the transformed document(s)
        :rtype: Dict | List[Dict]
        """
        ...

    async def async_transform_chunk(self, chunk : Iterable[Dict]) -> Union[Dict, List[Dict]]:
        return await asyncio.gather(*[asyncio.to_thread(self.transform, document) for document in chunk])
    
    def mp_transform_chunk(self, chunk : Iterable[Dict]) -> Union[Dict, List[Dict]]:
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            return pool.map(self.transform, chunk)

    def make_event(self,
        progress : Optional[float],
        level : EventLevel,
        message : str
    ) -> Event:
        
        return Event(
            progress=progress,
            level=level,
            total_processed=self.total_processed,
            total_saved=self.total_saved,
            total_skipped=self.total_skipped,
            message=message
        )
    
    def make_debug_event(self,
        progress : Optional[float],
        message : str
    ) -> Event:
        return self.make_event(progress, EventLevel.DEBUG, message)
    
    def make_info_event(self,
        progress : Optional[float],
        message : str
    ) -> Event:
        return self.make_event(progress, EventLevel.INFO, message)
    
    def make_error_event(self,
        progress : Optional[float],
        message : str
    ) -> Event:
        return self.make_event(progress, EventLevel.ERROR, message)
    
    def make_warn_event(self,
        progress : Optional[float],
        message : str
    ) -> Event:
        return self.make_event(progress, EventLevel.WARN, message)

    def transform_chunk(self, chunk : Iterable[Dict]) -> List[Dict]:
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
            chunk = asyncio.run(self.async_transform_chunk(chunk))
        elif self.parallelism_mode == 'multiprocessing':
            chunk = self.mp_transform_chunk(chunk)
        else:
            chunk = [self.transform(document) for document in chunk]

        transformed_chunk = []

        for document in chunk:

            if isinstance(document, list):
                transformed_chunk.extend(document)
            else:
                transformed_chunk.append(document)

        return transformed_chunk

    def __iter__(self) -> Generator[Event, None, None]:

        ### Notify Job started
        yield Event(
            progress=0.0,
            level=EventLevel.INFO,
            total_processed=self.total_processed,
            total_saved=self.total_saved,
            total_skipped=self.total_skipped,
            message=f"Job {self.__class__.__name__} has started."
        )
        
        for i,(chunk,progress) in enumerate(self.reader):

            error = None

            try:

                transformed_chunk = self.transform_chunk(chunk)
                self.total_processed += len(chunk)

                yield Event(
                    progress=progress,
                    level=EventLevel.INFO,
                    total_processed=self.total_processed,
                    total_saved=self.total_saved,
                    total_skipped=self.total_skipped,
                    message=f"Chunk {i} processing completed."
                )

            except Exception as e:
                self.total_skipped += 1
                error = e

            if error is None:

                transformed_chunk = [item for item in transformed_chunk if not self.skip(item)]
                written_documents = self.writer.write(transformed_chunk)
                self.total_saved += written_documents

                if written_documents > 0:

                    yield Event(
                        progress=progress,
                        level=EventLevel.INFO,
                        total_processed=self.total_processed,
                        total_saved=self.total_saved,
                        total_skipped=self.total_skipped,
                        message=f"{written_documents} documents written to disk."
                    )

            else:
 
                yield Event(
                    progress=progress,
                    level=EventLevel.ERROR,
                    total_processed=self.total_processed,
                    total_saved=self.total_saved,
                    total_skipped=self.total_skipped,
                    message=f"An error has occurred during processing chunk {i}, error : {error}"
                )

        self.total_saved += self.writer.flush()

        yield Event(
            progress=1.0,
            level=EventLevel.INFO,
            total_processed=self.total_processed,
            total_saved=self.total_saved,
            total_skipped=self.total_skipped,
            message=f"Job {self.__class__.__name__} has been completed."
        )