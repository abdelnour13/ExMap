from src.helpers.config import assert_file_exists
from typing import Generator, Dict, Iterable, Union, List, Annotated, Type
from src.shared.handlers import SocketHandler, ConsoleLogger, FileLogger
from abc import abstractmethod
from src.shared.file_tools import Reader, FileWriter, batched
from src.shared.handlers import Event, EventLevel
from pydantic import Field, BeforeValidator
from functools import wraps
from abc import ABC
from .base_command import BaseCommand, BaseConfig

class StageConfig(BaseConfig):

    input_filename : Annotated[str, BeforeValidator(assert_file_exists)] = Field(
        description="The path to the file holding the data."
    )
    
    batch_size : int = Field(default=1,description="How many documents to process on a single pass.")
    buffer_size : int = Field(default=4,description="The size of the buffer used to update the file on the disk.")

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
        reader : Reader,
        writer : FileWriter,
        batch_size : int,
        buffer_size : int
    ) -> None:
        super().__init__()

        self.reader = reader
        self.writer = writer
        self.batch_size = batch_size
        self.buffer_size = buffer_size

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

        transformed_chunk = []

        for document in chunk:

            document = self.transform(document)

            if isinstance(document, list):
                transformed_chunk.extend(document)
            else:
                transformed_chunk.append(document)

        return transformed_chunk

    def __iter__(self) -> Generator[Event, None, None]:

        ### Buffer
        buffer = []

        ### Stats
        total_processed = 0
        total_saved = 0
        total_skipped = 0

        ### Notify Job started
        yield Event(
            progress=0.0,
            level=EventLevel.INFO,
            total_processed=total_processed,
            total_saved=total_saved,
            total_skipped=total_skipped,
            message=f"Job {self.__class__.__name__} has started."
        )
        
        for i,(chunk,progress) in enumerate(batched(self.reader, self.batch_size)):

            error = None

            try:

                transformed_chunk = self.transform_chunk(chunk)
                total_processed += len(chunk)

                yield Event(
                    progress=progress,
                    level=EventLevel.INFO,
                    total_processed=total_processed,
                    total_saved=total_saved,
                    total_skipped=total_skipped,
                    message=f"Chunk {i} processing completed."
                )

            except Exception as e:
                total_skipped += 1
                error = e

            if error is None:

                for item in transformed_chunk:

                    if not self.skip(item):
                        buffer.append(item)

                    if len(buffer) >= self.buffer_size:

                        total_saved += len(buffer)
                        self.writer.write_chunk(buffer)
                        buffer.clear()

                        yield Event(
                            progress=progress,
                            level=EventLevel.INFO,
                            total_processed=total_processed,
                            total_saved=total_saved,
                            total_skipped=total_skipped,
                            message="Results buffer saved to disk."
                        )
            else:
 
                yield Event(
                    progress=progress,
                    level=EventLevel.ERROR,
                    total_processed=total_processed,
                    total_saved=total_saved,
                    total_skipped=total_skipped,
                    message=f"An error has occured during processing chunk {i}, error : {error}"
                )

        if len(buffer) > 0:
            total_saved += len(buffer)
            self.writer.write_chunk(buffer)
            buffer.clear()

        yield Event(
            progress=1.0,
            level=EventLevel.INFO,
            total_processed=total_processed,
            total_saved=total_saved,
            total_skipped=total_skipped,
            message=f"Job {self.__class__.__name__} has been completed."
        )