from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.handlers import Event, EventLevel
from src.methods.sentence_segmentors import SentenceSegmentorRegistry, SentenceSegmentor
from pycli import Field
from typing import Dict,List
from instance import cli

class Config(StageConfig):
    segmentor : str = Field(default="stanza")
    use_gpu : bool = Field(default=False,description="whether to use GPU acceleration or not, only available for strategy=stanza")

@cli.register_stage(
    name="sentence_segmentation",
    description=(
        "applies sentence segmentation as a processing step to the abstracts."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class SentenceSegmentation(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        segmentor : SentenceSegmentor
    ) -> None:
        super().__init__(reader, writer)
        self.segmentor = segmentor

    def before_start(self):
            
        yield Event(
            progress=0.0,
            level=EventLevel.INFO,
            extra={
                'total_processed' : 0,
                'total_saved' : 0,
                'total_skipped' : 0
            },
            message=f"Loading {self.segmentor.__class__.__name__}..."
        )

        self.segmentor.load()

        yield Event(
            progress=0.0,
            level=EventLevel.INFO,
            extra={
                'total_processed' : 0,
                'total_saved' : 0,
                'total_skipped' : 0
            },
            message=f"{self.segmentor.__class__.__name__} Loaded."
        )
        
    def transform(self, doc : Dict) -> List[Dict]:

        sentences = self.segmentor.segment(doc['abstract'])
        
        return [
            {
                "id" : f"{doc['id']}_{i}",
                "sentence" : sentence.strip(),
                "document" : doc
            }
            for i,sentence in enumerate(sentences)
        ]
    
    def skip(self, doc : Dict) -> bool:
        return len(doc['sentence']) == 0

    @classmethod
    def from_config(cls, config : Config) -> Stage:

        cache = TextfileCache(config.checkpoint_filename)
        
        return SentenceSegmentation(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            segmentor=SentenceSegmentorRegistry.create(
                config.segmentor, 
                use_gpu=config.use_gpu
            )
        )