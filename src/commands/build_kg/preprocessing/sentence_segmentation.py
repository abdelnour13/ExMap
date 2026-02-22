from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.shared.command_tools import Stage, StageConfig, add_handlers
from pycli import Field
from typing import Dict,List
from stanza import Pipeline
from instance import cli

class Config(StageConfig):
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
class SentenceSegmenter(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        use_gpu : bool
    ) -> None:
        super().__init__(reader, writer)

        self.use_gpu = use_gpu

        self.nlp = Pipeline(lang='en', processors='tokenize', use_gpu=use_gpu)

    def tokenize(self, text : str) -> List[str]:

        document = self.nlp(text)

        return [
            sentence.text 
            for sentence in document.sentences
        ]
        
    def transform(self, doc : Dict) -> List[Dict]:

        sentences = self.tokenize(doc['abstract'])
        
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
        
        return SentenceSegmenter(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            use_gpu=config.use_gpu
        )