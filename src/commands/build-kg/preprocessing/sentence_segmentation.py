from src.shared.file_tools import FileWriter, FileReader
from src.shared.command_tools import Stage, BaseConfig, command
from pycli import Field, Model
from typing import Dict,List
from stanza import Pipeline
from instance import cli

class Config(BaseConfig):
    use_gpu : bool = Field(default=False,description="whether to use GPU acceleration or not, only available for strategy=stanza")

class SentenceSegmenter(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        batch_size : int, 
        buffer_size : int,
        use_gpu : bool
    ) -> None:
        super().__init__(reader, writer, batch_size, buffer_size)

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
                "document_id" : doc['id'],
                "sentence" : sentence.strip()
            }
            for i,sentence in enumerate(sentences)
        ]
    
    def skip(self, doc : Dict) -> bool:
        return len(doc['sentence']) == 0
    
@cli.register_command(
    name="sentence_segmentation",
    description=(
        "applies sentence segmentation as a processing step to the abstracts."
    ),
    model=Model(Config, spread=False),
    add_from_conf=False,
    add_save_to=False
)
@command
def segment_sentences(config : Config) -> Stage:
    
    return SentenceSegmenter(
        reader=FileReader.from_filename(config.input_filename),
        writer=FileWriter.from_filename(config.output_filename),
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        use_gpu=config.use_gpu
    )