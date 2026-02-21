from keyphrasetransformer import KeyPhraseTransformer
from typing import Dict, List
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader
from definitions import hf_model_dir
from instance import cli

@cli.register_stage(
    name="extract_keywords",
    description=(
        "Extract additional keywords using 'KeyPhraseTransformer' T5 based keyword / topic extractor."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class KeywordExtractor(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        batch_size : int, 
    ) -> None:
        super().__init__(reader, writer, batch_size)
        self.key_phrase_transformer = KeyPhraseTransformer(model_name=hf_model_dir("snrspeaks/KeyPhraseTransformer"))
    
    def transform(self, doc : Dict) -> List[Dict]:
        
        keywords = self.key_phrase_transformer.get_key_phrases(doc["abstract"])

        doc["keywords"].extend(keywords)

        return doc

    @classmethod
    def from_config(cls, config : StageConfig) -> Stage:
        
        return KeywordExtractor(
            reader=FileReader.from_filename(config.input_filename),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size),
            batch_size=config.batch_size,
        )