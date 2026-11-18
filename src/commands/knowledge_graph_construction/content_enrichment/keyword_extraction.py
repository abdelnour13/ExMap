import torch
from typing import Dict, List, Tuple
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from instance import cli
from src.methods.keywords_extractors import KeyphraseExtractorRegistry, KeyphraseExtractor

class Config(StageConfig):
    device : int = -1


@cli.register_stage(
    name="keywords_extraction",
    description=(
        "Extract additional keywords using 'KeyPhraseTransformer' T5 based keyword / topic extractor."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class KeywordsExtractor(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        extractor : KeyphraseExtractor
    ) -> None:
        super().__init__(reader, writer)

        self.keywords_extractor = extractor


    def transform_chunk(self, chunk : List[Dict]) -> Tuple[List[Dict], List[Dict], List[Exception]]:
        
        results = []

        keywords_list = self.keywords_extractor.extract([
            doc["abstract"]
            for doc in chunk
        ])

        for doc, keywords in zip(chunk, keywords_list):

            new_doc = {
                **doc,
                "keywords" : [
                    *
                    [
                        {
                            "keyword" : keyword,
                            "source"  : "api"
                        }
                        for keyword in doc['keywords']
                    ],
                    *
                    [
                        {
                            "keyword" : keyword,
                            "source"  : "keyphrase_transformer"
                        }
                        for keyword in keywords
                        if keyword not in doc['keywords']
                    ]
                ]
            }

            results.append(new_doc)

        return chunk, results, []
    
    def transform(self, doc : Dict) -> Dict:
        
        keywords = self.key_phrase_transformer.get_key_phrases(doc["abstract"])

        ### Here i updated the step to specify the source of
        ### the keyword
        doc["keywords"] = [
            *
            [
                {
                    "keyword" : keyword,
                    "source"  : "hal"
                }
                for keyword in doc['keywords']
            ],
            *
            [
                {
                    "keyword" : keyword,
                    "source"  : "keyphrase_transformer"
                }
                for keyword in keywords
                if keyword not in doc['keywords']
            ]
        ]

        return doc

    @classmethod
    def from_config(cls, config : Config) -> Stage:

        cache = TextfileCache(checkpoint_filename=config.checkpoint_filename)
        
        return KeywordsExtractor(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            extractor=KeyphraseExtractorRegistry.create("kpt:snrspeaks/KeyPhraseTransformer")
        )