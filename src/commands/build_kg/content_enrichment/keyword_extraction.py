import torch
from typing import Dict, List
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from definitions import hf_model_dir
from instance import cli

def getkeyphrasetransformerClass():

    from keyphrasetransformer import KeyPhraseTransformer as _KeyPhraseTransformer

    class KeyPhraseTransformer(_KeyPhraseTransformer):

        def __init__(self, 
            model_type = "t5", 
            model_name = "snrspeaks/KeyPhraseTransformer",
            device : torch.types.Device = 'cpu'
        ) -> None:
            super().__init__(model_type, model_name)
            self.device = device
            self.to(device)

        def to(self, device : torch.types.Device) -> 'KeyPhraseTransformer':
            self.device = device
            self.model.to(device)
            return self

        def predict(self, doc: str):

            input_ids = self.tokenizer.encode(
                doc, return_tensors="pt", add_special_tokens=True
            ).to(self.device)

            generated_ids = self.model.generate(
                input_ids=input_ids,
                num_beams=2,
                max_length=512,
                repetition_penalty=2.5,
                length_penalty=1,
                early_stopping=True,
                top_p=0.95,
                top_k=50,
                num_return_sequences=1,
            )

            preds = [
                self.tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids.cpu()
            ]

            return preds
        
    return KeyPhraseTransformer

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
    ) -> None:
        
        super().__init__(reader, writer)

        cls = getkeyphrasetransformerClass()
        self.key_phrase_transformer = cls(model_name=hf_model_dir("snrspeaks/KeyPhraseTransformer"), device='cuda')

        # self.key_phrase_transformer.model # .to("cuda")

        # print(f"Device : {self.key_phrase_transformer.model.device}.\n")
    
    def transform(self, doc : Dict) -> List[Dict]:
        
        keywords = self.key_phrase_transformer.get_key_phrases(doc["abstract"])

        doc["keywords"].extend(keywords)

        return doc

    @classmethod
    def from_config(cls, config : StageConfig) -> Stage:

        cache = TextfileCache(checkpoint_filename=config.checkpoint_filename)
        
        return KeywordExtractor(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
        )