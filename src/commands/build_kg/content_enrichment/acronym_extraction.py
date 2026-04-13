import re
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.shared.command_tools import Stage, StageConfig, add_handlers
from pydantic import Field
from typing import Dict, List
from instance import cli
from definitions import hf_model_dir

class Config(StageConfig):

    qa_model : str = Field(
        default="timpal0l/mdeberta-v3-base-squad2",
        description="The question answering model used to extract the expnasion of candidate acronyms."
    )

    threshold : float = Field(
        default=0.5,
        description="The minimum threshold to retain the acronym expansion."
    )

@cli.register_stage(
    name="acronyms_extraction",
    description=(
        "Extract acronyms from the documents relying on a method that combines regular expressions "
        "and QA models."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers()
class AcronymsExtractor(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        qa_model : str,
        threshold : float
    ) -> None:
        
        from transformers import pipeline

        super().__init__(reader, writer)

        self.qa_model = qa_model
        self.threshold = threshold
        self.pipeline = pipeline("question-answering", hf_model_dir(qa_model), device='cuda')

        self.expansion_pattern = r"\([^()]*\b\w+\b[^()]*\b\w+\b[^()]*\)"

        self.base_acronym_patterns = [
            r"[A-Z0-9-]+s?",
            r"[A-Z]+[a-z-0-9-]+[A-Z]+s?",
            r"[a-z-0-9-][A-Z]+s?"
        ]

        self.acronym_patterns = [fr"\(\s*{pattern}\s*\)" for pattern in self.base_acronym_patterns]
        self.acronym_patterns.extend([fr"{pattern}\s*{self.expansion_pattern}" for pattern in self.base_acronym_patterns])

    def is_valid_acronym(self, candidate : str) -> bool:

        candidate = candidate.strip().replace('-','')
            
        return  not candidate.isnumeric() \
            and len(candidate) > 1
    
    def get_acronym(self, candidate : str) -> str:

        if candidate.startswith('('):
            candidate = candidate.lstrip('(').rstrip(')')
        else:
            candidate = re.sub(self.expansion_pattern, "", candidate)

        return candidate.strip()

    def parse_acronyms_type1(self, sentence : str) -> List[str]:

        ### Preprocessing
        sentence = re.sub(r"\s+", " ", sentence)        

        ### Extract Acronyms
        candidates = set()
        
        for pattern in self.acronym_patterns:
            candidates.update(re.findall(pattern, sentence))

        ### Filter-out invalid acronyms
        candidates = list(
            filter(
                self.is_valid_acronym,
                map(self.get_acronym,candidates)
            )
        )

        return candidates
    
    def transform(self, doc : Dict) -> List[Dict]:
        
        sentence = doc['sentence']
        candidates = self.parse_acronyms_type1(sentence)
        acronyms = []

        for acronym in candidates:

            ### Use QA Model to get the expansion of acronym & the score
            question = f"What is the expansion of the acronym \"{acronym}\" ?"
            result = self.pipeline(question = question, context = sentence)

            ### Filter by threshold
            if result['score'] >= self.threshold:

                acronyms.append({
                    "acronym" : acronym,
                    "expansion" : result['answer'].strip(),
                    "score" : result['score']
                })

        doc["acronyms"] = acronyms

        return doc

    @classmethod
    def from_config(cls, config : Config) -> Stage:

        cache = TextfileCache(config.checkpoint_filename)
        
        return AcronymsExtractor(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            qa_model=config.qa_model,
            threshold=config.threshold
        )