import re
from nltk import word_tokenize
from bs4 import BeautifulSoup
from src.shared.command_tools_v2 import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileReader, FileWriter, TextfileCache
from src.shared.language_detectors import LanguageDetector
from typing import Dict
from instance import cli

@cli.register_stage(
    name="clean",
    description=(
        "Performs an initial cleaning on the corpus, including : removing HTML tags and links, filtering non-english "
        "documents and keywords."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class CorpusCleaner(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        language_detector : LanguageDetector
    ) -> None:
        super().__init__(reader, writer)
        self.language_detector = language_detector

    def clean_keywords(self, keywords : list[str]) -> list[str]:

        new_keywords = []

        for keyword in keywords:
            new_keywords.extend(keyword.split('·'))

        new_keywords = list(set(
            re.sub(r'\s+', ' ', keyword.lower().replace('-', ' ').strip(' '))
            for keyword in new_keywords
            if self.language_detector.is_english(keyword)
        ))

        return new_keywords

    def remove_html_tags(self, text : str) -> str:
        return re.sub(" +"," ",BeautifulSoup(text, 'html.parser').get_text().replace("\n"," ")).strip()
    
    def remove_latex_equations(self, text : str, placeholder : str ="[EQUATION]"):

        # Replace display math $$...$$ first
        text = re.sub(r"\$\$.*?\$\$", placeholder, text, flags=re.DOTALL)
        
        # Replace inline math $...$
        text = re.sub(r"\$.*?\$", placeholder, text)
        
        # Replace \( ... \)
        text = re.sub(r"\\\(.*?\\\)", placeholder, text)
        
        # Replace \[ ... \]
        text = re.sub(r"\\\[.*?\\\]", placeholder, text, flags=re.DOTALL)
        
        # Replace \begin{...}...\end{...}
        text = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", placeholder, text, flags=re.DOTALL)
        
        return text
    
    def remove_links(self, text : str, placeholder : str ="[URL]"):
        
        # Remove plain URLs starting with http or https
        text = re.sub(r'https?://\S+', placeholder, text)
        
        # Remove URLs starting with www.
        text = re.sub(r'www\.\S+', placeholder, text)
        
        return text
    
    def clean_text(self, text : str) -> str:

        return self.remove_links(
            self.remove_latex_equations(
                self.remove_html_tags(text)
            )
        )

    def transform(self, doc : Dict) -> Dict:
        
        if not self.language_detector.is_english(doc['title']):
            doc['title'] = None

        if not self.language_detector.is_english(doc['abstract']):
            doc['abstract'] = None
        else:

            if len(doc['abstract']) == 0 or len(word_tokenize(doc['abstract'])) <= 10:
                doc['abstract'] = None
            else:
                doc['abstract'] = self.clean_text(doc['abstract'])

        doc['keywords'] = self.clean_keywords(doc['keywords'])

        return doc
    
    def skip(self, doc : Dict) -> bool:
        return (doc["title"] is None) or \
            (doc["abstract"] is None)

    @classmethod
    def from_config(cls, config : StageConfig) -> Stage:

        cache = TextfileCache(config.checkpoint_filename)

        return CorpusCleaner(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            language_detector=LanguageDetector.create("lingua")
        )