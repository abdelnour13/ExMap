import re
from bs4 import BeautifulSoup
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileReader, FileWriter
from src.shared.language_detectors import LanguageDetector
from typing import Dict
from instance import cli

# TODO: keywords and authors normalization
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
        batch_size : int, 
        buffer_size : int,
        language_detector : LanguageDetector
    ) -> None:
        super().__init__(reader, writer, batch_size, buffer_size)
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

    def clean_document(self, doc : Dict) -> Dict:
        
        if not self.language_detector.is_english(doc['title']):
            doc['title'] = None

        if not self.language_detector.is_english(doc['abstract']):
            doc['abstract'] = None
        else:
            doc['abstract'] = self.clean_text(doc['abstract'])

        doc['keywords'] = self.clean_keywords(doc['keywords'])

        return doc
    
    def valid(self, doc : Dict) -> bool:
        return (doc["title"] is not None) and \
            (doc["abstract"] is not None)

    def transform(self, doc : Dict) -> bool:
        return self.clean_document(doc)

    @classmethod
    def from_config(cls, config : StageConfig) -> Stage:

        return CorpusCleaner(
            reader=FileReader.from_filename(config.input_filename),
            writer=FileWriter.from_filename(config.output_filename),
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
            language_detector=LanguageDetector.create("lingua")
        )