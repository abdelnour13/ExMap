import re
from bs4 import BeautifulSoup
from src.shared.handlers import ConsoleLogger, FileLogger, SocketHandler
from src.shared.command_tools import Stage, BaseConfig, Connection
from src.shared.file_tools import FileReader, FileWriter
from src.shared.language_detectors import LanguageDetector
from typing import Dict, Optional
from instance import cli

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

        doc['keywords'] = list(
            filter(
                self.language_detector.is_english,
                doc['keywords']
            )
        )

        return doc
    
    def valid(self, doc : Dict) -> bool:
        return (doc["title"] is not None) and \
            (doc["abstract"] is not None)

    def transform(self, doc : Dict) -> bool:
        return self.clean_document(doc)

# TODO: keywords and authors normalization
@cli.register_command(
    name="clean",
    description=("Performs an initial cleaning on the corpus, including : removing HTML tags and links, filtering non-english "
        "documents and keywords."
    ),
    model=BaseConfig,
    add_from_conf=False,
    add_save_to=False
)
def clean(

    ### Fields
    connection : Optional[Connection],
    input_filename : str,
    output_dir : str,
    batch_size : int,
    buffer_size : int,
    format : str,

    ### Computed Field
    output_filename : str,
    log_filename : str,
    checkpoint_filename : str

) -> None:

    cleaner = CorpusCleaner(
        reader=FileReader.from_filename(input_filename),
        writer=FileWriter.from_filename(output_filename),
        batch_size=batch_size,
        buffer_size=buffer_size,
        language_detector=LanguageDetector.create("lingua")
    )

    if connection is not None:
        cleaner.add_handler(SocketHandler(host=connection.host, port=connection.port))

    cleaner.add_handler(ConsoleLogger("CorpusCleaner"))
    cleaner.add_handler(FileLogger(name="CorpusHandler", log_file=log_filename))

    cleaner.run()