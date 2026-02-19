import re
import dotenv
import os
from typing import Dict, List
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader
from src.helpers import load_json_file
from pydantic import computed_field
from pycli import Field
from dataclasses import dataclass
from openai import OpenAI
from definitions import ENV, domain_dir
from instance import cli

dotenv.load_dotenv(ENV)

@dataclass
class Domain:
    domain : str
    sub_domain : str
    definition : str

class Config(StageConfig):

    model : str = Field(
        default="deepseek-chat",
        description="The LLM to be used for triple extraction."
    )

    domain : str = Field(description="The json file that holds the labels and their definitions.")

    @computed_field
    @property
    def domains(self) -> List[Domain]:
        return [Domain(**domain) for domain in load_json_file(domain_dir(self.domain))]

    @computed_field
    @property
    def api_key(self) -> str:
        return dotenv.get_key(ENV, f"{self.model.upper()}_API_KEY")

MODELS = {
    "deepseek-chat": {
        "url": "https://api.deepseek.com",
        "price/1m_in_tokens": 0.28,
        "price/1m_out_tokens": 0.42,
    }
}

@cli.register_stage(
    name="annotate",
    description=(
        "Classify using the provided LLM each document into one of the domains defined in the json file passed."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class DocumentsAnnotator(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        batch_size : int, 
        buffer_size : int,
        client : OpenAI,
        domains : List[Domain],
        model : str
    ) -> None:
        super().__init__(reader, writer, batch_size, buffer_size)

        self.client = client
        self.domains = domains
        self.model = model

    def encode_document(self, doc: Dict) -> str:

        encoded_doc = [
            f"Title : {doc['title']}",
            f"Abstract : {doc['abstract']}",
            f"Keywords : {' | '.join(doc['keywords'])}",
        ]

        return "\n".join(encoded_doc)
        
    def get_prompt(self, domain : Domain) -> str:

        return (
            f"Given the label named \" {domain.sub_domain} \" and defined as : \" {domain.definition} \" . "
            "Does the article in question belong to this label"
        )
        
    def transform(self, doc: Dict) -> Dict:

        labels = []

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a binary scientific article classifier. "
                    "You will receive one article and multiple domain labels. "
                    "For each label, respond only with 'Yes' or 'No'. "
                    f"The article is defined as:\n{self.encode_document(doc)}"
                ),
            }
        ]

        for domain in self.domains:
            prompt = self.get_prompt(domain)
            messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            temperature=0,  # No randomness
            top_p=1,  # Disable sampling
            seed=42,  # Seed
        )

        content = response.choices[0].message.content
        answers = re.sub(r"(\n)+", "\n", content).split("\n")

        for domain, answer in zip(self.domains, answers):

            result = answer.strip().strip('"').lower()

            if result == "yes":
                labels.append(domain.sub_domain)

        return {
            **doc, 
            "labels" : labels
        }
    
    @classmethod
    def from_config(cls, config : Config) -> 'DocumentsAnnotator':
        
        return DocumentsAnnotator(
            reader=FileReader.from_filename(config.input_filename),
            writer=FileWriter.from_filename(config.output_filename),
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
            client=OpenAI(
                api_key=config.api_key,
                base_url=MODELS[config.model]['url']
            ),
            domains=config.domains,
            model=config.model
        )