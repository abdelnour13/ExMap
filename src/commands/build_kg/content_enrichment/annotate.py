import re
import dotenv
import asyncio
from typing import Dict, List
from src.shared.command_tools.vasync import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.shared.llms import LLM
from src.helpers import load_json_file
from pydantic import computed_field
from pycli import Field
from dataclasses import dataclass
from openai import AsyncClient
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
        default_factory=lambda : ["deepseek-chat"],
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
    name="corpus_annotation",
    description=(
        "Classify using the provided LLM each document into one of the domains defined in the json file passed."
    ),
    add_from_conf=False,
    add_save_to=False
)
@add_handlers
class CorpusAnnotatator(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
        client : AsyncClient,
        domains : List[Domain],
        model : str
    ) -> None:
        super().__init__(reader, writer)

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
        
    def get_prompt(self, domain : Domain, article : Dict) -> str:

        return (
            "### Article : \n"
            f"- Title : {article['title']}.\n"
            f"- Abstract : {article['abstract']}.\n"
            f"- Keywords : {' | '.join(article['keywords'])}.\n\n"

            "### Domain : \n"
            f"- Name : {domain.sub_domain}\n"
            f"- Definition : {domain.definition}"
        )
        
    async def transform(self, doc: Dict) -> Dict:

        ### The prompt was modified to take into account multiple levels of
        ### membership.

        requests = []

        for domain in self.domains:

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert in scientific article classification. Your task is to evaluate the alignment "
                        "between a provided research article defined by its title, abstract and an optional list of keywords and "
                        "and a domain defined by a name and a definition and your role is to assign a numerical score (0-3) "
                        "based on how well the article's problems and themes align with the domain : \n\n"
                        
                        "### Scoring System:\n"
                        "* **0 (Perfect Alignment):** The article’s core problems are exactly what is expected in this domain.\n"
                        "* **1 (Strong Alignment):** The article is highly relevant and fits well within this domain.\n"
                        "* **2 (Weak Alignment):** The article has minor topical overlap but is not a primary fit.\n"
                        "* **3 (No Alignment):** The article is completely unrelated to this domain.\n\n"
                        
                        "### Output Format:\n"
                        "The response should only contain the number that coresponds to the level with no additional explanation."
                    ),
                }
            ]

            prompt = self.get_prompt(domain, doc)
            messages.append({"role": "user", "content": prompt})

            req = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                temperature=0,  # No randomness
                top_p=1,  # Disable sampling
                seed=42,  # Seed
            )

            requests.append(req)

        responses = await asyncio.gather(*requests)
        responses = [res.choices[0].message.content for res in responses]

        sub_domains = {}
        domains = {}

        for domain, answer in zip(self.domains, responses):
            result = int(answer.strip('"\'* ').lower())
            sub_domains[domain.sub_domain] = result
            domains[domain.domain] = min(domains.get(domain.domain, result), result)

        ### The key to domains as well as the structure 
        ### (from list ==> dict) was modified
        return {
            **doc, 
            "sub_domains" : sub_domains,
            "domains" : domains
        }
    
    @classmethod
    def from_config(cls, config : Config) -> 'CorpusAnnotatator':

        cache = TextfileCache(config.checkpoint_filename)

        llm = LLM.from_name(config.model)
        
        cmd = CorpusAnnotatator(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            client=AsyncClient(
                api_key=config.api_key,
                base_url=llm.url
            ),
            domains=config.domains,
            model=config.model
        ).set_parallelism_mode('async')

        return cmd