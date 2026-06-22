import dotenv
import asyncio
from typing import Dict, List
from src.shared.command_tools.vasync import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache
from src.shared.llms import LLM
from src.helpers import load_json_file
from src.env import env, LLM, Domain
from pydantic import computed_field
from pycli import Field
from openai import AsyncClient
from definitions import domain_dir
from instance import cli

class Config(StageConfig):

    models : List[str] = Field(
        default_factory=lambda : ["deepseek-chat"],
        description="The LLM to be used for triple extraction."
    )

    domain : str = Field(description="The json file that holds the labels and their definitions.")

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
        clients : Dict[str, LLM],
        domains : List[Domain],
    ) -> None:
        super().__init__(reader, writer)

        self.clients = clients
        self.domains = domains

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
    
    async def classify(self, doc : Dict, model : LLM) -> Dict:

        ### The prompt was modified to take into account multiple levels of
        ### membership.
        requests = []

        for domain in self.domains:

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert in scientific article classification and domain relevance assessment. "
                        "Your task is to evaluate the degree of alignment between a research article and a target scientific domain. "
                        "The article is defined by its title, abstract, and an optional list of keywords. "
                        "The domain is defined by its name and definition. "
                        "Analyze the article’s main objectives, methodology, terminology, application area, scientific context, "
                        "and primary contributions to determine how strongly the article aligns with the specified domain.\n\n"

                        "### Classification Levels:\n"
                        "* **0 (Perfect Alignment):** The article’s core problems, objectives, and contributions are exactly what is expected in this domain.\n"
                        "* **1 (Strong Alignment):** The article is highly relevant and clearly fits within this domain, although the focus may not be exclusively centered on it.\n"
                        "* **2 (Weak Alignment):** The article has limited or indirect topical overlap with the domain but is not a primary fit.\n"
                        "* **3 (No Alignment):** The article is completely unrelated to the domain.\n\n"

                        "### Important Guidelines:\n"
                        "- Use semantic understanding rather than keyword matching alone.\n"
                        "- Consider synonyms, related concepts, interdisciplinary overlap, and implicit domain relevance.\n"
                        "- Focus on the article’s primary scientific contribution rather than incidental mentions of domain-related terms.\n"
                        "- Evaluate methodological and application-level relevance in addition to terminology overlap.\n"
                        "- Be strict and consistent when distinguishing between weak and strong alignment.\n\n"

                        "### Output Format:\n"
                        "Return only one of the following numbers:\n"
                        "0\n"
                        "1\n"
                        "2\n"
                        "3\n\n"

                        "Do not provide explanations, reasoning, confidence scores, or additional text."
                    ),
                }
            ]

            prompt = self.get_prompt(domain, doc)
            messages.append({"role": "user", "content": prompt})

            req = model.async_client.chat.completions.create(
                model=model.info.name,
                messages=messages,
                stream=False,
                temperature=0,  # No randomness
                top_p=1,  # Disable sampling
                # seed=42,  # Seed
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

        return {
            "model" : model.info.name,
            "sub_domains" : sub_domains,
            "domains" : domains
        }
    
    async def classify_all(self, doc : Dict, models : Dict[str, AsyncClient]) -> List[Dict]:
        result = await asyncio.gather(*[self.classify(doc, llm) for _, llm in models.items()])
        return result

        
    async def transform(self, doc: Dict) -> Dict:

        classification_results = await self.classify_all(doc, self.clients)

        return {
            **doc, 
            "pseudo_labels" : classification_results
        }
    
    @classmethod
    def from_config(cls, config : Config) -> 'CorpusAnnotatator':

        cache = TextfileCache(config.checkpoint_filename)
        
        cmd = CorpusAnnotatator(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            clients={
                model_name : model
                for model_name, model in env.llms.items()
                if model_name in config.models
            },
            domains=env.fields[config.domain].domains,
        ).set_parallelism_mode('async')

        return cmd
    
if __name__ == '__main__':
    
    example = {
        "id": "4409451", 
        "title": "Examining the Academic Trends in Neuropsychological Tests for Executive Functions Using Virtual Reality: Systematic Literature Review", 
        "abstract": "Background In neuropsychology, fully immersive virtual reality (VR) has been spotlighted as a promising tool. It is considered that VR not only overcomes the existing limitation of neuropsychological tests but is also appropriate for treating executive functions (EFs) within activities of daily living (ADL) due to its high ecological validity. While fully immersive VR offers new possibilities of neuropsychological tests, there are few studies that overview the intellectual landscape and academic trends in the research related to mainly targeted EFs with fully immersive VR. Objective The objective of this study is to get an overview of the research trends that use VR in neuropsychological tests and to analyze the research trends using fully immersive VR neuropsychological tests with experimental articles. Methods This review was carried out according to Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) guidelines. Articles were searched in three web databases using keywords related to VR, EFs, and cognitive abilities. The study was conducted in two steps, keyword analysis and in-depth systematic review. In the web database search from 2000 to 2019, 1167 articles were initially collected, of which 234 articles in the eligibility phase were used to conduct keyword analysis and a total of 47 articles were included for systematic review. Results In keyword analysis, the number of articles focused on dementia including the keywords \u201cMCI,\u201d \u201cSCD,\u201d and \u201cdementia\u201d were highlighted over the period, rather than other symptoms. In addition, we identified that the use of behavioral and physiological data in virtual environments (VEs) has dramatically increased in recent studies. In the systematic review, we focused on the purpose of study, assessment, treatment, and validation of usability and structure. We found that treatment studies and uncategorized studies including presence and cybersickness issues have emerged in the recent period. In addition, the target symptoms and range of participants were diversified. Conclusions There has been a continuously increasing interest in dealing with neuropsychology by using fully immersive VR. Target cognitive abilities have been diversified, as well as target symptoms. Moreover, the concept of embodied cognition was transplanted in this research area.", 
        "keywords": [], 
        "authors": ["Euisung Kim", "Jieun Han", "Hojin Choi", "Yannick Pri\u00e9", "Toinon Vigier", "Samuel Bulteau", "Gyu Hyun Kwon"], 
        "venues": ["info.info-dl"], 
        "teams": ["LS2N-DUKE"], 
        "date": "2021-01-01T00:00:00Z"
    }

    annotator = CorpusAnnotatator(
        None, 
        None, 
        env.llms, 
        [Domain(**info) for info in load_json_file(domain_dir('industry_of_future'))]
    )

    result = asyncio.run(annotator.transform(example))

    from pprint import pprint

    pprint(result)
