import time
from typing import Dict, List, Optional
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache, Reader, Cache
from SPARQLWrapper import SPARQLWrapper, JSON
from instance import cli

class EntitiesReader(Reader):

    def __init__(self, reader : Reader, cache : Optional[Cache] = None):
        super().__init__(cache)
        self.reader = reader

    def generate(self):

        p = 0

        for record, progress in self.reader:
            yield { "id" : record["head"], "name" : record["head"] }, p + (progress - p) / 2
            yield { "id" : record["tail"], "name" : record["tail"] }, progress
            p = progress


@cli.register_stage(
    name="entities_metadata_retrieval",
    description=(
        "Perform entity mapping using Wikidata aliases attributes combined with Semantic similarity using a language model."
    ),
    add_from_conf=False,
    add_save_to=False
) 
@add_handlers()
class EntitiesMetadataRetrieval(Stage):

    ENDPOINT : str = "https://query.wikidata.org/sparql"

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
    ) -> None:
        super().__init__(reader, writer)

        self.sparql = SPARQLWrapper(self.ENDPOINT, agent="KG2ExpertiseMap (abdelnour13.github.io/ab.fellah@esi-sba.dz)")

    def get_wikidata_info(self, label : str, language : str = "en") -> Dict:

        error = None
            
        for attempt in range(3):
        
            try:

                query = f"""
                    SELECT ?item ?itemLabel ?alias ?instanceOfLabel ?subclassOfLabel ?mainSubjectLabel
                    WHERE {{
                        ?item rdfs:label "{label}"@{language}.
                        OPTIONAL {{ ?item skos:altLabel ?alias FILTER (LANG(?alias) = "{language}") }}
                        OPTIONAL {{ ?item wdt:P31 ?instanceOf. }}
                        OPTIONAL {{ ?item wdt:P279 ?subclassOf. }}
                        OPTIONAL {{ ?item wdt:P921 ?mainSubject. }}
                        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language}". }}
                    }}
                """
                
                self.sparql.setQuery(query)
                self.sparql.setReturnFormat(JSON)
                results = self.sparql.query().convert()
                
                data = {
                    "id" : label,
                    "label": label, 
                    "aliases": set(), 
                    "instance of": set(), 
                    "subclass of": set(), 
                    "main subject": set(), 
                    "wikidata_id" : None
                }
                
                qid = None
                
                for r in results["results"]["bindings"]:
                    if not qid:
                        qid = r["item"]["value"].split("/")[-1]
                        data["wikidata_id"] = qid
                    if "alias" in r:
                        data["aliases"].add(r["alias"]["value"])
                    if "instanceOfLabel" in r:
                        data["instance of"].add(r["instanceOfLabel"]["value"])
                    if "subclassOfLabel" in r:
                        data["subclass of"].add(r["subclassOfLabel"]["value"])
                    if "mainSubjectLabel" in r:
                        data["main subject"].add(r["mainSubjectLabel"]["value"])
                
                # Convert sets to sorted lists
                for k in data:

                    if isinstance(data[k], set):
                        data[k] = sorted(data[k])
                
                # return Result.from_value(data)
                return data
            
            except Exception as e:
                error = e
                time.sleep(2 ** (0.8 * attempt))

        raise error
                
    def transform(self, doc : Dict) -> Dict:
        return self.get_wikidata_info(doc["name"])
        
        
    @classmethod
    def from_config(cls, config : StageConfig) -> 'EntitiesMetadataRetrieval':

        cache = TextfileCache(config.checkpoint_filename)
        # cache = None
        
        return EntitiesMetadataRetrieval(
            reader=EntitiesReader(FileReader.from_filename(config.input_filename), cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
        )