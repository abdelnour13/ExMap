import dotenv
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileReader, FileWriter, TextfileCache
from typing import Dict, List
from definitions import ENV
from instance import cli

dotenv.load_dotenv(ENV)

@cli.register_stage(
    name="graph2triples",
    description=(
        "Converts the cleaned triples into an intermediate representation to be used by later stages."
    ),
    add_from_conf=False,
    add_save_to=False
) 
@add_handlers
class Graph2Triples(Stage):

    def __init__(self, 
        reader : FileReader, 
        writer : FileWriter, 
    ) -> None:
        super().__init__(reader, writer)

        self.entities = {}
        self.documents = {}
    
    def transform(self, doc : Dict) -> List[Dict]:
        
        ### Records To Return
        records = []

        ### Get head, predicate and tail
        head, predicate, tail = doc['head'], doc['predicate'], doc['tail']

        head = head.replace(" ","_")
        predicate = predicate.replace(" ","_")
        tail = tail.replace(" ","_")

        ### [Entity]-(Predicate { value })->(Entity)
        records.append({
            "id" : f"Entity_{head}-{predicate}-Entity_{tail}",
            "n1" : {
                "type" : "Entity",
                "proprties" : {
                    "id" : f"Entity_{head}",
                    "name" : head
                }
            },
            "relation" : {
                "type" : "Predicate",
                "properties" : {
                    "value" : predicate
                }
            },
            "n2" : {
                "type" : "Entity",
                "proprties" : {
                    "id" : f"Entity_{tail}",
                    "name" : tail
                }
            }
        })

        ### [Entity]-(APPEARS-IN)->(ARTICLE)
        records.append({
            "id" : f"Entity_{head}--Article_{doc['sentence']['document']['id']}",
            "n1" : {
                "type" : "Entity",
                "proprties" : {
                    "id" : f"Entity_{head}",
                    "name" : head
                }
            },
            "relation" : {
                "type" : "APPEARS_IN",
                "properties" : {}
            },
            "n2" : {
                "type" : "Article",
                "proprties" : {
                    "id" : f"Article_{doc['sentence']['document']['id']}",
                    "title" : doc['sentence']['document']['title'],
                    "abstract" : doc['sentence']['document']['abstract'],
                    "labels" : doc['sentence']['document']['labels'],
                    "teams" : doc['sentence']['document']['teams'],
                    "authors" : doc['sentence']['document']['authors'],
                    "venues" : doc['sentence']['document']['venues'],
                    "keywords" : doc['sentence']['document']['keywords']
                }
            }
        })

        records.append({
            "id" : f"Entity_{tail}--Article_{doc['sentence']['document']['id']}",
            "n1" : {
                "type" : "Entity",
                "proprties" : {
                    "id" : f"Entity_{tail}",
                    "name" : tail
                }
            },
            "relation" : {
                "type" : "APPEARS_IN",
                "properties" : {}
            },
            "n2" : {
                "type" : "Article",
                "proprties" : {
                    "id" : f"Article_{doc['sentence']['document']['id']}",
                    "title" : doc['sentence']['document']['title'],
                    "abstract" : doc['sentence']['document']['abstract'],
                    "labels" : doc['sentence']['document']['labels'],
                    "teams" : doc['sentence']['document']['teams']
                }
            }
        })

        return records

    @classmethod
    def from_config(cls, config : StageConfig) -> 'Graph2Triples':

        cache = TextfileCache(config.checkpoint_filename)

        return Graph2Triples(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
        )