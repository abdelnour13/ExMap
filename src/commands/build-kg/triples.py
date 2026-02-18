import dotenv
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileReader, FileWriter
from typing import Dict, Any, List
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
        batch_size : int, 
        buffer_size : int,
    ) -> None:
        super().__init__(reader, writer, batch_size, buffer_size)

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
                    "teams" : doc['sentence']['document']['teams']
                }
            }
        })

        records.append({
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

        return Graph2Triples(
            reader=FileReader.from_filename(config.input_filename),
            writer=FileWriter.from_filename(config.output_filename),
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
        )