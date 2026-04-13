import numpy as np
import torch
import os
import json
from scipy.sparse.csgraph import connected_components
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Generator, Optional
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache, BatchedReader, BufferedWriter
from src.shared.handlers import EventLevel, Event
from src import helpers
from definitions import hf_model_dir
from pycli import Field
from instance import cli
from pydantic import computed_field

class Config(StageConfig):

    language_model : str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        description="The language model used to compute the similarity."
    )

    device : int = Field(
        default=-1, 
        description="The device used to perform the computation '-1' for cpu, '0' for gpu."
    )

    threshold : float = Field(
        default=0.95,
        description="If the similarity between two entities is equal to / higher than 'threshold' then they are counted as the same entity."
    )

    @computed_field
    @property
    def map_filename(self) -> str:
        return os.path.join(self.output_dir, f"map.json")

@cli.register_stage(
    name="entities_mapping",
    description=(
        "Perform entity mapping using Wikidata aliases attributes combined with Semantic similarity using a language model."
    ),
    add_from_conf=False,
    add_save_to=False
) 
@add_handlers()
class EntitiesMapping(Stage):

    def __init__(self, 
        reader : BatchedReader, 
        writer : BufferedWriter, 
        language_model : str,
        device : int,
        batch_size : int,
        threshold : float,
        map_filename : str
    ) -> None:
        super().__init__(reader, writer)

        self.device = "cpu" if device == -1 else "cuda"
        self.batch_size = batch_size
        self.threshold = threshold

        self.language_model = SentenceTransformer(
            hf_model_dir(language_model),
            local_files_only=True,
            device=self.device
        )

        self.entities_map = {}
        self.total_entities_mapped = 0
        self.map_filename = map_filename

    def get_representative_entity(self, group : List[str]) -> str:

        max_num_tokens = max([len(e.split(' ')) for e in group])
        candidates = [e for e in group if len(e.split(' ')) == max_num_tokens]
        single = [e for e in candidates if not e.split(' ')[-1].endswith('s')]

        if len(single) > 0:
            return single[0]
        
        return candidates[0]
    
    def before_start(self) -> Generator[Event, None, None]:

        yield self.make_info_event(
            progress=0.0,
            message="Starting mapping construction..."
        )

        yield self.make_info_event(
            progress=0.0,
            message="Computing entities embeddings..."
        )

        entities_dict = dict()
        edges = set()

        for batch,_ in self.reader:

            ### Acronyms
            for item in batch:
                    
                if 'sentence' in item and item['sentence'] is not None:

                    for acronym in item['sentence']['acronyms']:
                            
                        if acronym['score'] >= 0.8:
                            
                            if acronym['acronym'].lower() not in entities_dict:
                                entities_dict[acronym['acronym'].lower()] = [len(entities_dict), None]

                            if acronym['expansion'].lower() not in entities_dict:
                                entities_dict[acronym['expansion'].lower()] = [len(entities_dict), None]

                            edges.add((entities_dict[acronym['acronym'].lower()][0], entities_dict[acronym['expansion'].lower()][0]))

            for t in batch:

                if t['head'] not in entities_dict:
                    entities_dict[t['head']] = [len(entities_dict), None]

                if t['tail'] not in entities_dict:
                    entities_dict[t['tail']] = [len(entities_dict), None]

                if t['predicate'] == 'have abbreviation':
                    edges.add((entities_dict[t['head']][0], entities_dict[t['tail']][0]))

        ### Computing entities embeddings
        embeddings = self.language_model.encode(
            list(entities_dict.keys()),
            batch_size=self.batch_size,
            device=self.device,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        for entity, embedding in zip(entities_dict.keys(), embeddings):
            entities_dict[entity][1] = embedding

        ### Computing mapping
        yield self.make_info_event(
            progress=0.0,
            message="Adding similarity based edges..."
        )

        embeddings = torch.from_numpy(np.stack(list(map(lambda t : t[1], entities_dict.values())), axis=0))

        row, col = 0, 0
        
        for l in torch.split(embeddings, 8192):

            col = 0

            for r in torch.split(embeddings, 8192):

                S = l @ r.T
                A = (S > self.threshold).type(torch.long)

                for src, dst in torch.argwhere(A):
                    edges.add((row + src.item(), col + dst.item()))

                col += r.size(0)

            row += l.size(0)

        yield self.make_info_event(
            progress=0.0,
            message=f"Computing connected components : total edges : {len(edges)}..."
        )

        edges = torch.tensor(list(edges)).t()

        labels = helpers.torch_connected_components(edges, len(entities_dict)).numpy()

        comps, comps_counts = np.unique(labels, return_counts=True)
        comps = comps[comps_counts > 1]

        yield self.make_info_event(
            progress=0.0,
            message=f"Found {len(comps)} componenets."
        )

        entities = np.array(list(entities_dict.keys()))

        yield self.make_info_event(
            progress=0.0,
            message=f"Resolving representative entities..."
        )

        for comp in comps:

            group = entities[labels == comp]
            representaitve_entity = self.get_representative_entity(group.tolist())

            for entity in group:
                self.entities_map[entity.item()] = representaitve_entity

        yield self.make_info_event(
            progress=0.0,
            message="Entities map construction completed..."
        )

        with open(self.map_filename, 'a') as f:
            json.dump(self.entities_map, f, indent=4)

        yield self.make_info_event(
            progress=0.0,
            message="Entities map saved to disk..."
        )
                
    def transform(self, doc : Dict) -> Dict:

        if doc['head'] in self.entities_map:
            self.total_entities_mapped += 1
            doc['head'] = self.entities_map.get(doc['head'])

        if doc['tail'] in self.entities_map:
            self.total_entities_mapped += 1
            doc['tail'] = self.entities_map.get(doc['tail'])

        return doc
    
    def skip(self, doc : Dict):
        return doc['head'] == doc['tail']
    
    def make_event(self, 
        progress : float, 
        level : EventLevel, 
        message : str, 
        extra : Optional[Dict] = None
    ):
        
        return super().make_event(
            progress, 
            level, 
            message, 
            {
                'total_mapped' : self.total_entities_mapped,
                **(extra or {})
            },
        )
    
    @classmethod
    def from_config(cls, config : Config) -> 'EntitiesMapping':

        cache = TextfileCache(config.checkpoint_filename)
        
        return EntitiesMapping(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            language_model=config.language_model,
            device=config.device,
            batch_size=config.batch_size,
            threshold=config.threshold,
            map_filename=config.map_filename
        )