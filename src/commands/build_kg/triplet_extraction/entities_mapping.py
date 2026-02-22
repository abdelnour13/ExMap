import numpy as np
from scipy.sparse.csgraph import connected_components
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Generator
from src.shared.command_tools import Stage, StageConfig, add_handlers
from src.shared.file_tools import FileWriter, FileReader, TextfileCache, BatchedReader, BufferedWriter
from src.shared.handlers import Event
from definitions import hf_model_dir
from pycli import Field
from instance import cli

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
        default=0.9,
        description="If the similarity between two entities is equal to / higher than 'threshold' then they are counted as the same entity."
    )

@cli.register_stage(
    name="entities_mapping",
    description=(
        "Perform entity mapping using Wikidata aliases attributes combined with Semantic similarity using a language model."
    ),
    add_from_conf=False,
    add_save_to=False
) 
@add_handlers
class EntityMapper(Stage):

    def __init__(self, 
        reader : BatchedReader, 
        writer : BufferedWriter, 
        language_model : str,
        device : int,
        batch_size : int
    ) -> None:
        super().__init__(reader, writer)

        self.device = "cpu" if device == -1 else "cuda"
        self.batch_size = batch_size

        self.language_model = SentenceTransformer(
            hf_model_dir(language_model),
            local_files_only=True,
            device=self.device
        )

        self.entities_map = {}

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
            message="Computing entities embeddings..."
        )

        embeddings_dict = {}

        for batch,_ in self.reader:

            entities = [t['head'] for t in batch] + [t['tail'] for t in batch]
            entities = [e for e in entities if e not in embeddings_dict]
            entities = list(set(entities))

            embeddings = self.language_model.encode(
                entities,
                batch_size=self.batch_size,
                device=self.device,
                normalize_embeddings=True
            )

            for entity, embedding in zip(entities, embeddings):
                embeddings_dict[entity] = embedding

        yield self.make_info_event(
            progress=0.0,
            message="Constructing entities map..."
        )

        embeddings = np.stack(list(embeddings_dict.values()), axis=0)

        S = embeddings @ embeddings.T
        A = (S > 0.9).astype(np.int32)
        _, labels = connected_components(A)
        comps, comps_counts = np.unique(labels, return_counts=True)
        comps = comps[comps_counts > 1]

        entities = np.array(list(embeddings_dict.keys()))

        for comp in comps:

            group = entities[labels == comp]
            representaitve_entity = self.get_representative_entity(group.tolist())

            for entity in group:
                self.entities_map[entity.item()] = representaitve_entity

        yield self.make_info_event(
            progress=0.0,
            message="Entities map construction completed..."
        )
                
    def transform(self, doc : Dict) -> Dict:
        doc['head'] = self.entities_map.get(doc['head'], doc['head'])
        doc['tail'] = self.entities_map.get(doc['tail'], doc['tail'])
        return doc
    
    @classmethod
    def from_config(cls, config : Config) -> 'EntityMapper':

        cache = TextfileCache(config.checkpoint_filename)
        
        return EntityMapper(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            language_model=config.language_model,
            device=config.device,
            batch_size=config.batch_size
        )