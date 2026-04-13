import os
import dotenv
import pandas as pd
import numpy as np
from typing import Self
from src.shared.command_tools import BaseCommand, OutputDirOption, ConnectionOption, InputFileOption, add_handlers
from src.shared.handlers import ConsoleLogger
from src.shared.file_tools import BufferedWriter, FileWriter, FileReader, TextfileCache
from pydantic import computed_field
from definitions import ENV
from instance import cli

dotenv.load_dotenv(ENV)

class Config(InputFileOption, OutputDirOption, ConnectionOption):
    threshold : int

    @computed_field
    @property
    def cskg_dir(self) -> str:
        return dotenv.get_key(ENV, "CSKG_DIR")

@cli.register_stage(
    name="knowledge_completion",
    description="Add triples from CS-KG Knowledge Graph with a support higher than the given threshold.",
    add_from_conf=False,
    add_save_to=False
)
@add_handlers()
class KnowledgeCompletion(BaseCommand):
    
    def __init__(self, 
        reader : FileReader, 
        writer : BufferedWriter, 
        threshold : int,
        cskg_path : str
    ) -> None:
        super().__init__()
        self.reader = reader
        self.writer = writer
        self.threshold = threshold
        self.cskg_path = cskg_path
        self.entities = None

    def before_start(self):
        
        yield self.make_info_event(
            progress=0.0,
            message="Loading all entities..."
        )

        entities = set()

        for triples, _ in self.reader:
            for t in triples:
                entities.add(t['head'])
                entities.add(t['tail'])

        self.entities = np.array(['_'.join(item.split(' ')) for item in entities])

        yield self.make_info_event(
            progress=0.0,
            message=f"Found {len(entities)} entities..."
        )


    def __iter__(self):
        
        n = 32

        total_triples = 0

        mapper = {
            "skos:broader/is/hyponym-of" : "is"
        }

        for i in range(n):

            yield self.make_info_event(
                progress=i/n,
                message=f"Processing subset {i}..."
            )
            
            path = os.path.join(self.cskg_path, f"cskg_data_{i}.csv")
            subset = pd.read_csv(path)
            mask = subset['subj'].isin(self.entities) & subset['obj'].isin(self.entities) & (subset['support'] >= self.threshold)
            subset = subset[mask][['subj', 'rel', 'obj','triple_id']]

            total_triples += subset.shape[0]

            yield self.make_info_event(
                progress=i/n,
                message=f"Adding {subset.shape[0]} triples, total so far {total_triples}..."
            )

            yield self.make_info_event(
                progress=i/n,
                message=f"Writing {subset.shape[0]} triples to disk..."
            )

            d = subset.to_dict(orient='list')

            for subj, rel, obj, triple_id in zip(*d.values()):

                t = {
                    "head" : subj.replace('_',' '),
                    "predicate" : rel,
                    "tail" : obj.replace('_', ' ')
                }

                t['predicate'] = mapper.get(t['predicate'], t['predicate']).replace('-',' ')
                t['predicate_group'] = t['predicate'] # TODO: change based on the step
                t['sentence'] = None
                t['id'] = f'cskg_{i}_{triple_id}'
                t['source'] = 'cskg'

                n_saved = self.writer.write([t])

                if n_saved > 0:

                    yield self.make_info_event(
                        progress=i/n,
                        message=f"{n_saved} triples saved to disk..."
                    )

            n_saved = self.writer.flush()

            if n_saved > 0:

                yield self.make_info_event(
                    progress=i/n,
                    message=f"{n_saved} triples saved to disk..."
                )

            yield self.make_info_event(
                progress=(i+1)/n,
                message=f"Processing subset {i} completed."
            )

    def after_end(self):

        yield self.make_info_event(
            progress=1.0,
            message="Adding the rest of triples from the input file..."
        )
        
        for triples, _ in self.reader:

            for t in triples:
            
                t['source'] = 'llm'
                n_saved = self.writer.write([t])

                if n_saved > 0:

                    yield self.make_info_event(
                        progress=1.0,
                        message=f"{n_saved} triples saved to disk..."
                    )

        n_saved = self.writer.flush()

        if n_saved > 0:

            yield self.make_info_event(
                progress=1.0,
                message=f"{n_saved} triples saved to disk..."
            )

    @classmethod
    def from_config(cls, config : Config) -> Self:

        cache = TextfileCache(config.checkpoint_filename)

        downloader = KnowledgeCompletion(
            reader=FileReader
                .from_filename(config.input_filename, cache)
                .batched(config.batch_size),
            writer=FileWriter
                .from_filename(config.output_filename)
                .buffered(config.buffer_size, cache),
            threshold=config.threshold,
            cskg_path=config.cskg_dir
        )
        
        downloader \
            .add_handler(ConsoleLogger("KnowledgeCompletion"))

        return downloader