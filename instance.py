import inspect
from pycli import Cli as PyCli, Example, Model
from typing import List, Optional, TypeVar, Callable

T = TypeVar('T')
E = TypeVar('E')

class Cli(PyCli):

    def register_stage(self, 
        name : Optional[str] = None,
        description : Optional[str] = None,
        examples : Optional[List[Example]] = None,
        add_help : bool = True,
        add_save_to : bool = True,
        add_from_conf : bool = True,
        strict : bool = True,
    ) -> Callable[[E], E]:
        
        def __wrap__(cls : E) -> E:
            
            def _cmd(config : T) -> None:
                instance = cls.from_config(config)
                instance.run()

            config_cls = inspect \
                .signature(cls.from_config) \
                .parameters['config'].annotation
                            
            self.add_command(
                _cmd, 
                name, 
                description, 
                examples, 
                Model(config_cls, spread=False),
                add_help, 
                add_save_to, 
                add_from_conf,
                strict
            )

            return cls
        
        return __wrap__

cli = Cli(
    name="Text2KG.",
    description="A CLI tool for LLM based Knowledge Graphs generation.",
    version="1.0.0"
)