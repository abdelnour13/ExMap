import re
import warnings
from typing import TypeVar, Type, Callable, List, Any, Dict, Protocol, Tuple, Optional
from string import Formatter

T = TypeVar("T")
E = TypeVar("E")


class Registry(Protocol[T]):

    @classmethod
    def available(cls) -> List[str]:
        ...

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> T:
        ...

    @classmethod
    def register(
        cls,
        name: str
    ) -> Callable[[Type[E]], Type[E]]:
        ...

def template_to_regex(template: str) -> Tuple[re.Pattern, List[str]]:

    pattern = ""
    keys = []

    for literal, field, _, _ in Formatter().parse(template):
        pattern += re.escape(literal)

        if field is not None:
            pattern += r"(.+?)"
            keys.append(field)

    return re.compile(pattern), keys

def parse(template : str, text : str) -> Optional[Dict[str, Any]]:

    regex, keys = template_to_regex(template)

    match = regex.fullmatch(text)

    result = None

    if match:
        result = dict(zip(keys, match.groups()))

    return result


def registry(cls : Type[T]) -> Registry[T]:

    __registry__: Dict[str, Type[T]] = {}

    class _Return(cls):

        __id__ = id

        @classmethod
        def available(cls) -> List[str]:
            return list(__registry__.keys())

        @classmethod
        def create(cls, name: str, **kwargs: Any) -> T:

            _cls = None
            output = None

            for _cls_name, _cls in __registry__.items():

                output = parse(template=_cls_name, text=name)

                if output is not None:
                    break

            if output is None:

                raise ValueError(
                    f"class under name: '{name}' didn't match any of {__registry__.keys()}."
                )

            has_update_fn = hasattr(_cls, 'update_kwargs') and callable(getattr(_cls, 'update_kwargs'))

            if has_update_fn:
                update_fn = getattr(_cls, 'update_kwargs')
                kwargs = update_fn(kwargs, output)
            else:

                if len(kwargs) > 0:
                    warnings.warn(f"Arguments {output} are escaped.")
            
            # Get constructor arguments
            try:
                args = _cls.__init__.__annotations__.copy()
            except AttributeError:
                args = {}

            args.pop("return", None)

            retained_args = {}

            for k, v in kwargs.items():

                if k in args:
                    retained_args[k] = v
                else:
                    warnings.warn(
                        f"Unused argument {k} for class {name}."
                    )

            return _cls(**retained_args)

        @classmethod
        def register(
            cls,
            name: str
        ) -> Callable[[Type[T]], Type[T]]:

            def inner(_cls: Type[T]) -> Type[T]:

                if name in __registry__:
                    raise ValueError(
                        f"Class under name: '{name}' is already registered."
                    )

                __registry__[name] = _cls

                return _cls

            return inner

    return _Return