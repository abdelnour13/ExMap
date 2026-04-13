import os
from typing import Dict, TypeVar
from pydantic_core import PydanticCustomError

def assert_file_exists(path : str) -> str:

    if not os.path.exists(path):

        raise PydanticCustomError(
            'file_not_exists',
            'File {path} is not a valid file or directory.',
            { "path" : path }
        )
    
    return path


def makedirs(path : str) -> str:

    if not os.path.exists(path):
        os.makedirs(path)

    return path

def override(dict1 : Dict, dict2 : Dict) -> Dict:

    dict1 = dict1.copy()

    for key in dict2:
        dict1[key] = dict2[key]

    return dict1