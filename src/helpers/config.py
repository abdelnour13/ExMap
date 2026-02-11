import os
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