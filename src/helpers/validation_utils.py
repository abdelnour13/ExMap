import torch
from src.helpers.torch_utils import get_all_devices
from typing import Annotated
from pydantic import BeforeValidator

def validate_device(device : int | str) -> str:

    if isinstance(device, int):

        if device < 0:
            return "cpu"
        elif device < torch.cuda.device_count():
            return f"cuda:{device}"
        else:
            cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            raise ValueError(f"device 'cuda:{device}' is not available. Available cuda devices are : {'|'.join(cuda_devices)}")

    elif isinstance(device, str):

        all_devices = get_all_devices()

        if device in all_devices:
            return device

        raise ValueError(f"device '{device}' is not available. Available devices are : {'|'.join(all_devices)}")

    raise ValueError(f"Device should be of type 'int' or 'str'. found : {type(device)}.")

Device = Annotated[str | int, BeforeValidator(validate_device)]