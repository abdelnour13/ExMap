import json
from typing import List, Dict, Any

def load_json_file(path : str) -> List[Dict] | Dict[str, Any]:

    with open(path, "r") as f:
        return json.load(f)
    
def save_json_file(path : str, data : List[Dict] | Dict[str, Any]) -> None:

    with open(path, "a") as f:
        return json.dump(data, f, indent=4)

def override_dict(arg1 : Dict, arg2 : Dict) -> Dict:

    result = {}

    for key, value in arg1.items():

        if key in arg2:

            if not isinstance(value, dict):
                result[key] = arg2[key]
            elif not isinstance(arg2[key], dict):
                result[key] = arg2[key]
            else:
                result[key] = override_dict(value, arg2[key])

        else:

            result[key] = value

    for key, value in arg2.items():

        if key not in arg1:
            result[key] = value

    return result