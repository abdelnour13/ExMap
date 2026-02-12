import json
from typing import List, Dict, Any

def load_json_file(path : str) -> List[Dict] | Dict[str, Any]:

    with open(path, "r") as f:
        return json.load(f)