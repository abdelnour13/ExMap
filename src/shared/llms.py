from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class LLM:
    name : str
    url : str
    price_per_1m_in_tokens : float
    price_per_1m_out_tokens : float

_llms : Dict[str, LLM] = {
    "deepseek-chat" : LLM(
        name="deepseek-chat",
        url="https://api.deepseek.com",
        price_per_1m_in_tokens=0.28,
        price_per_1m_out_tokens=0.42
    )
}

def get_by_name(name : str) -> LLM:

    if name not in _llms:
        raise ValueError(f"Invalid LLM {name}")
    
    return _llms[name]