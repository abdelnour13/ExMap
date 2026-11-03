from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class LLM:
    name : str
    url : str
    price_per_1m_in_tokens : float
    price_per_1m_out_tokens : float

    @staticmethod
    def from_name(name : str) -> 'LLM':

        if name not in _llms:
            raise ValueError(f"Invalid LLM {name}")
        
        return _llms[name]

_llms : Dict[str, LLM] = {
    "deepseek-chat" : LLM(
        name="deepseek-chat",
        url="https://api.deepseek.com",
        price_per_1m_in_tokens=0.28,
        price_per_1m_out_tokens=0.42
    ),
    "deepseek-v4-flash" : LLM(
        name="deepseek-v4-flash",
        url="https://api.deepseek.com",
        price_per_1m_in_tokens=0.28,
        price_per_1m_out_tokens=0.42
    ),
    "deepseek-flash" : LLM(
        name="deepseek-flash",
        url="https://api.deepseek.com",
        price_per_1m_in_tokens=0.28,
        price_per_1m_out_tokens=0.42
    ),
    "deepseek-v4-pro" : LLM(
        name="deepseek-v4-pro",
        url="https://api.deepseek.com",
        price_per_1m_in_tokens=0.28,
        price_per_1m_out_tokens=0.42
    ),
    "gemini" : LLM(
        name="gemini-3.1-flash-lite",
        url="https://generativelanguage.googleapis.com/v1beta/openai/",
        price_per_1m_in_tokens=0.28,
        price_per_1m_out_tokens=0.42
    )
}