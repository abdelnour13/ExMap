from abc import ABC, abstractmethod
from lingua import LanguageDetectorBuilder, Language
from typing import Optional

class LanguageDetector(ABC):

    __AVAILABLE__ = [
        "lingua"
    ]

    @abstractmethod
    def is_english(self, text : str) -> Optional[bool]:
        """
        Determines if a document is written in the english language or not. It returns `None`
        if the language can not be determined reliably.

        **Args :**
        - text : the text to be detetcted.
        - Returns : bool or None.
        """

        ...

    @classmethod
    def create(cls, name : str) -> 'LanguageDetector':

        match name:
            case "lingua":
                return LinguaLanguageDetector()
            case _:
                raise ValueError(f"Unsupported detector {name} available : {', '.join(cls.__AVAILABLE__)}.")

class LinguaLanguageDetector(LanguageDetector):

    def __init__(self):
        super().__init__()

        self.language_detector = LanguageDetectorBuilder \
            .from_languages(Language.ENGLISH, Language.FRENCH) \
            .build()

    def is_english(self, text : str) -> Optional[bool]:

        res = self.language_detector.detect_language_of(text)

        if res is not None:
            res = (res == Language.ENGLISH)

        return res