import importlib.util
import shutil
import subprocess
import sys
import nltk
import urllib.request
import spacy
import json
from huggingface_hub import snapshot_download
from abc import abstractmethod
from pathlib import Path
from typing import List, Literal, Sequence, get_args
from pydantic import BaseModel
from definitions import MANIFEST, PLUGINS_DIR, hf_model_dir

RESOURCE_KIND = Literal["spacy_models", "hf_models", "coreferee_resources", "nltk_resources", "neo4j_plugins"]
HF_MODEL_KIND = Literal["question-answering", "coreference-resolution", "keyword-extraction", "document-representation"]

# Folders where nltk.data.find looks when no explicit path is given in the manifest
NLTK_CATEGORIES = ("tokenizers", "taggers", "corpora", "chunkers", "models",
                   "sentiment", "stemmers", "grammars", "misc", "help")


class Resource(BaseModel):
    name: str
    required: bool = True

    @abstractmethod
    def exists(self) -> bool: ...

    @abstractmethod
    def download(self) -> None: ...

    def download_if_not_exists(self) -> bool:

        if self.exists():
            return False
        
        self.download()

        return True


class SpacyResource(Resource):

    def exists(self) -> bool:
        return spacy.util.is_package(self.name)

    def download(self) -> None:
        subprocess.run([sys.executable, "-m", "spacy", "download", self.name], check=True)
        importlib.invalidate_caches()


class NLTKResource(Resource):
    path: str | None = None        # e.g. "tokenizers/punkt"; guessed if omitted
    fallback: str | None = None    # e.g. "averaged_perceptron_tagger" for older NLTK

    @staticmethod
    def _find(name: str, path: str | None = None) -> bool:

        candidates = [path] if path else [f"{cat}/{name}" for cat in NLTK_CATEGORIES]

        for candidate in candidates:

            try:
                nltk.data.find(candidate)
                return True
            except LookupError:
                continue
        
        return False

    def exists(self) -> bool:

        return self._find(self.name, self.path) or (
            self.fallback is not None and self._find(self.fallback)
        )

    def download(self) -> None:

        try:
            nltk.download(self.name, quiet=True, raise_on_error=True)
        except Exception:

            if self.fallback is None:
                raise

            nltk.download(self.fallback, quiet=True, raise_on_error=True)


class CorefereeResource(Resource):
    
    def exists(self) -> bool:
        return importlib.util.find_spec(f"coreferee_model_{self.name}") is not None

    def download(self) -> None:
        subprocess.run([sys.executable, "-m", "coreferee", "install", self.name], check=True)
        importlib.invalidate_caches()


class Neo4JPlugin(Resource):
    url: str
    filename: str | None = None  # defaults to the last segment of the URL

    def plugin_path(self) -> Path:
        return Path(PLUGINS_DIR) / (self.filename or self.url.rsplit("/", 1)[-1])

    def exists(self) -> bool:
        return self.plugin_path().is_file()

    def download(self) -> None:
        target = self.plugin_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        try:
            with urllib.request.urlopen(self.url) as response, open(tmp, "wb") as f:
                shutil.copyfileobj(response, f)
            tmp.replace(target)  # atomic: no half-written jar if interrupted
        finally:
            tmp.unlink(missing_ok=True)


class HFModel(Resource):
    kind: HF_MODEL_KIND
    revision: str | None = None  # pin a commit/tag for reproducibility

    def model_dir(self) -> Path:
        return Path(hf_model_dir(self.name))

    def exists(self) -> bool:
        return (self.model_dir() / "config.json").exists()

    def download(self) -> None:
        snapshot_download(repo_id=self.name, local_dir=self.model_dir(), revision=self.revision)


class Manifest(BaseModel):
    hf_models: List[HFModel] = []
    spacy_models: List[SpacyResource] = []
    coreferee_resources: List[CorefereeResource] = []
    nltk_resources: List[NLTKResource] = []
    neo4j_plugins: List[Neo4JPlugin] = []

    @classmethod
    def load(cls) -> "Manifest":

        with open(MANIFEST, "r") as f:
            raw_data = json.load(f)

        return cls.model_validate(raw_data)

    def download_all(
        self,
        required_only: bool = True,
        kind: RESOURCE_KIND | Sequence[RESOURCE_KIND] | None = None,
    ) -> None:
        
        all_kinds = list(get_args(RESOURCE_KIND))

        if kind is None:
            kinds = [k for k in type(self).model_fields if k in all_kinds]
        elif isinstance(kind, str):  # str is a Sequence too, so check it first
            kinds = [kind]
        else:
            kinds = list(kind)

        unknown = set(kinds) - set(all_kinds)

        if unknown:
            raise ValueError(f"Unknown resource kind(s): {sorted(unknown)}")

        failures: list[tuple[str, str, Exception]] = []

        for resource_kind in kinds:

            resources: List[Resource] = getattr(self, resource_kind)

            for resource in resources:

                if required_only and not resource.required:
                    continue

                label = f"[{resource_kind.upper()}::{resource.name.upper()}]"

                try:
                    downloaded = resource.download_if_not_exists()
                    print(f"{label} {'downloaded' if downloaded else 'already present'}.")
                except Exception as e:
                    print(f"{label} FAILED: {e}")
                    if resource.required:
                        failures.append((resource_kind, resource.name, e))

        if failures:
            names = ", ".join(f"{k}::{n}" for k, n, _ in failures)
            raise RuntimeError(f"Failed to download required resources: {names}") from failures[0][2]


if __name__ == "__main__":
    manifest = Manifest.load()
    manifest.download_all()