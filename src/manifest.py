import importlib.util
import shlex
import shutil
import subprocess
import sys
import urllib.request
from abc import abstractmethod
from pathlib import Path
from typing import List, Literal, Sequence, get_args
from pydantic import BaseModel
from definitions import MANIFEST, PLUGINS_DIR, hf_model_dir
from src.helpers.utils import load_json_file

RESOURCE_KIND = Literal["spacy_models", "hf_models", "coreferee_resources", "nltk_resources", "neo4j_plugins"]
HF_MODEL_KIND = Literal["question-answering", "coreference-resolution", "keyword-extraction", "document-representation"]

# Folders where nltk.data.find looks when no explicit path is given in the manifest
NLTK_CATEGORIES = ("tokenizers", "taggers", "corpora", "chunkers", "models",
                   "sentiment", "stemmers", "grammars", "misc", "help")


# --------------------------------------------------
# Base resource
# --------------------------------------------------

class Resource(BaseModel):
    name: str
    required: bool = True

    @abstractmethod
    def exists(self) -> bool:
        """Return True if the resource is already installed."""

    @abstractmethod
    def download(self) -> None:
        """Download/install the resource from Python."""

    @abstractmethod
    def install_commands(self) -> list[str]:
        """Shell lines that install this resource (for the generated script)."""

    def download_if_not_exists(self) -> bool:
        if self.exists():
            return False
        self.download()
        return True


# --------------------------------------------------
# spaCy
# --------------------------------------------------

class SpacyResource(Resource):

    def exists(self) -> bool:
        import spacy
        return spacy.util.is_package(self.name)

    def download(self) -> None:
        # subprocess: spacy.cli.download may sys.exit() on failure instead of raising
        subprocess.run([sys.executable, "-m", "spacy", "download", self.name], check=True)
        importlib.invalidate_caches()

    def install_commands(self) -> list[str]:
        return [shlex.join(["python", "-m", "spacy", "download", self.name])]


# --------------------------------------------------
# NLTK
# --------------------------------------------------

class NLTKResource(Resource):
    path: str | None = None        # e.g. "tokenizers/punkt"; guessed if omitted
    fallback: str | None = None    # e.g. "averaged_perceptron_tagger" for older NLTK

    @staticmethod
    def _find(name: str, path: str | None = None) -> bool:
        import nltk
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
        import nltk
        try:
            nltk.download(self.name, quiet=True, raise_on_error=True)
        except Exception:
            if self.fallback is None:
                raise
            nltk.download(self.fallback, quiet=True, raise_on_error=True)

    def install_commands(self) -> list[str]:
        def dl(name: str) -> str:
            code = f"import nltk; nltk.download({name!r}, quiet=True, raise_on_error=True)"
            return shlex.join(["python", "-c", code])

        cmd = dl(self.name)
        if self.fallback:
            cmd = f"{cmd} || {dl(self.fallback)}"
        return [cmd]


# --------------------------------------------------
# Coreferee
# --------------------------------------------------

class CorefereeResource(Resource):
    # name is the language code, e.g. "en"

    def exists(self) -> bool:
        return importlib.util.find_spec(f"coreferee_model_{self.name}") is not None

    def download(self) -> None:
        subprocess.run([sys.executable, "-m", "coreferee", "install", self.name], check=True)
        importlib.invalidate_caches()

    def install_commands(self) -> list[str]:
        return [shlex.join(["python", "-m", "coreferee", "install", self.name])]


# --------------------------------------------------
# Neo4j plugins
# --------------------------------------------------

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

    def install_commands(self) -> list[str]:
        path = self.plugin_path()
        return [
            shlex.join(["mkdir", "-p", str(path.parent)]),
            shlex.join(["curl", "-fsSL", "-o", str(path), self.url]),
        ]


# --------------------------------------------------
# Hugging Face models
# --------------------------------------------------

class HFModel(Resource):
    kind: HF_MODEL_KIND
    revision: str | None = None  # pin a commit/tag for reproducibility

    def model_dir(self) -> Path:
        return Path(hf_model_dir(self.name))

    def exists(self) -> bool:
        return (self.model_dir() / "config.json").exists()

    def download(self) -> None:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=self.name, local_dir=self.model_dir(), revision=self.revision)

    def install_commands(self) -> list[str]:
        cmd = ["hf", "download", self.name, "--local-dir", str(self.model_dir())]
        if self.revision:
            cmd += ["--revision", self.revision]
        return [shlex.join(cmd)]


# --------------------------------------------------
# Manifest
# --------------------------------------------------

class Manifest(BaseModel):
    # Field order = install order (coreferee needs spaCy installed first)
    hf_models: List[HFModel] = []
    spacy_models: List[SpacyResource] = []
    coreferee_resources: List[CorefereeResource] = []
    nltk_resources: List[NLTKResource] = []
    neo4j_plugins: List[Neo4JPlugin] = []

    @classmethod
    def load(cls) -> "Manifest":
        return cls.model_validate(load_json_file(MANIFEST))

    def _resolve_kinds(
        self,
        kind: RESOURCE_KIND | Sequence[RESOURCE_KIND] | None,
    ) -> list[str]:
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
        return kinds

    def _selected(self, resource_kind: str, required_only: bool) -> List[Resource]:
        resources: List[Resource] = getattr(self, resource_kind)
        return [r for r in resources if not required_only or r.required]

    def download_all(
        self,
        required_only: bool = True,
        kind: RESOURCE_KIND | Sequence[RESOURCE_KIND] | None = None,
    ) -> None:
        failures: list[tuple[str, str, Exception]] = []

        for resource_kind in self._resolve_kinds(kind):
            for resource in self._selected(resource_kind, required_only):
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

    def to_install_script(
        self,
        required_only: bool = True,
        kind: RESOURCE_KIND | Sequence[RESOURCE_KIND] | None = None,
    ) -> str:
        sep = "# " + "-" * 50
        lines = [
            "#!/bin/bash",
            "# AUTO-GENERATED from the resources manifest. Do not edit by hand.",
            "# Regenerate with: python -m src.resources generate-script -o install_resources.sh",
            "set -euo pipefail",
            "",
        ]

        for resource_kind in self._resolve_kinds(kind):
            selected = self._selected(resource_kind, required_only)
            if not selected:
                continue

            lines += [sep, f"# {resource_kind}", sep, ""]
            for resource in selected:
                lines.append(f'echo "[{resource_kind.upper()}::{resource.name}]"')
                lines += resource.install_commands()
                lines.append("")

        return "\n".join(lines)


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(description="Manage NLP resources from the manifest.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    kinds = list(get_args(RESOURCE_KIND))

    dl = sub.add_parser("download", help="download resources now")
    dl.add_argument("--all", action="store_true", help="include optional resources")
    dl.add_argument("--kind", nargs="+", choices=kinds, help="only these resource kinds")

    gen = sub.add_parser("generate-script", help="write a bash install script")
    gen.add_argument("-o", "--output", default="-", help="file path, or - for stdout")
    gen.add_argument("--all", action="store_true", help="include optional resources")
    gen.add_argument("--kind", nargs="+", choices=kinds, help="only these resource kinds")

    args = parser.parse_args()
    manifest = Manifest.load()

    if args.cmd == "download":
        manifest.download_all(required_only=not args.all, kind=args.kind)
        return

    script = manifest.to_install_script(required_only=not args.all, kind=args.kind)

    if args.output == "-":
        print(script)
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(script)
        out.chmod(0o755)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()