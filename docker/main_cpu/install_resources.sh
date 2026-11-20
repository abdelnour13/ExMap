#!/bin/bash
# AUTO-GENERATED from the resources manifest. Do not edit by hand.
# Regenerate with: python -m src.resources generate-script -o install_resources.sh
set -euo pipefail

# --------------------------------------------------
# hf_models
# --------------------------------------------------

echo "[HF_MODELS::timpal0l/mdeberta-v3-base-squad2]"
hf download timpal0l/mdeberta-v3-base-squad2 --local-dir /home/fellah-a/Documents/Text2KG/resources/models/timpal0l--mdeberta-v3-base-squad2

echo "[HF_MODELS::SpanBERT/spanbert-large-cased]"
hf download SpanBERT/spanbert-large-cased --local-dir /home/fellah-a/Documents/Text2KG/resources/models/SpanBERT--spanbert-large-cased

echo "[HF_MODELS::snrspeaks/KeyPhraseTransformer]"
hf download snrspeaks/KeyPhraseTransformer --local-dir /home/fellah-a/Documents/Text2KG/resources/models/snrspeaks--KeyPhraseTransformer

echo "[HF_MODELS::sentence-transformers/all-MiniLM-L6-v2]"
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir /home/fellah-a/Documents/Text2KG/resources/models/sentence-transformers--all-MiniLM-L6-v2

# --------------------------------------------------
# spacy_models
# --------------------------------------------------

echo "[SPACY_MODELS::en_core_web_lg]"
python -m spacy download en_core_web_lg

echo "[SPACY_MODELS::en_core_web_sm]"
python -m spacy download en_core_web_sm

echo "[SPACY_MODELS::en_core_web_trf]"
python -m spacy download en_core_web_trf

# --------------------------------------------------
# coreferee_resources
# --------------------------------------------------

echo "[COREFEREE_RESOURCES::en]"
python -m coreferee install en

# --------------------------------------------------
# nltk_resources
# --------------------------------------------------

echo "[NLTK_RESOURCES::stopwords]"
python -c 'import nltk; nltk.download('"'"'stopwords'"'"', quiet=True, raise_on_error=True)'

echo "[NLTK_RESOURCES::punkt_tab]"
python -c 'import nltk; nltk.download('"'"'punkt_tab'"'"', quiet=True, raise_on_error=True)'

echo "[NLTK_RESOURCES::punkt]"
python -c 'import nltk; nltk.download('"'"'punkt'"'"', quiet=True, raise_on_error=True)'

echo "[NLTK_RESOURCES::averaged_perceptron_tagger_eng]"
python -c 'import nltk; nltk.download('"'"'averaged_perceptron_tagger_eng'"'"', quiet=True, raise_on_error=True)' || python -c 'import nltk; nltk.download('"'"'averaged_perceptron_tagger'"'"', quiet=True, raise_on_error=True)'

# --------------------------------------------------
# neo4j_plugins
# --------------------------------------------------

echo "[NEO4J_PLUGINS::apoc]"
mkdir -p /home/fellah-a/Documents/Text2KG/.neo4j/plugins
curl -fsSL -o /home/fellah-a/Documents/Text2KG/.neo4j/plugins/apoc-2025.11.2-core.jar https://github.com/neo4j/apoc/releases/download/2025.11.2/apoc-2025.11.2-core.jar
