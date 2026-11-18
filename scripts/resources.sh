mkdir -p ../resources/models

hf download timpal0l/mdeberta-v3-base-squad2 --local-dir ./resources/models/timpal0l--mdeberta-v3-base-squad2
hf download SpanBERT/spanbert-large-cased --local-dir ./resources/models/SpanBERT--spanbert-large-cased
hf download snrspeaks/KeyPhraseTransformer --local-dir ./resources/models/snrspeaks--KeyPhraseTransformer
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./resources/models/sentence-transformers--all-MiniLM-L6-v2
