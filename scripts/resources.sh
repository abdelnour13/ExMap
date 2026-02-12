mkdir -p ../resources/models

hf download timpal0l/mdeberta-v3-base-squad2 --local-dir ../resources/models/timpal0l--mdeberta-v3-base-squad2
hf download SpanBERT/spanbert-large-cased --local-dir ../resources/models/SpanBERT--spanbert-large-cased # TODO: change this