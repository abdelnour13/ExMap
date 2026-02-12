python3 text2kg.py extract_acronyms --input-filename /home/abdelnour-main/Documents/Projects/Text2KG/data/ls2n/cleaned/data.jsonl \
    --output-dir /home/abdelnour-main/Documents/Projects/Text2KG/data/ls2n/acronyms_extracted --format jsonl \
    --batch-size 1 --buffer-size 4 --qa-model timpal0l/mdeberta-v3-base-squad2 --threshold 0.5
