python3 text2kg.py coreference_resolution --input-filename /home/abdelnour-main/Documents/Projects/Text2KG/data/ls2n/cleaned/data.jsonl \
    --output-dir /home/abdelnour-main/Documents/Projects/Text2KG/data/ls2n/coreferences_resolved --format jsonl \
    --batch-size 1 --buffer-size 4 --model spanbert-large --device -1
