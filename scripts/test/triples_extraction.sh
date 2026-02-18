python3 text2kg.py extract_triples --input-filename /home/abdelnour-main/Documents/Projects/Text2KG/data/ls2n/acronyms_extracted/data.jsonl \
    --output-dir /home/abdelnour-main/Documents/Projects/Text2KG/data/ls2n/raw_triples --format jsonl \
    --batch-size 1 --buffer-size 4 --type two-step --llm deepseek-chat