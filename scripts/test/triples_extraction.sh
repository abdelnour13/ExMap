python3 text2kg.py extract_triples --input-filename /home/abdelnour-main/Documents/Projects/Text2KG/data/ls2n/sentence_segmented/data.jsonl \
    --output-dir /home/abdelnour-main/Documents/Projects/Text2KG/data/ls2n/raw_triples --format jsonl \
    --batch-size 1 --buffer-size 4 --type two-step --api-key sk-46e0b8e6cb7d4c989b13af3980c397c3 --llm deepseek-chat