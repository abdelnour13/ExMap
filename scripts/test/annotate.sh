python3 text2kg.py annotate --input-filename /home/abdelnour-main/Documents/Projects/Text2KG/data/ls2n/coreferences_resolved/data.jsonl \
    --output-dir /home/abdelnour-main/Documents/Projects/Text2KG/data/ls2n/annotated --format jsonl \
    --batch-size 1 --buffer-size 4 --model deepseek-chat --api-key sk-46e0b8e6cb7d4c989b13af3980c397c3 \
    --domains-filename /home/abdelnour-main/Documents/Projects/Text2KG/resources/domains/dsis.json