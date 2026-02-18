python3 text2kg.py annotate --input-filename /home/abdelnour-main/Documents/Projects/Text2KG/data/ls2n/coreferences_resolved/data.jsonl \
    --output-dir /home/abdelnour-main/Documents/Projects/Text2KG/data/ls2n/annotated --format jsonl \
    --batch-size 1 --buffer-size 4 --model deepseek-chat \
    --domains-filename /home/abdelnour-main/Documents/Projects/Text2KG/resources/fields/computer_science/domains.json