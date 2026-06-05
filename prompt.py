TOPIC_EXTRACTION_PROMPT = """
You are an expert in knowledge graph analysis and topic modeling.

Your task is to transform a noisy cluster of entities into a hierarchical set of fine-grained research topics.

────────────────────────────────────────
INPUT
────────────────────────────────────────
You will receive a list of entities extracted from a knowledge graph cluster.

Entities:
{entity_list}

────────────────────────────────────────
OBJECTIVE
────────────────────────────────────────
1. Identify latent research topics represented in the cluster.
2. Group entities into fine-grained, semantically coherent topics.
3. Produce a clean hierarchical taxonomy of topics.

────────────────────────────────────────
CRITICAL FILTERING RULES
────────────────────────────────────────
Do NOT treat the following as topics:

- Generic hub terms:
  model, method, system, framework, approach,
  performance, accuracy, efficiency, effectiveness,
  technique, algorithm (when used generically),
  data, information, process (when generic)

If such nodes appear:
- Treat them as noise or weak connectors
- Do NOT use them to define topics

────────────────────────────────────────
TOPIC FORMATION RULES
────────────────────────────────────────
A valid topic MUST:
- Represent a known research area or application domain
- Contain multiple semantically related entities
- Be more specific than the root domain
- Be suitable as a survey paper title

────────────────────────────────────────
MERGING RULES
────────────────────────────────────────
- Merge synonyms (e.g., "IoT system" = "Internet of Things system")
- Merge lexical variants (e.g., "graph neural network" variants)
- Prefer canonical research terms (e.g., "Graph Neural Networks")

────────────────────────────────────────
HUBLESS PRINCIPLE
────────────────────────────────────────
Do NOT allow generic high-frequency nodes to influence clustering.

If ambiguous:
- prefer specific technical concepts
- discard groupings driven only by shared generic words

────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────
Return ONLY:

1. A hierarchical topic tree
2. Entity assignments per topic

Format:

ROOT TOPIC: <name>

- Topic 1
  - Subtopic (optional)
    - entities: [...]

- Topic 2
  - entities: [...]

────────────────────────────────────────
QUALITY CONSTRAINT
────────────────────────────────────────
Each topic must be coherent enough to represent a research paper or survey area.

Return only the final structured output.
"""