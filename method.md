# Goal : Guided Expertise Map construction by predefined set of domains of interest.

# 1- Knowledge Graph Construction :

    - Corpus cleaning : handling html tags, links, latex equations...etc.
    - Co-reference resolution.
    - Sentence segmentation.
    - LLM-based triples extraction module : extract entities => extract triples that link those entities.
    - Entities cleaning : remove articles , singualrize entities for example graph neural networks => graph neural network.
    - Predicates cleaning : pattern-based rules to extract the main verb.
    - Entities mapping : two entities are equivelent if : one is an acronym (therefore the other is an expansion) of the other,
        or there embeddings similarity exceeds a certain threshold.

# 2- LLM labeling :

Use an LLM as a multi-label binary classifier to classify articles into a set of pre-defined domains.

# 3- Feature Extraction :

    - Create a graph with two nodes :
        - entity.
        - article.
    - Relations :
        - entity<->entity kg links.
        - article<->entity appearance links.
        - article<->article links (share-author)

    Feature extraction : intfloat/e6-base was used.
        - Entities : entity names.
        - Article : articles abstract.
        - entity<->entity : predicate embedding.

# 4- Model architecture :

## 4-a Encoder Layer :

- Process Entity and entity<->entity links with an efficient GNN layer TransGNN that scales automatically for large number of relations.
- Process Article and article<->article link with R-GCN
- Use a bipartite GNN to update entities representations based on articles representations and article<->entity links.

## 4-b readout function :

- Use a readout function to generate article-level embeddings based on entity-level embeddings

## 4-b classification head : 

- Use a Linear layer as classification head to predict article-level labels.

# 5- Expertise Map construction :

## 5-a clustering :

- use classification head on entity-level representation skipping readout function to get entity-level labels.
- use entity-level labels to weight graph edges using jaccard index.
- run a community detection algorith like louvain / leiden.

## 5-b clusters labeling :

- For each cluster combine entities representations using the same readout function to generate community-level
embeddings and then use classification head to get community-level domains.

## 5-c Entities filtering & Community labeling :

- We use two scores to filter out irrelevant and general entities from each community (relevancy and generality scores).
- Measure node importance using the following mechanism :
    - Run louvain / leiden on unweighted graph and get probability assignments and compute entropy of each node.
    - initial node scores : (1-entropy) * (1-relevancy_score) * (1-generality_scores)
    - run personalized pagerank using the initial scores to get a score that combines both semantic and structural importance.
- For each community chose top-k most important nodes and feed them to an LLM to generate community-level name. 