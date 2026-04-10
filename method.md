

Goal : Constructing expertise map from knowledge graph & textual data guided by a set of pre-defined domains of interest.

Current Method : 

    - Data Preparation :
        - Starts with a corpus of articles with their abstracts, keywords, titles.
        - Preprocessing : Handle html tags, latex equations and urls and keep only english articles.
        - Use an LLM to zero-shot classify the articles and assign them to zero, one or more domains, an article is represented 
        by its title, abstract and a set of keywords a domain is represented by a name and a description.

    - Acronym Extraction : 
        briefly it uses a combination of patterns for candidates extraction + QA model to extract expansions of the acronyms +
            thresholding the confidance score to minimize wrong pairs ration to total extracted pairs.

    - Knowledge Graph Construction :
        - Coreference resolution.
        - Sentence segmentation.
        - LLM based triple extractor.
        - Entity & Predicate cleaning.
        - Entities mapping : map entities that their embedding using a pre-trained language model is greater than 0.95 + map acronyms to their expansions.
        - Knowledge graph completion : add triples from external knowledge graph to enhance connectivity and add facts that maybe missing.

    - Training : 

        - Graph : Create a super graph that contains 
            - Knowledge Graph.
            - The links between entities and articles.
            - Relations between articles based on common-author and common-keyword.

        - Model : 
            - Use a GNN layer specific to KGs so it can handle large number of relations to update entities representations based on their neighbors.
            - Use an R-GNN (GCN or GAT) to update articles representations based on their neighbors.
            - Use a regular GNN (bipartite) to update entities representations based on the articles.
            - Jumping Knowledge added to prevent over-smoothing.
            - Encoder is composed of several layers of the previously described process.
            - Final entities representations are aggregated using (mean, max or attn) readout function to generate article-level representations.
            - article-level representations passed to a classification head.
            - probabilities are compared with LLM labels the loss is computed and weights updated.

    - Expertise Map Construction : 

        - Take knowledge graph and weight edges based on the similarity of the head and tail embeddings.
        - Perform louvain or leiden algorithms.
        - For each community a community-level representation is computed using the same readout function, passed to classification head to assign them to domains.
        - For each community a representative node is selected (currently by using a set of rules like : degree limitation to avoid common nodes, sort by page-rank score, prefer entities that are also keywords...etc).

Areas where enhancement is needed : 

    - Some domains are very rare causing them not have correspondant exprtises in the final map.
    - The need to filter-out entities that are too general for example some entities include : equation, model, data, france are not clearly associated with a given domains in contrast to entities like : cobot, natural language processing, ontology that are. A potential solution that i found is to only restric to the set of keyword that came with the artciles + set of keywords extracted automatically using "keyphrasetransformer" which is a model that used for keyword or noun phrases extraction from text based on transformers (maybe add aother models ensemble them or take the union of their results for a more rich set).


Suggestions :

