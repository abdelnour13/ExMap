from xmlrpc.client import ServerProxy

allennlp = ServerProxy(uri="http://localhost:8080")

def load_predictor_from_path(path : str, device : int) -> None:
    allennlp.load_predictor_from_path(path, device)

def allennlp_predict(path : str, device : int, text : str):
    return allennlp.allennlp_predict(path, device, text)

if __name__ == '__main__':
    
    load_predictor_from_path('https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz', -1)
    print("model loaded")

    result = allennlp_predict(
        'https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz',
        -1,
        """
        The main objective of this thesis is to combine the advantages of probabilistic graphical model learning and formal verification 
        in order to build a novel strategy for security assessments. The second objective is to assess the security of a given system 
        by veri- fying whether it satisfies given properties and, if not, how far is it from satisfying them. We are interested in 
        performing formal verification of this system based on event sequences col- lected from its execution. Consequently, 
        we propose a model-based approach where a Re- cursive Timescale Graphical Event Model (RT- GEM), learned from the event streams, 
        is con- sidered to be representative of the underlying system. This model is then used to check a se- curity property. 
        If the property is not verified, we propose a search methodology to find an- other close model that satisfies it.
        """
    )

    print(result)