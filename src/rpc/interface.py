from xmlrpc.client import ServerProxy

allennlp = ServerProxy(uri="http://localhost:8080")

def load_predictor_from_path(path : str, device : int) -> None:
    allennlp.load_predictor_from_path(path, device)

def allennlp_predict(path : str, device : int, text : str):
    return allennlp.allennlp_predict(path, device, text)