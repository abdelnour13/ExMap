from allennlp.predictors import Predictor
from allennlp_models.coref import CoreferenceResolver
from xmlrpc.server import SimpleXMLRPCServer
from typing import Dict,Hashable

### Store Allennlp models
PORT = 8080
models : Dict[Hashable,Predictor] = {}

def get_predictor_from_path(path : str, device : int) -> Predictor:

    key = (path,)

    if key not in models:

        models[key] = Predictor.from_path(
            path, 
            cuda_device = device, 
            import_plugins=False
        )

    return models[key]

def allennlp_predict(path : str, device : int, text : str):
    model = get_predictor_from_path(path,device)
    return model.predict(text)

def main():

    ### Create Server
    server = SimpleXMLRPCServer(addr=("0.0.0.0", PORT), allow_none=True)

    ### Register functions
    server.register_function(allennlp_predict, name="allennlp_predict")
    server.register_function(get_predictor_from_path, name="load_predictor_from_path")

    ### Start Server
    print(f"Allennlp server Running at Port : {PORT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Allennlp server shutting down...")

if __name__ == '__main__':
    main()