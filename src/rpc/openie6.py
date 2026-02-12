import argparse
import torch
import os
import shutil
import tempfile
import logging
import sys
sys.path.insert(0, os.path.join('external', 'openie6', 'openie6', 'imojie', 'benchmark'))
sys.path.insert(0, os.path.join('external', 'openie6', 'openie6', 'imojie'))
sys.path.insert(0, os.path.join('external', 'openie6', 'openie6'))
from pytorch_lightning import Trainer
from external.openie6.openie6.model import Model
from external.openie6.openie6.params import add_args
from external.openie6.openie6.data import override_args, convert_to_namespace, process_data, pad_data
from pytorch_lightning import Trainer
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from types import SimpleNamespace
from xmlrpc.server import SimpleXMLRPCServer

logging.getLogger('transformers.configuration_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)

PORT = 8081
OPENIE6_DIR = os.path.join('external', 'openie6', 'openie6')
OIE_MODEL_PATH = os.path.join(OPENIE6_DIR, "models","oie_model","epoch=14_eval_acc=0.551_v0.ckpt")
CONJ_MODEL_PATH = os.path.join(OPENIE6_DIR, "models","conj_model", "epoch=28_eval_acc=0.854.ckpt")

MODELS = {}

def clean_up():
    """
    clean up openie6 side effects.
    """
        
    if os.path.exists(path='carb'):
        shutil.rmtree('carb')

    if os.path.exists(path="lightning_logs"):
        shutil.rmtree('lightning_logs')

    if os.path.exists(path="utils"):
        shutil.rmtree('utils')

def copy_resources():
    """
    copies resources needed by openie6 into the interpreter path
    """

    for root, _, files in os.walk(OPENIE6_DIR):
        
        for file in files:
            
            extention = os.path.splitext(file)[1]

            if extention in ['.tsv','.csv','.text']:
                src = os.path.join(root, file)
                dst = src.replace(OPENIE6_DIR, ".")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
                print(f"Copied {src} to {dst}")

def cleanup():
    os.remove(os.path.join())

def load_model_for_task(
    task : str,
    gpus : int = 0
) -> Tuple[Trainer,Model,SimpleNamespace]:

    task = task.lower()
    
    if task in MODELS:
        return MODELS[task]

    ### Arg-List
    args = {
        "mode": "splitpredict",
        "inp": "dummy_input",
        "out": "dummy_output",
        "task" : "oie",
        "gpus": gpus,
        "oie_model": OIE_MODEL_PATH,
        "conj_model": CONJ_MODEL_PATH,
        "num_extractions" : 5
    }

    args = ' '.join([f"--{key} {value}" for key, value in args.items() ]).split()

    ### Parse Hyper-parameters & get the default parameters
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = add_args(parser)

    hparams = parser.parse_args(args)

    hparams.model_str = 'bert-base-cased'
    hparams.mode = 'predict'

    ### Model-Specific Parameters
    if task == 'conj':
        hparams.task = 'conj'
        hparams.checkpoint = hparams.conj_model
    elif task == "oie":
        hparams.task = 'oie'
        hparams.checkpoint = hparams.oie_model

    ### Load The Model
    loaded_hparams_dict = torch.load(hparams.checkpoint, map_location='cpu')['hparams']
    current_hparams_dict = vars(hparams)
    loaded_hparams_dict = override_args(loaded_hparams_dict, current_hparams_dict, args)
    loaded_hparams = convert_to_namespace(loaded_hparams_dict)
    
    model = Model(loaded_hparams, None)

    trainer = Trainer(gpus=hparams.gpus,resume_from_checkpoint=hparams.checkpoint)

    MODELS[task] = (trainer, model, loaded_hparams)

    return MODELS[task]

def load_conj_model() -> Tuple[Trainer,Model,SimpleNamespace]:
    return load_model_for_task(task='conj')

def load_oie_model() -> Tuple[Trainer,Model,SimpleNamespace]:
    return load_model_for_task(task='oie')

def conj_model_predict(
    model : Model,
    trainer : Trainer, 
    hparams : SimpleNamespace, 
    sentences : List[str]
) -> List[Tuple[str, str]]:
    
    model.all_predictions_conj = []
    model.all_sentence_indices_conj = []
    model.all_conjunct_words_conj = []
    model.all_predictions_oie = []

    with tempfile.TemporaryDirectory() as tdir:
        
        ### Temporary Files
        inp = os.path.join(tdir, 'sentences.txt')
        out = os.path.join(tdir, 'triples')

        with open(inp, "w") as f:
            f.write("\n".join(sentences))

        hparams.inp = inp
        hparams.out = out
        
        ### Create dataset & dataloader
        _, _, test_dataset, META_DATA, _ = process_data(hparams)
        model._meta_data_vocab = META_DATA
        model.all_sentences = sentences

        dataloader = DataLoader(
            test_dataset, 
            batch_size=hparams.batch_size,
            collate_fn=pad_data, 
            shuffle=False, 
            num_workers=1
        )
        
        ### Run model
        trainer.test(model, test_dataloaders=dataloader)

        ### Retrieve sentences
        sentences = []

        for sent_idx, line in enumerate(model.all_predictions_conj):

            lines = line.split('\n')
            og_sentence = lines[0]
            rest = lines[1:]
            rest = list(filter(lambda x : len(x) > 0, map(lambda x : x.strip(), rest)))

            if len(rest) == 0:
                sentences.append((sent_idx,og_sentence))
            else:
                for sent in rest:
                    sentences.append((sent_idx,sent))

        return sentences

def oie_model_predict(
    model : Model,
    trainer : Trainer, 
    hparams : SimpleNamespace, 
    sentences : List[str]
) -> List[Dict]:
        
    model.all_predictions_conj = []
    model.all_sentence_indices_conj = []
    model.all_conjunct_words_conj = []
    model.all_predictions_oie = []

    with tempfile.TemporaryDirectory() as tdir:

        ### Temporary Files
        inp = os.path.join(tdir, 'sentences.txt')
        out = os.path.join(tdir, 'triples')

        with open(inp, "w") as f:
            f.write("\n".join(sentences))

        hparams.inp = inp
        hparams.out = out
        
        ### Create dataset & dataloader
        _, _, test_dataset, META_DATA, _ = process_data(hparams)
        model._meta_data_vocab = META_DATA

        dataloader = DataLoader(
            test_dataset, 
            batch_size=hparams.batch_size,
            collate_fn=pad_data, 
            shuffle=False, 
            num_workers=1
        )
        
        ### Run model
        trainer.test(model, test_dataloaders=dataloader)

        ### Read output & Parse Triples
        triples = []
        sent_idx = -1

        with open(f"{out}.oie", "r") as f:

            current_sentence = None
            flag = True
            
            for line in f:

                if flag:
                    sent_idx += 1
                    current_sentence = line.strip()
                    flag = False
                elif line != "\n":

                    ### Initialize Triple
                    triple = {}
                    triple['sentence'] = current_sentence
                    triple['sent_idx'] = sent_idx

                    ### Split Score from the triple
                    score, t = line.strip().split(':')
                    triple['score'] = float(score)

                    ### Split head, predicate and tail
                    head, predicate, tail = t.split(';')
                    triple['head'] = head.strip("()\n ")
                    triple['predicate'] = predicate.strip("()\n ")
                    triple['tail'] = tail.strip("()\n ")

                    ### Add To Triples
                    triples.append(triple)
                else:
                    flag = True

        return triples
    
def extract_triples(sentences : List[str]) -> List[Dict]:

    ### Get Models
    conj_trainer, conj_model, conj_hparams = load_conj_model()
    oie_trainer, oie_model, oie_hparams = load_oie_model()

    ### Run Conj Models
    sentences_coordinated = conj_model_predict(conj_model, conj_trainer, conj_hparams, sentences)

    ### Run OIE Model
    input_sentences = [
        sent 
        for _, sent in sentences_coordinated
    ]

    map = {
        sent_idx  : og_sent_idx
        for sent_idx,(og_sent_idx, sent) in enumerate(sentences_coordinated)
    }

    triples = oie_model_predict(oie_model, oie_trainer, oie_hparams, input_sentences)

    for triple in triples:
        triple['sent_idx'] = map[triple['sent_idx']]

    return triples

def main():

    ### Create Server
    server = SimpleXMLRPCServer(addr=("localhost", PORT), allow_none=True)

    ### Register functions
    server.register_function(load_conj_model, name="load_conj_model")
    server.register_function(load_oie_model, name="load_oie_model")
    server.register_function(conj_model_predict, 'conj_model_predict')
    server.register_function(oie_model_predict, 'oie_model_predict')
    server.register_function(extract_triples, 'extract_triples')

    ### Start Server
    print(f"Openie6 server Running at Port : {PORT}")

    try:
        copy_resources()
        server.serve_forever()
    except KeyboardInterrupt:
        print("Openie6 server shutting down...")
    finally:
        clean_up()

if __name__ == '__main__':
    main()