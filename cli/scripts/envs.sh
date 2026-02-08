### AllenNLP env
conda create -n text2kg-allennlp python=3.8.20 pip
conda activate text2kg-allennlp
pip install --no-cache-dir allennlp allennlp_models

### Openi6 env
conda create -n text2kg-allennlp python=3.6 pip
conda activate text2kg-allennlp
cd ../RPC/openie6
chmod +x setup.sh && bash setup.sh

### Main env
cd ../../main
conda create -n text2kg-main python=3.12.12
pip install -r requirements.txt