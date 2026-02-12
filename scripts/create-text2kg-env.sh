conda create -n text2kg-allennlp python=3.6 pip
conda activate text2kg-allennlp
cd ../docker/text2kg
chmod +x setup.sh && bash setup.sh