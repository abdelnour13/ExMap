cd ../../external && mkdir openie6 && cd openie6
git clone https://github.com/dair-iitd/openie6.git
cd openie6
pip install -r requirements.txt

python -m nltk.downloader stopwords
python -m nltk.downloader punkt 

pip uninstall gevent greenlet -y
pip install --no-binary :all: gevent greenlet # 1.4.0 2.0.2
pip install bs4 # 0.0.2

zenodo_get 4054476 && tar -xvf openie6_data.tar.gz && rm openie6_data.tar.gz
zenodo_get 4055395 && tar -xvf openie6_models.tar.gz && rm openie6_models.tar.gz

############ NOTE ############
# If you are using docker you need gpu driver access even when not intending to use gpu "--gpus 0" is not enough i guess.
# Also you may need to go to openie6/run.py and change this line "has_cuda = torch.cuda.is_available()" to
# this : "has_cuda = False".
##############################