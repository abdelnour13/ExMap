pip install --no-cache-dir -r requirements.txt
python -m nltk.downloader stopwords
python -m nltk.downloader punkt 
python -m nltk.downloader averaged_perceptron_tagger_eng

pip uninstall gevent greenlet -y
pip install --no-binary :all: gevent==1.4.0 greenlet==2.0.2
pip install bs4==0.0.2 

# NOTE ###################################################################################################################
# If you are using docker you need gpu driver access even when not intending to use gpu "--gpus 0" is not enough i guess.#
# Also you may need to go to openie6/run.py and change this line "has_cuda = torch.cuda.is_available()" to               #
# this : "has_cuda = False".                                                                                             #
##########################################################################################################################