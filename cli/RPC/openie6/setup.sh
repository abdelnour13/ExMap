git clone https://github.com/dair-iitd/openie6.git
cd openie6
pip install -r requirements.txt

python -m nltk.downloader stopwords
python -m nltk.downloader punkt 

pip uninstall gevent greenlet -y
pip install --no-binary :all: gevent greenlet
pip install bs4

zenodo_get 4054476 && tar -xvf openie6_data.tar.gz && rm openie6_data.tar.gz
zenodo_get 4055395 && tar -xvf openie6_models.tar.gz && rm openie6_models.tar.gz

cd ..