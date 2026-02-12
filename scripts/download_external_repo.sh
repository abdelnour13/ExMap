cd ../external

git clone git@github.com:abdelnour13/pycli.git
mkdir openie6 && cd openie6 && git clone https://github.com/dair-iitd/openie6.git

cd openie6
zenodo_get 4054476 && tar -xvf openie6_data.tar.gz && rm openie6_data.tar.gz
zenodo_get 4055395 && tar -xvf openie6_models.tar.gz && rm openie6_models.tar.gz