cd ../docker/text2kg
conda create -n text2kg-main python=3.12.12
pip install -r requirements.txt

cd ../../external
git clone git@github.com:abdelnour13/pycli.git
pip install -e .