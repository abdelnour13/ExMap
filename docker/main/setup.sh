# PyTorch
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu118

# requirements
pip install --no-cache-dir -r requirements.txt

# pycli
apt-get update && apt-get install -y git
pip install git+https://github.com/abdelnour13/pycli.git#egg=pycli


# resources
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt_tab
python -m nltk.downloader punkt averaged_perceptron_tagger_en

# hf-cli
curl -LsSf https://hf.co/cli/install.sh | bash