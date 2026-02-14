# requirements
pip install --no-cache-dir -r requirements.txt
cd ../../external && git clone git@github.com:abdelnour13/pycli.git
pip install -e .

# resources
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt_tab
python -m nltk.downloader punkt averaged_perceptron_tagger_en

# hf-cli
curl -LsSf https://hf.co/cli/install.sh | bash

# PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126