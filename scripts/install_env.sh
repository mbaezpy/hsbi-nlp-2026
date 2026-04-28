#!/bin/bash

# -------------------------------------------------------
# NLP WS25 Course Environment Setup (with GPU Support)
# -------------------------------------------------------

# Optional: Install Miniconda3 (if not already available)
# Uncomment the lines below if you don't have conda installed
# echo "Installing Miniconda..."
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# bash miniconda.sh -b -p $HOME/miniconda3
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# echo "Miniconda installed and ready to use"

# Load Miniconda module 
module load miniconda3

# Create new environment with Python 3.10
conda create -n nlp-course python=3.10 conda pip --yes

# +--+ Activate the environment
conda activate nlp-course

# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Alternative: Install PyTorch for CPU only (uncomment to use - do not install cpu and cuda versions at the same time)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install NLP and data science libraries from PyPI
pip install \
  transformers datasets \
  pandas scikit-learn matplotlib \
  nltk spacy tqdm ipykernel jupyter

# Register the environment as a Jupyter kernel
python -m ipykernel install --user --name nlp-course --display-name "NLP Course"

# Download essential NLP data
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

echo "NLP Course environment setup complete!"
