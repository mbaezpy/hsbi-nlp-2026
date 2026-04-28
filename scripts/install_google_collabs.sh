
# Download essential NLP data
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger wordnet punkt_tab averaged_perceptron_tagger_eng
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

pip install compound-split emoji