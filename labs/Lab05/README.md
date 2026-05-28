# Practical Session — Week 6 (Embeddings and Vector Semantics)

This session moves beyond sparse count-based representations such as Bag-of-Words and TF-IDF and introduces **dense vector representations** of language using **word embeddings**.  
We will work with pre-trained embeddings through Gensim and use them to measure word similarity, solve simple analogies, visualize semantic neighborhoods, and build document representations for retrieval.

The focus for this session is on understanding what embeddings capture, how semantic similarity can be computed with vectors, and how embedding-based document retrieval compares to classical lexical TF-IDF search. Our running example is the FAQ retrieval task from previous weeks.

## Materials (notebooks)
Open the Jupyter notebooks in the `labs/` or repository root:
- `VL06-Embeddings.ipynb` — **seminar notebook**: live coding and conceptual exercises about word embeddings, cosine similarity, analogies, document vectors, and embedding visualization.  
- `P03-FAQ_Embeddings.ipynb` — **lab notebook**: hands-on implementation of FAQ retrieval using TF-IDF, mean-pooled embeddings, and TF-IDF-weighted embeddings.

Follow the repository instructions for setting up the Python environment and running Jupyter notebooks. You will also need the required NLP libraries and models used in the notebooks, especially Gensim and the pretrained models. Other libraries such scikit-learn, and the `en_core_web_sm` spaCy model should already be in our conda environment. 

## Seminar
Work through `VL06-Embeddings.ipynb`.  
This notebook provides guided examples of concepts we study in the lecture. Key topics:

- Loading and inspecting **pre-trained GloVe word embeddings** with Gensim  
- Understanding **dense word vectors**, vocabulary lookup, vector dimensions, and token coverage  
- Computing **cosine similarity** manually and with the Gensim API  
- Exploring **nearest neighbors** and what they reveal about semantic relatedness  
- Testing **word analogies** using vector arithmetic and reflecting on model bias  
- Building simple **document vectors** using mean pooling and TF-IDF-weighted pooling  
- Comparing semantic retrieval results and optionally visualizing embeddings with PCA  

> 💡 You are encouraged to try different words, analogy examples, and queries. Pay attention not only to successful examples, but also to surprising or biased results. You might even try other pretrained models. 

## Lab session (what to do in the lab)
Open `P03-FAQ_Embeddings.ipynb`. The notebook contains structured exercises that guide you through a FAQ retrieval task using different text representations:

1. **Dataset preparation** — download the English `clips/mfaq` JSON data, load it, and flatten question-answer pairs using `build_qa_simple()`.  
2. **Tokenization and preprocessing** — use spaCy to tokenize, remove punctuation and stop words, and compare lemma-based and raw-token representations.  
3. **TF-IDF retrieval baseline** — build a sparse TF-IDF index over FAQ answers and retrieve answers by cosine similarity.  
4. **Embedding-based retrieval** — load the `glove-wiki-gigaword-50` model and use pre-trained embeddings to represent answers and questions.  
5. **Mean-pooled embeddings** — implement `doc_vector_mean(text)` and an embedding search function that mirrors the provided retrieval code.  
6. **Weighted embeddings** — compare mean-pooled embeddings with TF-IDF-weighted embedding averages.  
7. **Evaluation** — compute Top-1 accuracy and Top-k accuracy / Recall@k for the different retrieval methods.  
8. **Analysis** — study the effect of document length, lemmatization, and allowing multiple retrieved answers.

Each step is described directly in the notebook, with the corresponding tasks. Follow the in-cell instructions carefully and add your explanations and observations.

## Deliverables / Submission
Follow the submission instructions provided in class and in the notebook. The deliverables include:
- Completed and **executed** notebook `P03-FAQ_Embeddings.ipynb`, including:
  - all code cells run successfully,  
  - implementation of the mean-pooled embedding retrieval task,  
  - evaluation results for TF-IDF, mean-pooled embeddings, and TF-IDF-weighted embeddings, and  
  - short written observations about document length, lemmatization, and Top-k retrieval.

Submissions are to be made through the **ILIAS system**, unless otherwise announced.

## Contact
If you need help or want to provide feedback, please contact me via:
- Email: `<name.lastname>@hsbi.de`
- The course ILIAS page
