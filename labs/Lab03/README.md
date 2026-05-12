# Practical Session — Week 4 (Naïve Bayes Text Classification)

This session extends our work on text processing and introduces **statistical classification** using the **Naïve Bayes (NB)** model.  
We will build on concepts from the lecture and implement a complete spam classifier using word frequencies, feature engineering, and evaluation metrics.

The focus for this session is on understanding how probabilistic classifiers work, experimenting with text representations, and interpreting model behavior.

## Materials (notebooks)
Open the Jupyter notebooks in the `labs/` or repository root:
- `VL04_Text_Classification_NB.ipynb` — **seminar notebook**: live coding, and conceptual exercises about Naïve Bayes, bag-of-words, inference, and smoothing.  
- `P02_NB_Spam_filter.ipynb` — **lab notebook**: hands-on implementation of a spam classifier using Naïve Bayes, dataset preparation, and error analysis.

Follow the repository instructions for setting up the Python environment and running Jupyter notebooks.

## Seminar
Work through `VL04_Text_Classification_NB.ipynb`.  
This notebook provides guided examples of concepts we study in the lecture. Key topics:

- Revisiting the **Bag-of-Words** representation  
- Understanding **training vs inference** in Naïve Bayes  
- Interpreting **model outputs**: log probabilities, priors, and feature importance  
- Exploring **model limitations** (qualitative analysis) and introducing engineered features  

> 💡 You are encouraged to run, modify, and discuss small code snippets to observe the effect of changes on probabilities and predictions.

## Lab session (what to do in the lab)
Open `P02_NB_Spam_filter.ipynb`. The notebook contains structured exercises that guide you through the full Naïve Bayes pipeline:

1. **Dataset preparation** — load, inspect, and split the spam dataset into training and test partitions.  
2. **Text representation** — apply the Bag-of-Words vectorizer, and understand how the vocabulary is built.  
3. **Model training** — fit a `MultinomialNB` model and interpret the learned probabilities.  
4. **Evaluation** — use accuracy, precision, recall, and confusion matrices to assess performance.  
5. **Error analysis** — inspect false positives and false negatives using helper functions.  
6. **Feature engineering** — experiment with additional signals such as punctuation, uppercase words, URLs, or emojis, and observe their impact.

Each step is described directly in the notebook, with the corresponding tasks. Follow the in-cell instructions carefully and add your explanations and observations.

## Deliverables / Submission
Follow the submission instructions provided in class and in the notebook. The deliverables include:
- Completed and **executed** notebook `P02_NB_Spam_filter.ipynb`, including:
  - all code cells run successfully,  
  - answers to open questions, and  
  - notes on your observations during feature analysis and error inspection.

Submissions are to be made through the **ILIAS system**, unless otherwise announced.

## Contact
If you need help or want to provide feedback, please contact me via:
- Email: `<name.lastname>@hsbi.de`
- The course ILIAS page