# Data preprocessing

## Feature extraction

Intended features:
```
{tokens}, has_ip, length, n_dots, n_underscores, n_dashes, n_numbers
```

### Tokenization

Tokens can include `/`, `:`.  
URLS are tokenized into words for vectorization.  

#### Word embedding/Vectorization

Tier 1 (count based vectorization):  
- TF-IDF (weighted bag of words) -> RF/XGBoost/stacking.
- Bag of Words

**Tier 2 (predictive embeddings):**  
These use neural networks.
- Word2Vec or GloVe or FastText

Tier 3 (contextual embeddings):  
- BERT, GPT

Tokens are also lowercase, lemmentaized and stop words are removed.

# Machine learning models

Base classifiers: 
- Logistic Regression.
- SVM for high dimensions, avoid overfitting with small dataset.

Proposed model: 
- Logistic Regression + SVC + Decision Tree.
- Character level CNN/LTSM or tranformer tokenizers + lexical features.

## Testing

Dataset will be split into learning and training datasets using sklearn `train_test_split`.

---

# Wanted improvements

Better model testing procedure:

1. Train set → model learning
2. Validation set → tuning
3. In-distribution test set → same source, unseen samples
4. Out-of-distribution (OOD) test set → different source/time

Test character level CNN.  
Test BERT/transformers.  
Combined datasets.  

---

Other resources:  
- [Phishing Detection System Through Hybrid Machine Learning Based on URL](https://ieeexplore.ieee.org/abstract/document/10058201)
