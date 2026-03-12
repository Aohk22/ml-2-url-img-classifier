Evaluation: description, training process, results, demonstration.

# URL Classification

## Data preprocessing

**Summary:**

```
{tokens}, has_ip, length, n_dots, n_underscores, n_dashes, n_numbers
```

### Tokenization

Split using on `urllib` + custom splitting using charset.

Tokens are also turned lowercase, lemmatized.

Remove stop-words.

### Word embedding

TF-IDF (weighted bag of words).

## Model training

A set of models will be trained then compared.

Logistic Regression, Random Forest, Character CNN, XGBoost.

# Results

---

# Image Classification

## Data processing

- Data should be labeled.
- Data augmentation: resizing, normalizing pixel values.
- Feature extraction: deep learning models learn their own features from the extracted raw image data.

## Models training

KNN: each image is represented as feature vector (pixel values, color histogram).

CNN: AlexNet - simple, ReLU as activation function.  

# Results

---

Other resources:  
- [Phishing Detection System Through Hybrid Machine Learning Based on URL](https://ieeexplore.ieee.org/abstract/document/10058201)
