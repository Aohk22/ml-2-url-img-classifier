Evaluation: description, training process, results, demonstration.

# URL Classification

## Data preprocessing

[Standard feautures.](./notes.md#features)

### Tokenization

Split using on `urllib` + custom splitting using charset.

Tokens are also turned lowercase, lemmatized.

Remove stop-words.

### Word embedding

TF-IDF (weighted bag of words).

## Model training

A set of models will be trained then compared.

Logistic Regression, Random Forest,  XGBoost.

Character CNN.

# Results

[Models results.](./notes.md#classical-models)

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

[Models results.](./notebooks/train-image-model.ipynb)

---

Other resources:  

- [Phishing Detection System Through Hybrid Machine Learning Based on URL](https://ieeexplore.ieee.org/abstract/document/10058201)
- [ROC-AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
