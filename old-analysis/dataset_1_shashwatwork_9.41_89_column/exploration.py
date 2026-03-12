#!/usr/bin/env python

# old code, main in baseline-url.ipynb file

import pandas as pd
import numpy as np
import gensim
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Read data.
original_data = pd.read_csv('/var/my-data/ds-1-url/dataset_phishing.csv')
print(original_data)

# Trimming.
trimmed_data = original_data.filter(items=['url', 'status'])
print(trimmed_data)

# Checking.
## For null values.
assert((True in trimmed_data.isna().values) == False)

# Exploring.
## Value statistics.
print(trimmed_data.describe())
print()
print(trimmed_data['status'].value_counts(normalize=True))

# Data conversion.
status_mapping = {'legitimate': 0, 'phishing': 1}
trimmed_data['status'] = trimmed_data['status'].map(lambda x: status_mapping[x]).astype('int32')
print(trimmed_data)

# Partitioning data.
X = trimmed_data[['url']]
y = trimmed_data[['status']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Tokenizing all the URLs into a single list.
def tokens_from_url():


tokens = []
for url in X_train.loc[:, 'url']:


exit()
df_train = pd.DataFrame(data={'text': X_train, 'label': y_train})
df_test = pd.DataFrame(data={'text': X_test, 'label': y_test})
mapping = {'legitimate': 0, 'phishing': 1}
df_train['label'] = df_train['label'].map(lambda x: mapping[x]).astype('float32')
df_test['label'] = df_test['label'].map(lambda x: mapping[x]).astype('float32')

tokens = list()
for url in X_train:
    o = urlparse(url)
    temp = list()
    for part in [o.scheme, o.netloc, o.path, o.params, o.query, o.fragment]:
        if part:
            temp.append(part)
    tokens.append(temp)
wem = Word2Vec(tokens, min_count=1, vector_size=100, window=5, sg=0)

def to_vec(tokens, model):
    vecs = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

X = np.vstack([
    doc_vector(tokens, wem)
    for tokens in tokenized_texts
])

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

