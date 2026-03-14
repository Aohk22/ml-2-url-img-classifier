## Todo

- [x] Data cleaning through model pipeline
- [x] Choose a testing model from comparisons
- [x] Specify a feature standard for future unprocessed data
- [x] More visualization
    - [x] First rows, null values, label distribution
    - [x] Visualize AUC
- [x] Word tokenization into word embedding

- [ ] Use a neural network to detect phishing site from images.

- [ ] Character CNN
- [ ] Add outside database

# Models training

## Feature extraction

From: https://huggingface.co/datasets/pirocheto/phishing-url.

Features are from three different classes:

- 56 extracted from the structure and syntax of URLs
- 24 extracted from the content of their correspondent pages
- 7 are extracetd by querying external services.

For the feature standard, only features available directly in URL will be taken.

```python
[
    'url', 'length_url', 'length_hostname', 'ip', 

    'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 'nb_subdomains',

    # ratio
    'ratio_digits_url', 'ratio_digits_host',
     
    # lexical
    'not_https', 'http_in_path', 'tld_in_path', 'tld_in_subdomain',    

    'prefix_suffix', # checks for hyphen in domain (boolean)

    'path_extension',    

    'shortest_words_raw', 'shortest_word_host', 'shortest_word_path', 'longest_words_raw', 'longest_word_host', 'longest_word_path', 

    'brand_in_subdomain', 'brand_in_path', 'status'
]
```

## Baseline metrics

```
...trimmed
Dataset: pirocheto/phishing-url
Train shape: (6126, 89)
Val shape: (1532, 89)
Test shape: (3772, 89)
Columns (first 25): ['url', 'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore', 'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma', 'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 'nb_dslash', 'http_in_path']
```

```
Features: 87
Best model: hgb
VAL metrics: {
        'roc_auc': 0.9898424558078656,
        'accuracy': 0.9543080939947781,
        'f1': 0.9544270833333334,
        'precision': 0.951948051948052,
        'recall': 0.9569190600522193
}
TEST metrics: {
        'roc_auc': 0.9934132622021503,
        'accuracy': 0.9647401908801697,
        'f1': 0.9649354073292908,
        'precision': 0.9596224436287363,
        'recall': 0.9703075291622482
}

All candidates:
    ...trimmed
    rf {
        'roc_auc': 0.9877896433952102,
        'accuracy': 0.9575718015665796,
        'f1': 0.9574885546108568,
        'precision': 0.9593709043250328,
        'recall': 0.9556135770234987
    }
    logreg {'roc_auc': 0.980352991703536,
        'accuracy': 0.9386422976501305,
        'f1': 0.9381578947368421,
        'precision': 0.9456233421750663,
        'recall': 0.9308093994778068
   }
```

## Results using features

### Confusion matrix

![CF](./graphs/CF-manual-features.png)

### ROC

![ROC](./graphs/ROC-manual-features.png)

Model acheives high TPR and low FPR.

There is a 96% probability that model ranks a random phising URL higher than legitimate URL.

For phising a false negative is very costly therefore we will need to maximize the TPR (TPR and FNR are complements).

### Summary

| | model | roc_auc  | accuracy | f1       | precision | recall (TPR) |
|--- | ----- | -------- | -------- | -------- | --------- | ------------ |
| 0| hgb   | 0.961867 | 0.887728 | 0.886991 | 0.892857  | 0.881201     |


## Results with standard features + word embedding

### Confusion matrix

![CF](./graphs/CF-manual-features-embed.png)

### AUC

![AUC](./graphs/AUC-manual-features-embed.png)

Model acheives even higher AUC with word embedding.

### Summary

| | model | roc_auc | accuracy | f1 | precision | recall |
|---|---|---|---|---|---|---|
| 0 | hgb |	0.974091 | 0.91188 | 0.912281 | 0.90815 | 0.916449 |