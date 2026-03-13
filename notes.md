## Todo

- [x] Data cleaning through model pipeline
- [x] Choose a model from comparisons
- [x] Specify a feature standard for future unprocessed data
- [x] More visualization
    - [x] First rows, null values, label distribution
    - [x] Visualize AUC
- [ ] Word tokenization into word embedding
- [ ] Character CNN
- [ ] Add outside database

## Other

Validation set: used during training to make descisions.

Test set: used at the end for estimate of performance.

<!-- //<Explain pipeline definitions> -->

### Metrics explanation

#### Accuracy

(TP+TN) / (TP+TN+FP+FN). Percentage of predictions that are correct.

Can be misleading if dataset is imbalanced.

#### Precision

TP / (TP+FP). 

E.g of all phishing URLs detected, which ones were actually phishing. Important when false alarms are costly.

#### Recall

TP / (TP+FN). True positive rate.

E.g of all phishing URLs, how many were detected. Important for when missing a threat is dangerous.

#### F1 score

F1 = 2 * (Precision * Recall) / (Precision + Recall).

Performance between detecting threats and avoiding false alarms.

#### ROC (receiver-operating characteristic) - AUC


ROC curve is drawn by graphing TPR over FPR, it is a visual representation of model performance across all thresholds.

The area under the ROC curve (AUC) represents the probability that the model, if given a randomly chosen positive and negative example, will rank the positive higher than the negative.

Example: An AUC of 1.0 means a classifier will **always** assign a random spam email a higher probability of being spam than a legitimate email.

The model with larger AUC will generally be the better one.

<img src="https://developers.google.com/static/machine-learning/crash-course/images/auc_abc.png" width="50%"/><br/>
<sub>Credit Google</sub>

A: for lower FPR  
B: for balance between TPR, FPR  
C: for higher TPR

This project will be optimized for higher TPR (increase TP and FP)

# Getting started using hugginface dataset

## Feature extraction

From: https://huggingface.co/datasets/pirocheto/phishing-url.

Features are from three different classes:

- 56 extracted from the structure and syntax of URLs
- 24 extracted from the content of their correspondent pages
- 7 are extracetd by querying external services.

For the feature standard, only features available directly in URL will be taken.


### Features

Which features will be extracted.

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

## Classical Models

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

Chosen model construction:

```python
'''
Imputer: replace missing values with median of value.
'''
(
    "hgb",
    Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
            "clf",
                HistGradientBoostingClassifier(
                    random_state=random_state,
                    max_depth=None,
                    learning_rate=0.1,
                ),
            ),
        ]
    ),
),
```

## What is `HistGradientBoostingClassifier()`



## Results using manual feature extraction

### Confusion matrix

![CF](./graphs/CF-manual-features.png)

### ROC

![ROC](./graphs/ROC-manual-features.png)

Model acheives high TPR and low FPR.

There is a 96% probability that model ranks a random phising URL higher than legitimate URL.

For phising a false negative is very costly therefore we will need to maximize the TPR (TPR and FNR are complements).

### Summary

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>roc_auc</th>
      <th>accuracy</th>
      <th>f1</th>
      <th>precision</th>
      <th>recall (TPR)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hgb</td>
      <td>0.961867</td>
      <td>0.887728</td>
      <td>0.886991</td>
      <td>0.892857</td>
      <td>0.881201</td>
    </tr>
  </tbody>
</table>
</div>