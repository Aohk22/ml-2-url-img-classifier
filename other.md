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

ROC curve is drawn by graphing TPR over FPR, it summarizes all confusion matrix each threshold produce, it is used to identify threshold for making descision.

AUC makes it easy to compare one ROC to another.


The model with larger AUC will generally be the better one.

<img src="https://developers.google.com/static/machine-learning/crash-course/images/auc_abc.png" width="50%"/><br/>
<sub>Credit Google</sub>

A: for lower FPR  
B: for balance between TPR, FPR  
C: for higher TPR

This project will be optimized for higher TPR (increase TP and FP)

## CV

A convolutional layer uses filters and kernel to detect information about images.

https://setosa.io/ev/image-kernels/

A convolutional nerual network learns what kernel to use in training process.

Downsamples using pooling