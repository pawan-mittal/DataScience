# Applied Machine Learning: Module 3 (Evaluation)

## Evaluation for Classification


## [Confusion matrices](confusion-matrices.md "1")

## Load Libs
```
%matplotlib notebook
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

dataset = load_digits()
X, y = dataset.data, dataset.target

#print ("X", y)


for class_name, class_count in zip(dataset.target_names, np.bincount(dataset.target)):
    print(class_name,class_count)

```

## Creating a dataset with imbalanced binary classes:  
```
# Negative class (0) is 'not digit 1' 
# Positive class (1) is 'digit 1'
y_binary_imbalanced = y.copy()
y_binary_imbalanced[y_binary_imbalanced != 1] = 0

print('Original labels:\t', y[1:30])
print('New binary labels:\t', y_binary_imbalanced[1:30])
```

## Accuracy of Support Vector Machine classifier

```
X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1).fit(X_train, y_train)
svm.score(X_test, y_test)
```

## Dummy Classifiers and Linear SVC

DummyClassifier is a classifier that makes predictions using simple rules, which can be useful as a baseline for comparison against actual classifiers, especially with imbalanced classes.

```
from sklearn.dummy import DummyClassifier

# Negative class (0) is most frequent
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
# Therefore the dummy 'most_frequent' classifier always predicts class 0
y_dummy_predictions = dummy_majority.predict(X_test)

y_dummy_predictions

dummy_majority.score(X_test, y_test)

svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
svm.score(X_test, y_test)

```

