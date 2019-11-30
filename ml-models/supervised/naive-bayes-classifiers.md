# Naive Bayes Classifiers

## Gaussian Naive Bayes classifier: Dataset 1
```
from sklearn.naive_bayes import GaussianNB
from adspy_shared_utilities import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state=0)
nbclf = GaussianNB().fit(X_train, y_train)
plot_class_regions_for_classifier(nbclf, X_train, y_train, X_test, y_test, 'Gaussian Naive Bayes classifier: Dataset 1')
```

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module4/1.png)

## Gaussian Naive Bayes classifier: Dataset 2

```
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)
nbclf = GaussianNB().fit(X_train, y_train)
plot_class_regions_for_classifier(nbclf, X_train, y_train, X_test, y_test, 'Gaussian Naive Bayes classifier: Dataset 2')
```

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module4/2.png)

## Application to a real-world dataset

```
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

nbclf = GaussianNB().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(nbclf.score(X_train, y_train)))
print('Accuracy of GaussianNB classifier on test set: {:.2f}'.format(nbclf.score(X_test, y_test)))
```

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/naive-bayes-classifiers/1.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/naive-bayes-classifiers/2.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/naive-bayes-classifiers/3.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/naive-bayes-classifiers/4.png)
