# Random Forests

# Ensembles of Decision Trees

## Complex Binary Dataset

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state = 0)
fig, subaxes = plt.subplots(1, 1, figsize=(6, 6))

clf = RandomForestClassifier().fit(X_train, y_train)
title = 'Random Forest Classifier, complex binary dataset, default settings'
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, X_test, y_test, title, subaxes)

plt.show()
```

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module4/3.png)

## Random forest: Fruit dataset

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

X_train, X_test, y_train, y_test = train_test_split(X_fruits.as_matrix(),
                                                   y_fruits.as_matrix(),
                                                   random_state = 0)
fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))

title = 'Random Forest, fruits dataset, default settings'
pair_list = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]

for pair, axis in zip(pair_list, subaxes):
    X = X_train[:, pair]
    y = y_train
    
    clf = RandomForestClassifier().fit(X, y)
    plot_class_regions_for_classifier_subplot(clf, X, y, None, None, title, axis, target_names_fruits)
    
    axis.set_xlabel(feature_names_fruits[pair[0]])
    axis.set_ylabel(feature_names_fruits[pair[1]])
    
plt.tight_layout()
plt.show()

clf = RandomForestClassifier(n_estimators = 10,random_state=0).fit(X_train, y_train)

print('Random Forest, Fruit dataset, default settings')
print('Accuracy of RF classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
```

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module4/4.png)

### Random Forests on a real-world dataset
```
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = RandomForestClassifier(max_features = 8, random_state = 0)
clf.fit(X_train, y_train)

print('Breast cancer dataset')
print('Accuracy of RF classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
```


![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/random-forests/1.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/random-forests/2.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/random-forests/3.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/random-forests/4.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/random-forests/5.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/random-forests/6.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/random-forests/7.png)
