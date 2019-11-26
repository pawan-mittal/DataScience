# K-Nearest Neighbors

## Cheat Sheet

### Scaling 
```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)
```

### Result (Classification)

```
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_scaled, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_scaled, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_scaled, y_test)))

example_fruit = [[5.5, 2.2, 10, 0.70]]
example_fruit_scaled = scaler.transform(example_fruit)
print('Predicted fruit type for ', example_fruit, ' is ', target_names_fruits[knn.predict(example_fruit_scaled)[0]-1])

```       

### Result (Regression)

```
from sklearn.neighbors import KNeighborsRegressor

X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)

knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)

print(knnreg.predict(X_test))
print('R-squared test score: {:.3f}'
     .format(knnreg.score(X_test, y_test)))
```

## Classification

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/knn/classified/1.png)

![2](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/knn/classified/2.png)

![3](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/knn/classified/3.png)

## Regression


