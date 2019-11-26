# Import required modules and load data file

```
%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_table('file-path')
df.head()

```
## Dateset
![scatter](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/datasets/m1.png)

# Examining the data
## plotting a scatter matrix

``` 
from matplotlib import cm
X = df[['height', 'width', 'mass', 'color_score']]
y = df['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)
```

![scatter](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module1/scatter.png)

## create a mapping from fruit label value to fruit name to make results easier to interpret
``` 
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))   
lookup_fruit_name
```

## plotting a 3D scatter plot
```
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()
```
![3D](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module1/scatter-3d.png)

## Default Steps for ML Model (Train Model)
```
# For this example, we use the mass, width, and height features of each fruit instance
X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

# default is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

```

## Test Model
```
knn.score(X_test, y_test) // 0.53333333333333333

fruit_prediction = knn.predict([[100, 6.3, 8.5]])
lookup_fruit_name[fruit_prediction[0]]
```

## Visualise - Plot the decision boundaries of the k-NN classifier

### How sensitive is k-NN classification accuracy to the choice of the 'k' parameter?

```
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);
```

![3D](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module1/scatter-visualization.png)


### How sensitive is k-NN classification accuracy to the train/test split proportion?

```
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 5)

plt.figure()

for s in t:

    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy');
```

![3D](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module1/scatter-visualization2.png)
