# Import required modules and load data file

%matplotlib notebook

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_table('file-path')
df.head()

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

![scatter](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/m1-s1.png)


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
![3D](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/m1-s1-3d.png)

