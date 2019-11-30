## Neural Networks

### Activation functions

```
xrange = np.linspace(-2, 2, 200)

plt.figure(figsize=(7,6))

plt.plot(xrange, np.maximum(xrange, 0), label = 'relu')
plt.plot(xrange, np.tanh(xrange), label = 'tanh')
plt.plot(xrange, 1 / (1 + np.exp(-xrange)), label = 'logistic')
plt.legend()
plt.title('Neural network activation functions')
plt.xlabel('Input value (x)')
plt.ylabel('Activation function output')

plt.show()
```
![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module4/7.png)


## Neural networks: Classification
#### Synthetic dataset 1: single hidden layer
```
from sklearn.neural_network import MLPClassifier
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

fig, subaxes = plt.subplots(3, 1, figsize=(6,18))

for units, axis in zip([1, 10, 100], subaxes):
    nnclf = MLPClassifier(hidden_layer_sizes = [units], solver='lbfgs', random_state = 0).fit(X_train, y_train)
    
    title = 'Dataset 1: Neural net classifier, 1 layer, {} units'.format(units)
    
    plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train, X_test, y_test, title, axis)
    plt.tight_layout() 
```
![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module4/8.png)


### Synthetic dataset 1: two hidden layers
```
from adspy_shared_utilities import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

nnclf = MLPClassifier(hidden_layer_sizes = [10, 10], solver='lbfgs', random_state = 0).fit(X_train, y_train)

plot_class_regions_for_classifier(nnclf, X_train, y_train, X_test, y_test, 'Dataset 1: Neural net classifier, 2 layers, 10/10 units')
```
![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module4/9.png)

#### Regularization parameter: alpha
```
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

fig, subaxes = plt.subplots(4, 1, figsize=(6, 23))

for this_alpha, axis in zip([0.01, 0.1, 1.0, 5.0], subaxes):
    nnclf = MLPClassifier(solver='lbfgs', activation = 'tanh', alpha = this_alpha,
                         hidden_layer_sizes = [100, 100],
                         random_state = 0).fit(X_train, y_train)
    
    title = 'Dataset 2: NN classifier, alpha = {:.3f} '.format(this_alpha)
    
    plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train, X_test, y_test, title, axis)
    plt.tight_layout()
    
```
![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module4/10.png)

#### The effect of different choices of activation function
```
X_train, X_test, y_train, y_test = train_test_split(X_D2, y_D2, random_state=0)

fig, subaxes = plt.subplots(3, 1, figsize=(6,18))

for this_activation, axis in zip(['logistic', 'tanh', 'relu'], subaxes):
    nnclf = MLPClassifier(solver='lbfgs', activation = this_activation,
                         alpha = 0.1, hidden_layer_sizes = [10, 10],
                         random_state = 0).fit(X_train, y_train)
    
    title = 'Dataset 2: NN classifier, 2 layers 10/10, {} activation function'.format(this_activation)
    
    plot_class_regions_for_classifier_subplot(nnclf, X_train, y_train, X_test, y_test, title, axis)
    plt.tight_layout()
```
![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module4/11.png)

## Overview

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/neural-networks/1.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/neural-networks/2.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/neural-networks/3.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/neural-networks/4.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/neural-networks/5.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/neural-networks/6.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/neural-networks/7.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/neural-networks/8.png)

![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/models/supervised/neural-networks/9.png)


