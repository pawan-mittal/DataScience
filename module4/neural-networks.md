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

## Neural networks: Regression

```
from sklearn.neural_network import MLPRegressor

fig, subaxes = plt.subplots(2, 3, figsize=(11,8), dpi=70)

X_predict_input = np.linspace(-3, 3, 50).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X_R1[0::5], y_R1[0::5], random_state = 0)

for thisaxisrow, thisactivation in zip(subaxes, ['tanh', 'relu']):
    for thisalpha, thisaxis in zip([0.0001, 1.0, 100], thisaxisrow):
        mlpreg = MLPRegressor(hidden_layer_sizes = [100,100],
                             activation = thisactivation,
                             alpha = thisalpha,
                             solver = 'lbfgs').fit(X_train, y_train)
        y_predict_output = mlpreg.predict(X_predict_input)
        thisaxis.set_xlim([-2.5, 0.75])
        thisaxis.plot(X_predict_input, y_predict_output,
                     '^', markersize = 10)
        thisaxis.plot(X_train, y_train, 'o')
        thisaxis.set_xlabel('Input feature')
        thisaxis.set_ylabel('Target value')
        thisaxis.set_title('MLP regression\nalpha={}, activation={})'.format(thisalpha, thisactivation))
        plt.tight_layout()
```
![1](https://pawan-mittal.github.io/allassets.github.io/data-science/machine-learning-python/charts/module4/12.png)


#### Application to real-world dataset for classification
```
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes = [100, 100], alpha = 5.0,
                   random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)

print('Breast cancer dataset')
print('Accuracy of NN classifier on training set: {:.2f}'.format(clf.score(X_train_scaled, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'.format(clf.score(X_test_scaled, y_test)))
     
```

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


