#Logistic Regression model

## Introduction
   The aim of this project is to come up with a logistic regression model for predicting customer churn in 
   the given dataset containing 14colums and 2000 entries,

### Binary Classification
    For binary classification, all the columns from the dataset should be in numeric for easy training by the logistic regression.
    This is achieved through end-hot encoding or by the dummies function
### Logistic Function

Previously we used the simple ```y=mx + b``` to guess our value but since classification isn't a linear problem we will instead have to use a different hypothesis function. 

![alt text](https://www.dropbox.com/s/u91yq42uegnz46x/logistic%20function.png?raw=1 "logistic function")

The benefit of using this function is that it gives us a 0 or 1 value given any input number and is thus better suited for classification.

![alt text](https://www.dropbox.com/s/z9pkvcyfd5o4epg/logistic%20curve.png?raw=1 "logistic curve")

The implementation in the code for this functions is:

```Python
def sigmoid(z):
    return  1 / (1 + np.exp(-z))
```

### Cost Function

Like with our hypothesis function, we cannot use a the same cost function as we use in linear regression as this would result in a wavy line with many local optima so gradient descent wouldn't find the global minimum value. To solve this, we have to use a modified cost function that gives us a convex function so we can find optimal values. The updated cost function:

![alt text](https://www.dropbox.com/s/gagmn7ocgpjblqk/cost%20function.png?raw=1 "cost function")

Or a vectorized version:

![alt text](https://www.dropbox.com/s/opojq3mnbtvgiov/vector%20cost%20function.png?raw=1 "vectorized cost function")

This is implemented in our code as:

```python
def cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
```

### Gradient Descent

To minimize our cost function we can use the exact same gradient descent algorithm used in linear regression.

```python
def gradient(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]

def logistic_regression(X, y, theta, alpha, iters):
