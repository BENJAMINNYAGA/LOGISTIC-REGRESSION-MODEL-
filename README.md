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
Simple steps to come up with the logistic regression model to train and prediction of a customer churning.
#Import libraries
Libraries such as numpy,pandas, matplotlib ana seaborne are the basic
Ones to interact with the dataset.
#Loading the dataset.
Using the read_csv function,the very famous one for from python for loading dataset into IDE 
is used.
#Data cleaning
This is where all outliers are do away with.
This is to ensure that data is cleaned and with no duplicates for easy training.
#Data preprocessing
This comes in to ensure that all data is converted into numeric form . 
##Importing the logistic regression model.
This is done by invoking the sklearn libraries to bring the model to the task.
#Data training and splitting.
Here the cleaned data is divided into training and testing data.
The model is the called to fit with the data for easy prediction of an instance.
#Conclusions.
  The logistic regression model performed Soo well such that it was able to predict a customer churning by an accuracy of 78.9%.
   I recommend the ude of logistic regression as it became the best with the prediction percentage accuracy 
 


