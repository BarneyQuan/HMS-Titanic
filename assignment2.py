# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 17:11:07 2022

@author: 85384
"""
#%%

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.discrete.discrete_model import Probit
#%%
    
# import dataset
training = pd.read_csv(r"C:\\Users\85384\Desktop\Econ 2824\HW2\assignment2_train.csv")
test = pd.read_csv(r"C:\\Users\85384\Desktop\Econ 2824\HW2\assignment2_test.csv")

#explore the data a little bit by checking the number of rows and columns in our datasets
#and to see the statistical details of the dataset
training.shape
test.shape
training.describe()
test.describe()

# Remove useless columns
train = training.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Embarked'], axis=1)

# Filling empty numerical values with median
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
# Filling empty embarked with S

train['Sex'] = encoder.fit_transform(train['Sex'])  
train.head()

# Remove useless columns
test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# Filling empty numerical values with median
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

# Replacing categorical sex with integer values (0 for F and 1 for M)
test['Sex'] = encoder.fit_transform(test['Sex'])  
test.head()

# feature selection
xtrain = train[['Pclass','Sex','Age','SibSp','Parch','Fare']] # Features
ytrain = train['Survived'] # Target variable

xtest = test[['Pclass','Sex','Age','SibSp','Parch','Fare']] # Features
ytest = test['Survived'] # Target variable

# define our basic tree classifier
tree = DecisionTreeClassifier(random_state=0)

# fit it to the training data
tree.fit(xtrain, ytrain)
#tree.fit(xtest, ytest)

# compute accuracy in the test data
print("Accuracy on test set: {:.3f}".format(tree.score(xtest, ytest)))
#%%

# plot the tree
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["Died", "Survived"], impurity=True, filled=True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# apply cost complexity pruning

# call the cost complexity command
path = tree.cost_complexity_pruning_path(xtrain, ytrain)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# for each alpha, estimate the tree
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(xtrain, ytrain)
    clfs.append(clf)


# plot accuracy (in test and training) over alpha; first compute accuracy for each alpha
train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(xtrain)
    y_test_pred = c.predict(xtest)
    train_acc.append(accuracy_score(y_train_pred,ytrain))
    test_acc.append(accuracy_score(y_test_pred,ytest))

# second, plot it
plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy vs alpha')
plt.show()


# estimate the tree with the optimal alpha and display accuracy
clf_ = DecisionTreeClassifier(random_state=0,ccp_alpha=0.01)
clf_.fit(xtrain,ytrain)

print("Accuracy on test set: {:.3f}".format(clf_.score(xtest, ytest)))

# plot the pruned tree
export_graphviz(clf_, out_file="tree.dot", class_names=["Died", "Survived"],
     impurity=True, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))
#%%

from sklearn.linear_model import LinearRegression
from sklearn import metrics
regressor = LinearRegression()
#training the algorithm
regressor.fit(xtrain, ytrain)
#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
print(regressor.coef_)
y_pred = regressor.predict(xtest)
test = pd.DataFrame({'Actual': ytest.value.flatten(), 'Predicted': y_pred.value.flatten()})
test
plt.scatter(xtest, ytest,  color='gray')
plt.plot(xtest, y_pred, color='red', linewidth=2)
plt.show()

# the simple probit classifier
model = Probit(ytrain, xtrain).fit()
print(model.summary())
probit_predict = round(model.predict(xtest), 0)
print("Probit's accuracy on test set: {:.3f}".format(accuracy_score(ytest, probit_predict)))

