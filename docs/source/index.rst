.. pspso documentation master file, created by
   sphinx-quickstart on Mon Mar 16 12:41:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
.. image:: ../../LOGO.png
    :align: center
    :alt: alternate text

Welcome to pspso's documentation!
=================================

Overview and Installation
=========================

Overview
------------  
**pspso** is a python library for selecting machine learning algorithms
parameters. The first version supports two single algorithms:
Multi-Layer Perceptron (MLP) and Support Vector Machine (SVM). It
supports two ensembles: Extreme Gradient Boosting (XGBoost) and Gradient
Boosting Decision Trees (GBDT).

Two types of machine learning tasks are supported by pspso:

* Regression.

* Binary classification.

Three scores are supported in the first version of pspso:

* **Regression** :

	* Root Mean Square Error (RMSE)
	
* **Binary Classication** :

	* Area under the Curve (AUC) of the Receiver Operating Characteristic (ROC)
	* Accuracy


Installation
------------

Use the package manager `pip <https://pip.pypa.io/en/stable/>`__ to
install pspso.

.. code:: bash

    pip install pspso

Usage
=================================

MLP Example (Binary Classification)
-----------------------------------

**pspso** is used to select the machine learning algorithms parameters. 
Below is an example for using the pspso to select
the parameters of the MLP. pspso handles the
MLP random weights intialization issue that may cause losing the best
solution in consecutive iterations.

The following example demonstrates the selection process of the MLP parameters.
A variable named *params* was not given by the user. Hence, the default search space of the MLP is loaded. 
This search space contains five parameters:

.. code:: python

	params = {"optimizer": ["RMSprop", "adam", "sgd",'adamax','nadam','adadelta'] ,
		  "learning_rate":  [0.1,0.3,2],
		  "neurons": [1,40,0],
		  "hiddenactivation": ['relu','sigmoid','tanh'],
		  "activation":['relu','sigmoid','tanh']} 
		  
The task and the score were defined as *binary classification* and *auc* respectively.
Then, the PSO was used to select the parameters of the MLP. 
Results are provided back to the user through the **print_results()** function. 


.. code:: python

	from sklearn.preprocessing import MinMaxScaler
	from pspso import pspso
	from sklearn import datasets
	from sklearn.model_selection import train_test_split

	breastcancer = datasets.load_breast_cancer()
	data=breastcancer.data#get the breast cancer dataset input features
	target=breastcancer.target# target
	X_train, X_test, Y_train, Y_test = train_test_split(data, target,test_size=0.1,random_state=42,stratify=target)
	normalize = MinMaxScaler(feature_range=(0,1))#normalize input features 
	X_train=normalize.fit_transform(X_train)
	X_test=normalize.transform(X_test)
	X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,test_size=0.15,random_state=42,stratify=Y_train)
	p=pspso(estimator='mlp',task='binary classification', score='auc')
	pos,cost,duration,model,optimizer=p.fitpspso(X_train,Y_train,X_val,Y_val)
	p.print_results()#print the results
	testscore=pspso.predict(p.model,p.estimator,p.task,p.score, X_test, Y_test)
	print(1-testscore)

In this example, four parameters were examined: optimizer,
learning_rate, hiddenactivation, and activation. The number of neurons
in the hidden layer was kept as default.

Output:

.. code:: python

	Estimator: mlp
	Task: binary classification
	Selection type: PSO
	Number of attempts:50
	Total number of combinations: 45360
	Parameters:
	{'optimizer': 'nadam', 'learning_rate': 0.29, 'neurons': 4, 'hiddenactivation': 'sigmoid', 'activation': 'sigmoid'}
	Global best position: [3.8997699  0.28725911 4.21218138 1.41200923 0.84643591]
	Global best cost: 0.0
	Time taken to find the set of parameters: 160.3374378681183
	Number of particles: 5
	Number of iterations: 10
	0.9867724867724867

XGBoost Example (Binary Classification)
---------------------------------------

.. code:: python

	from sklearn.preprocessing import MinMaxScaler
	from pspso import pspso
	from sklearn import datasets
	from sklearn.model_selection import train_test_split

	breastcancer = datasets.load_breast_cancer()
	data=breastcancer.data#get the breast cancer dataset input features
	target=breastcancer.target# target
	X_train, X_test, Y_train, Y_test = train_test_split(data, target,test_size=0.1,random_state=42,stratify=target)
	normalize = MinMaxScaler(feature_range=(0,1))#normalize input features 
	X_train=normalize.fit_transform(X_train)
	X_test=normalize.transform(X_test)
	X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,test_size=0.15,random_state=42,stratify=Y_train)

	params = {
			"learning_rate":  [0.01,0.2,2],
			"max_depth": [1,10,0],
			"n_estimators": [2,200,0],
			"subsample": [0.7,1,1]}
	p=pspso(estimator='xgboost',params=params,task='binary classification', score='auc')
	pos,cost,duration,model,optimizer=p.fitpspso(X_train,Y_train,X_val,Y_val)
	p.print_results()#print the results
	testscore=pspso.predict(p.model,p.estimator,p.task,p.score, X_test, Y_test)
	print(1-testscore)
	
XGBoost Example (Regression)
---------------------------------------

The XGBoost is an implementation of boosting decision trees. 
Five parameters were utilized for selection: objective, learning rate, maximum depth, number of estimators, and subsample.
Three categorical values were selected for the objective parameter. 
The learning rate parameter values range between *0.01* and *0.2* with *2* decimal point, 
maximum depth ranges between *1* and *10* with *0* decimal points *(1,2,3,4,5,6,7,8,9,10)*, etc. 
The task and score are selected as regression and RMSE respectively. 
The number of particles and number of iterations can be left as default values if needed.
Then, a pspso instance is created. By applying the fitpspso function, the selection process is applied. 
Finally, results are printed back to the user. 
The best model, best parameters, score, time, and other details will be saved in the created instance for the user to check.

.. code:: python

	from sklearn.preprocessing import MinMaxScaler
	from pspso import pspso
	from sklearn import datasets
	from sklearn.model_selection import train_test_split

	boston_data = datasets.load_boston()
	data=boston_data.data
	target=boston_data.target

	X_train, X_test, Y_train, Y_test = train_test_split(data, target,test_size=0.1,random_state=42)
	normalize = MinMaxScaler(feature_range=(0,1))#normalize input features
	normalizetarget = MinMaxScaler(feature_range=(0,1))#normalize target

	X_train=normalize.fit_transform(X_train)
	X_test=normalize.transform(X_test)
	Y_train=normalizetarget.fit_transform(Y_train.reshape(-1,1))
	Y_test=normalizetarget.transform(Y_test.reshape(-1,1))

	X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,test_size=0.25,random_state=42)
	params = {
			"objective":['reg:tweedie',"reg:linear","reg:gamma"],
			"learning_rate":  [0.01,0.2,2],
			"max_depth": [1,10,0],
			"n_estimators": [2,200,0],
			"subsample": [0.7,1,1]}
	p=pspso(estimator='xgboost',params=params,task='regression', score='rmse')
	pos,cost,duration,model,optimizer=p.fitpspso(X_train,Y_train,X_val,Y_val)
	p.print_results()#print the results
	testscore=pspso.predict(p.model,p.estimator,p.task,p.score, X_test, Y_test)
	print(testscore)

User Input
----------
The user is required to select the type of the algorithm ('mlp', 'svm', 'xgboost', 'gbdt'); the task type ('binary classification','regression'), score ('rmse', 'acc', or 'auc'). The user can keep the parameters variable empty, where a default set of parameters and ranges is loaded 
for each algorithm.

.. code:: python

    from pspso import pspso
    task='binary classification'
    score='auc'
    p=pspso.pspso('xgboost',None,task,score)
	

Pspso allows the user to provide a range of parameters for exploration.
The parameters vary between each algorithm. Any parameter supported by the Scikit-Learn API for GBDT and XGBoost can be added to the selection process. 
A set of parameters that contains five XGBoost parameters is shown below. The parameters are encoded in JSON object that consists of *key,value* pairs:

.. code:: python

	params = {"objective":['reg:tweedie',"reg:linear","reg:gamma"],
			"learning_rate":  [0.01,0.2,2],
			"max_depth": [1,10,0],
			"n_estimators": [2,200,0],
			"subsample": [0.7,1,1]}

The key can be any parameter belonging to to the algorithm under investigation.
The value is a list.
Pspso will check the type of the first element in the list, which will determine if the values of the parameter are categorical or numerical.

**Categorical Parameters**

If the parameter values are *categorical*, string values are expected to be found in the list, as shown in *objective* parameter. 
The values in the list will be automatically mapped into a list of integers, where each integer represents a value in the original list. 
The order of the values inside the list affect the position of the value in the search space.

**Numerical Parameters**

If the parameter is numerical, a list of three elements [lb,ub, rv] is expected to be found:

* **lb**: repesents the lowest value in the search space
* **ub**: represents the maximum value in the search space
* **rv**: represents the number of decimal points the parameter values are rounded to before being added for training the algorithm 

For e.g if you want pspso to select n_estimators, add the following list *[2,200,0]*.
By that, the lowest n_estimators will be *2*, the highest to be examined is *200*, and each possible value is rounded to an integer value ( *0* decimal points).



**Other parameters**


The user is given the chance to handle some of the default parameters
such as the number of epochs in the MLP. Although this parameter can be optimized, but its not encouraged. The user can modify this by changing a
pspso class instance. For e.g., to change the number of
epochs from default to 10 in MLP training:

.. code:: python

    from pspso import pspso
    task='binary classification'
    score='auc'
    p=pspso.pspso('mlp',None,task,score)# in case of empty set of params (None) default search space is loaded
    p.defaultparams['epochs']=10


The verbosity can be modified for any algorithm, which allows showing details of the training process:

.. code:: python

    from pspso import pspso
    task='binary classification'
    score='auc'
    p=pspso.pspso('mlp',None,task,score)
    p.verbosity=1

Early stopping rounds can alos be modified, the user can set a value different to the default value:

.. code:: python

    from pspso import pspso
    task='binary classification'
    score='auc'
    p=pspso.pspso('xgboost',None,task,score)
    p.early_stopping=10

Other parameters such that n_jobs in XGBoost can also be modified before the start of the selection process.  


Functions
==================

ML Algorithms Functions
-------------------------

.. currentmodule:: pspso.pspso

.. autosummary:: 

	forward_prop_gbdt
	forward_prop_xgboost
	forward_prop_svm
	forward_prop_mlp
	
Selection Functions
-------------------------

.. currentmodule:: pspso.pspso

.. autosummary:: 

	fitpspso
	fitpsgrid
	fitpsrandom
	
The fitpsrandom() and fitpsgrid() were implmented as two default selection methods. 
With fit random search, the number of attempts to be tried is added by the user as a variable. 
In grid search, all the possible combinations are created and investigated by the package. 
These functions follow the same encoding schema used in fitpspso(), and were basically added for comparison.

Parameters Functions
--------------------------------------

.. currentmodule:: pspso.pspso

.. autosummary:: 

	read_parameters
	decode_parameters
	get_default_params
	get_default_search_space
	
Other Functions
-------------------------

.. currentmodule:: pspso.pspso

.. autosummary:: 
	
	f
	rebuildmodel
	print_results
	calculatecombinations
	predict
	
	

  

Module Summary
==================

.. currentmodule:: pspso

.. autoclass:: pspso
    :members:


Future Work
===========


New Algorithms
----------------------------

Other machine learning algorithms and packages will be added such as the catboost.


Cross Validation
-------------------------
 
We are working towards adding the cross validation support that will take the training data and number of folds. 

Then split the records and train each fold. The average performance of cross-validation will be retuned back to the user.

Multi-Class Classification
---------------------------

We are also working on adding multi-class classification and data oversampling techniques.


Contributing
==================

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

We are working towards adding the cross validation support that will take the training data and number of folds, then split the records and train each fold. Finally, the average performance is retuned to the user.

We are also working on adding multi-class classification and data oversampling techniques.


License
==================

Copyright (c) [2020] [Ali Haidar]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.




