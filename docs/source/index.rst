.. pspso documentation master file, created by
   sphinx-quickstart on Mon Mar 16 12:41:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pspso's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   
pspso is a python library for selecting machine learning algorithms
parameters. The first version supports two single algorithms:
Multi-Layer Perceptron (MLP) and Support Vector Machine (SVM). It
supports two ensembles: Extreme Gradient Boosting (XGBoost) and Gradient
Boosting Decision Trees (GBDT).

Two types of machine learning tasks are supported by pspso:

* Regression.

* Binary classification.

Three scores can be used with the first version of pspso:

* **Regression** :

	* Root Mean Square Error (RMSE) for regression tasks
	
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
-----

MLP Example
~~~~~~~~~~~

pspso is used to select the machine learning algorithms parameters. It
is assumed that the user has already processed and prepared the training
and validation datasets. Below is an example for using the pso to select
the parameters of the MLP. It should be noticed that pspso handles the
MLP random weights intialization issue that may cause losing the best
solution in consecutive iterations.

.. code:: python

    from pspso import pspso
    params = {"optimizer":['adam','nadam','sgd','adadelta'],
        "learning_rate":  [0.01,0.2,2],
        "hiddenactivation": ['sigmoid','tanh','relu'],
        "activation": ['sigmoid','tanh','relu']}
    task='binary classification'
    score='auc'
    number_of_particles=4
    number_of_iterations=5
    p=pspso('mlp',params,task,score)
    p.fitpspso(X_train,Y_train,X_val,Y_val,number_of_particles=number_of_particles,
                   number_of_iterations=number_of_iterations)
    p.printresults()

In this example, four parameters were examined: optimizer,
learning\_rate, hiddenactivation, and activation. The number of neurons
in the hidden layer was kept as default.

XGBoost Example
~~~~~~~~~~~~~~~

Five parameters of the xgboost are searched and explored.

.. code:: python

	from pspso import pspso
	params = {
			"objective":['reg:tweedie',"reg:linear","reg:gamma"],
		"learning_rate":  [0.01,0.2,2],
		"max_depth": [1,10,0],
		"n_estimators": [2,200,0],
		"subsample": [0.7,1,1]}
	task="regression"
	score="rmse"
	number_of_particles=20
	number_of_iterations=40
	p=pspso('xgboost',params,task,score)
	p.fitpspso(X_train,Y_train,X_val,Y_val,
				   number_of_particles=number_of_particles,
				   number_of_iterations=number_of_iterations)
	print("PSO search:")
	p.printresults()

User Input
~~~~~~~~~~

Pspso allows the user to provide a range of parameters for exploration.
The parameters vary between each algorithm. 
For this current version, up to 5 paramaters can be explored at the same time. 
The user can provide an empty set of parameters. By that, a default search space is created.

**How are the parameters encoded ?**

The parameters are encoded in json object that consists of *key,value* pairs:

.. code:: python

	params = {
		"objective":['reg:tweedie',"reg:linear","reg:gamma"],
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

On the other side, if the parameter is numerical, a list with three elements is expected [lb,ub, rv]:

* **lb**: repesents the lowest value in the search space
* **ub**: represents the maximum value in the search space
* **rv**: represents the number of decimal points the parameter values are rounded to before being added for training the algorithm 

For e.g if you want pspso to select n_estimators, you add the following list [2,200,0] as in the example.
By that, the lowest n_estimators will be 2, the highest to be examined is 200, and each possible value is rounded to an integer value ( 0 decimal points).



Training details
~~~~~~~~~~~~~~~~~~

The user is given the chance to handle some of the default parameters
such as the number of epochs. The user can modify this by changing a
pspso class intance. For e.g., if you need to change the number of
epochs from 50 to 10 for an MLP training:

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

Early stopping rounds for supporting algorithm can be modified, default is 60:

.. code:: python

    from pspso import pspso
    task='binary classification'
    score='auc'
    p=pspso.pspso('xgboost',None,task,score)
    p.early_stopping=10


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
    

Summary
==================

.. currentmodule:: pspso

.. autoclass:: pspso
    :members:


Future Work
===========

Additional Parameters
---------------------------

To add new parameters to the currently supported algorithms, two functions should be updated

The **read_params** function should include default details about the parameter,
The **forward_prop_algorithmname** function should add the parameter to the initialization process

New Algorithms
----------------------------

Adding a new algorithm is more complex as you will be required to add an objective function that will detail the training and evaluation process.

New Optimizers
Two main optimizers are currently supported. These algorithms are built in the pyswams function. 

The default is globalbest pso, however the user can specify the local pso
The pso parameters are set to default in each case and can be modified by the user.

Crossvalidation
-------------------------
 
We are working towards adding the cross validation support that will take the training data and number of folds, then split the records and train each fold. Finally, the average performance is retuned to the user.

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
