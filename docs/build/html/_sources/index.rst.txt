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


Details
~~~~~~~

The user is given the chance to handle some of the default parameters
such as the number of epochs. The user can modify this by changing a
pspso class intance. For e.g., if you need to change the number of
epochs from 50 to 10 for an MLP training:

.. code:: python

    from pspso import pspso
    task='binary classification'
    score='auc'
    p=pspso.pspso('mlp',params,task,score)
    p.defaultparams['epochs']=10

Contributing
==================

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
