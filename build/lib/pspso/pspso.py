# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 08:53:50 2020

@author: AliHaidar

This package allows using PSO for optimizing Machine Learning algorithms parameters. 
Four algorithms were initiated in the first study: MLP, SVM, XGBoost, GBDT
The class contains various static classes, to allow running seperate functions alone.
In some cases, i was forced to follow the hardest way since I didnt want to modify any part of the package
that supports pso (pwswarms)
"""
import random
import numpy as np
import time
import itertools as it
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC,SVR
#keras MLP
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,ReduceLROnPlateau
from keras.layers import Dense
import pyswarms as ps# Import PySwarms
from sklearn.metrics import mean_squared_error,accuracy_score,auc,roc_curve


class pspso:
    """
    This class searches for algorithm parameters by using the Particle Swarm Optimization (PSO) algorithm.
    """
    best_paricle_cost_ann =None
    best_model_ann=None
    best_history_ann=None
    best_particle_position_ann=None
       
    verbose=0
    early_stopping=20
    
    defaultparams= None #contains the default parameters of the algorithm that were not selected for optimization, or cant be selected for optimization
    parameters=None #contains the list of the parameters selected for optimization
    paramdetails= None #contains the dictionary given as input
    rounding=None # contains a list that determines to what extent the parameter will be loadad e.g. learning rate selected is 0.342 will be loaded to 0.34 if rounding value is 2 (two integers after the decimal point)
    
    def __init__(self, estimator='xgboost', params=None, task="regression",score= 'rmse'):
        """Construct an istance of the class pspso.
        
        Inputs
        ------
        estimator: a variable that can be 'xgboost', 'gbdt','mlp', or 'svm'. Default 'xgboost'
            The name of the estimators whose parameters to be optimized.
            
        params: a dictionary that determines the parameters to be optimized
        
        task: a variable (regression, binary classification, or binary classification -r)
            determines the type of the application
            
        score: the fitness evaluation score while selecting the hyper-parameters.
        """
        self.estimator = estimator # this variable is used to save the estimator, can be: xgboost, gbdt, mlp, or svm (cnn + catboost are not supported yet) 
        self.task=task # can be 'regression' or 'classification'
        self.score=score # score: can be 'rmse' for regression 'acc' or 'auc' for binary classification. Multi-class classification (not supported yet)
        self.cost=None
        self.pos=None
        self.model=None
        self.duration=None
        self.rmse=None
        self.optimizer=None
        pspso.parameters,pspso.defaultparams,self.x_min,self.x_max,pspso.rounding,self.bounds, self.dimensions,pspso.paramdetails=pspso.read_parameters(params,self.estimator,self.task)

    @staticmethod
    def get_default_search_space(estimator,task):
        """Create a dictionary of default parameters if the user didnt provide parameters.
        
        Inputs
        
        estimator: string value
            A string value that determines the estimator: 'mlp','xgboost','svm', or 'gbdt'
            
        task: string value
            A string value that determines the task under consideration: 'regression' or 'binary classification'
        
    
        Returns
        
        params: Dictionary
            A dictionary that contains default parameters to be used. 
            
        """
        if estimator == 'xgboost':
            if task == 'binary classification':
                params = {"learning_rate":  [0.1,0.3,2],
                  "max_depth": [1,10,0],
                  "n_estimators": [2,70,0],
                  "subsample": [0.7,1,2]}
            else:
                params = {"objective": ["reg:linear","reg:tweedie","reg:gamma"],
                  "learning_rate":  [0.1,0.3,2],
                  "max_depth": [1,10,0],
                  "n_estimators": [2,70,0],
                  "subsample": [0.7,1,2]}
        elif estimator == 'gbdt':
            if task == 'binary classification':
                params = {"learning_rate":  [0.1,0.3,2],
                  "max_depth": [1,10,0],
                  "n_estimators": [2,70,0],
                  "subsample": [0.7,1,2]}
            else:
                params = {"objective": ["tweedie","regression"],
                  "learning_rate":  [0.1,0.3,2],
                  "max_depth": [1,10,0],
                  "n_estimators": [2,70,0],
                  "subsample": [0.7,1,2]}
        elif estimator == 'svm':
            params = {"kernel": ["linear", "rbf", "poly"] ,
                  "gamma":  [0.1,10,1],
                  "C": [0.1,10,1],
                  "degree": [0,6,0]}
        elif estimator == 'mlp':
            params = {"optimizer": ["RMSprop", "adam", "sgd",'adamax','nadam','adadelta'] ,
                  "learning_rate":  [0.1,0.3,2],
                  "neurons": [1,40,0],
                  "hiddenactivation": ['relu','sigmoid','tanh'],
                  "activation":['relu','sigmoid','tanh']}   
        return params
    
    @staticmethod
    def get_default_params(estimator, task):
        """Set the default parameters of the estimator. 
        This function assigns the default parameters for the user.
        Each algorithm has a set of parameters. To allow the user to search for some parameters
        instead of the supported parameters, this function is used to assign a default value for each parameter. 
        In addition, it gets other parameters for each algorithm. For e.g, it returns the number of epochs, batch_size, and loss for the mlp. 
        
        Inputs
        
        estimator: string value
            A string value that determines the estimator: 'mlp','xgboost','svm', or 'gbdt'
            
        task: string value
            A string value that determines the task under consideration: 'regression' or 'binary classification'
        
    
        Returns
        
        defaultparams: Dictionary
            A dictionary that contains default parameters to be used.
        """
        
        defaultparams= {}
        if estimator == 'xgboost':
            defaultparams.update({'learning_rate':0.01,'max_depth':6,'n_estimators':40,'subsample':0.99})
            if task =='binary classification': # default activation
                defaultparams.update({'objective':'binary:logistic','eval_metric':["aucpr","auc"]})
            elif task =='regression':
                defaultparams.update({'objective':'reg:tweedie','eval_metric':["rmse"]})    
        elif estimator == 'gbdt':
            if task =='regression':
                defaultparams['objective'] = 'tweedie' 
                eval_metric ='rmse'
            elif task =='binary classification':
                defaultparams['objective'] = 'binary'
                eval_metric =['auc'] 
            defaultparams.update({'learning_rate':0.01,'max_depth':6,'n_estimators':40,'subsample':0.99,
                                  'boosting_type':'gbdt','eval_metric':eval_metric})
                
        elif estimator == 'mlp':
            #steps_per_epoch=4000 // batch_size
            defaultparams.update({'batch_size':12,'epochs':50,'shuffle':True,
                                  'neurons':13,'hiddenactivation':'sigmoid',
                                  'activation':'sigmoid','learning_rate':0.01,
                                  'mode':'auto'})#batchsize, epochs, and shuffling default values.
            if task =='binary classification': # set the optimizer based on the task
                defaultparams.update({'optimizer':'adam','metrics':['binary_accuracy'],'loss':'binary_crossentropy'})  
            elif task=='regression':
                defaultparams.update({'optimizer':'RMSprop','metrics':['mse'],'loss':'mse'}) 
        elif estimator == 'svm':
            defaultparams.update({'kernel':'rbf','C':5,'gamma':5})
        
        return defaultparams
        
    
    @staticmethod
    def read_parameters(params=None,estimator=None, task=None):
        """Read the parameters provided by the user.
        
        Inputs
        
        params: dictionary of key,values added by the user
            This dictionary determines the parameters and ranges of parameters the user wants to selection values from.
            
        estimator: string value
            A string value that determines the estimator: 'mlp','xgboost','svm', or 'gbdt'
            
        task: string value
            A string value that determines the task under consideration: 'regression' or 'binary classification'
        
    
        Returns
        
        parameters
            The parameters selected by the user
            
        defaultparams
            Default parameters
            
        x_min: list
            The lower bounds of the parameters search space
            
        x_max: list
            The upper bounds of the parameters search space
            
        rounding: list
            The rounding value in each dimension of the search space
        
        bounds: dict
            A dictionary of the lower and upper bounds
        
        dimensions: integer
            Dimensions of the search space
        
        params: Dict
            Dict given by the author
        
        """
        if params == None:
            params=pspso.get_default_search_space(estimator,task)
        x_min,x_max,rounding,parameters=[],[],[],[]
        for key in params:
            if all(isinstance(item, str) for item in params[key]):
                of=params[key]
                x_min.append(0)
                x_max.append(len(of)-1)
                parameters.append(key)
                rounding.append(0)
            else:
                thelist=params[key]
                x_min.append(thelist[0])
                x_max.append(thelist[1])
                parameters.append(key)
                rounding.append(thelist[2])
        bounds = (np.asarray(x_min), np.asarray(x_max))   
        dimensions=len(x_min)
        defaultparams=pspso.get_default_params(estimator, task)                              
        return parameters,defaultparams, x_min,x_max,rounding,bounds, dimensions,params
    
    @staticmethod
    def decode_parameters(particle):
        """Decodes the parameters of a list into a meaningful set of parameters.
        To decode a particle, we need the following global variables:parameters,
        defaultparameters, paramdetails, and rounding.
        """
        decodeddict={}
        # read each value in the particle
        for d in range(0,len(particle)):
            #get the key of the particle
            key=pspso.parameters[d]# expected to save parameter name, like objective, neurons, learning_rate, etc.
            #get the value of the particle
            particlevalueatd=particle[d]
            # if it is a numerical variable, i want to round it
            # if it is a categorical variable, i want to check its meaning
            # to do that, i can check the first value in the list
            if all(isinstance(item, str) for item in pspso.paramdetails[key]):
                #this means all the values are string, round the particlevalueatd and get the value
                index=int(round(particlevalueatd))
                decodeddict[key] = pspso.paramdetails[key][index]
            else:
                #get the rounding for the parameter
                decodeddict[key] =round(particlevalueatd,pspso.rounding[pspso.parameters.index(key)])
                if pspso.rounding[pspso.parameters.index(key)] == 0:
                    decodeddict[key]=int(decodeddict[key])#neurons, max_depth, estimators should be integers
        return decodeddict
        
         
    @staticmethod
    def forward_prop_gbdt(particle,task,score,X_train,Y_train,X_val,Y_val):
        #print(pspso.decode_parameters(particle))
        """Train the GBDT after decoding the parameters in variable particle.
        The particle is decoded into parameters of the gbdt. Then, The gbdt is trained and the score is sent back to the fitness function.
        
        Inputs
        
        particle: list of values (n dimensions)
            A particle in the swarm
        
        task: regression, binary classification
            the task to be conducted 
            
        score: rmse (regression), auc (binary classification), acc (binary classification)
            the type of evaluation
            
        X_train: numpy.ndarray of shape (m, n)
            Training dataset
            
        Y_train: numpy.ndarray of shape (m,1)
            Training target
            
        X_val: numpy.ndarray of shape (x, y)
            Validation dataset
            
        Y_val: numpy.ndarray of shape (x,1)
            Validation target
    
        Returns
       
        variable, model
            the score of the trained algorithm over the validation dataset, trained model

        """ 
        model=None 
        eval_set = [(X_val, np.squeeze(Y_val))]#eval set is the same in regression and classification
        try:
            decodedparams = pspso.decode_parameters(particle)
            modelparameters = {**pspso.defaultparams,**decodedparams}
            eval_metric=modelparameters['eval_metric']
            del modelparameters['eval_metric']
            if task !='binary classification':
                model = lgb.LGBMRegressor(**modelparameters)
            else : # if it is a binary classification task, will use XGBClassifier, note the different decoder since we have objective as fixed this time.
                model = lgb.LGBMClassifier(**modelparameters)

            model.fit(X_train,np.squeeze(Y_train),
                      early_stopping_rounds=pspso.early_stopping,
                      eval_set=eval_set,
                      eval_metric=eval_metric,
                      verbose=pspso.verbose )
            return pspso.predict(model,'gbdt',task, score,X_val,np.squeeze(Y_val)),model
        except Exception as e:
            print('An exception occured in GBDT training.')
            print(e)
            return None,None
    

        
    @staticmethod
    def forward_prop_xgboost(particle,task,score,X_train,Y_train,X_val,Y_val):
        """Train the XGBoost after decoding the parameters in variable particle.
        The particle is decoded into parameters of the XGBoost.
        This function is similar to forward_prop_gbdt
        The gbdt is trained and the score is sent back to the fitness function.
        
        Inputs
        
        particle: list of values (n dimensions)
            A particle in the swarm
        
        task: regression, binary classification
            the task to be conducted 
            
        score: rmse (regression), auc (binary classification), acc (binary classification)
            the type of evaluation
            
        X_train: numpy.ndarray of shape (m, n)
            Training dataset
            
        Y_train: numpy.ndarray of shape (m,1)
            Training target
            
        X_val: numpy.ndarray of shape (x, y)
            Validation dataset
            
        Y_val: numpy.ndarray of shape (x,1)
            Validation target
    
        Returns
        
        variable, model
            the score of the trained algorithm over the validation dataset, trained model

        """         
        model=None 
        eval_set = [(X_val, Y_val)]#eval set is the same in regression and classification
        try:
            decodedparams = pspso.decode_parameters(particle)
            modelparameters = {**pspso.defaultparams,**decodedparams}
            if task !='binary classification':
                model = xgb.XGBRegressor(**modelparameters)
            else : # if it is a binary classification task, will use XGBClassifier, note the different decoder since we have objective as fixed this time.
                model = xgb.XGBClassifier(**modelparameters)
            model.fit(X_train,Y_train,early_stopping_rounds=pspso.early_stopping,eval_set=eval_set,verbose=pspso.verbose )
            return pspso.predict(model,'xgboost',task, score,X_val,Y_val),model
        except Exception as e:
            print('An exception occured in XGBoost training.')
            print(e)
            return None,None
        
    @staticmethod
    def forward_prop_svm(particle,task,score,X_train,Y_train,X_val,Y_val):
      """Train the SVM after decoding the parameters in variable particle.
      
      """
      try:
          decodedparams = pspso.decode_parameters(particle)
          modelparameters = { **pspso.defaultparams,**decodedparams}
          if task == 'regression': # if it is a regression task, use SVR
              if modelparameters['kernel']!='poly': # the fourth parameter is only usable with kernel being polynomial : 'poly'
                  model = SVR(kernel=modelparameters['kernel'], C=modelparameters['C'],gamma=modelparameters['gamma']).fit(X_train, np.squeeze(Y_train))
              else:
                  model = SVR(kernel=modelparameters['kernel'], C=modelparameters['C'],gamma=modelparameters['gamma'],degree=modelparameters['degree']).fit(X_train, np.squeeze(Y_train))
          elif task == 'binary classification': # if it is a binary classification task, use SVC
              if modelparameters['kernel']!='poly':
                  model = SVC(kernel=modelparameters['kernel'], C=modelparameters['C'],gamma=modelparameters['gamma'],probability=True).fit(X_train, np.squeeze(Y_train))
              else:
                  model = SVC(kernel=modelparameters['kernel'], C=modelparameters['C'],gamma=modelparameters['gamma'],degree=modelparameters['degree'],probability=True).fit(X_train, np.squeeze(Y_train))
          return pspso.predict(model,'svm',task, score,X_val,Y_val),model
                         
      except Exception as e:
          print(e)
          print('An exception occured in SVM training.')
          return None,None
  

    @staticmethod
    def forward_prop_mlp(particle,task,score,X_train,Y_train,X_val,Y_val):
      """Train the MLP after the decoding the parameters in variable particle.
      
      """
      try:
          decodedparams = pspso.decode_parameters(particle)
          modelparameters = {**pspso.defaultparams,**decodedparams}              
          model=Sequential()
          model.add(Dense(int(modelparameters['neurons']), input_dim=X_train.shape[1], activation=modelparameters['hiddenactivation']))#particle,task='regression',score='rmse',X_train,Y_train,X_val,Y_val
          model.add(Dense(1, activation=modelparameters['activation']))#kernel_initializer='lecun_uniform',bias_initializer='zeros'
          model.compile(loss=modelparameters['loss'], optimizer=modelparameters['optimizer'], metrics=modelparameters['metrics'])
          model.optimizer.lr=modelparameters['learning_rate']
          #checkpoint=ModelCheckpoint('mlp.h5',monitor='val_loss',verbose=pspso.verbose,save_best_only=True,mode=mode)
          es = EarlyStopping(monitor='val_loss', mode=modelparameters['mode'], verbose=pspso.verbose,patience=pspso.early_stopping)
          #callbacks_list=[checkpoint,es]   
          callbacks_list=[es] 
          history=model.fit(X_train,
                            Y_train,
                            batch_size=modelparameters['batch_size'],
                            epochs=modelparameters['epochs'],
                            shuffle=modelparameters['shuffle'],
                            validation_data=(X_val,Y_val),
                            callbacks=callbacks_list,
                            verbose=pspso.verbose)
          #model.load_weights('mlp.h5')
          #model.compile(loss=loss, optimizer=modelparameters['optimizer'], metrics=metrics)
          return pspso.predict(model,'mlp',task, score,X_val,Y_val),model,history
            
      except Exception as e:
          print("An exception occured in MLP training.")
          print(e)
          return None,None
      

    @staticmethod
    def f(q,estimator,task,score,X_train,Y_train,X_val,Y_val):
        """Higher-level method to do forward_prop in the
        whole swarm.
    
        Inputs
        
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search
    
        Returns
        
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        
        n_particles = q.shape[0]
        if estimator=='xgboost':
            e = [pspso.forward_prop_xgboost(q[i],task,score,X_train,Y_train,X_val,Y_val) for i in range(n_particles)]# forward_prop varies based on each classifier
            j=[e[i][0] for i in range(n_particles)]
        elif estimator == 'gbdt':
            e = [pspso.forward_prop_gbdt(q[i],task,score,X_train,Y_train,X_val,Y_val) for i in range(n_particles)] 
            j=[e[i][0] for i in range(n_particles)]
        elif estimator == 'svm':
            e = [pspso.forward_prop_svm(q[i],task,score,X_train,Y_train,X_val,Y_val) for i in range(n_particles)] 
            j=[e[i][0] for i in range(n_particles)]
        elif estimator == 'mlp': # as mentioned in paper, the mlp should be treated differently
            e = [pspso.forward_prop_mlp(q[i],task,score,X_train,Y_train,X_val,Y_val) for i in range(n_particles)] 
            j=[e[i][0] for i in range(n_particles)]
            if pspso.best_particle_position_ann is not None:
                #if a position has been already assigned to this global variable, then it implies that this is not the first iteration.
                #if the same particle is found as the global best, check if it has better solution and swap, otherwise, keep the previous one and update the particle best
                for i in range(n_particles):
                    # if it is the same particle as the global best
                    if pspso.decode_parameters(q[i]) == pspso.decode_parameters(pspso.best_particle_position_ann):
                        #if error is higher than global best error
                        if j[i]>pspso.best_paricle_cost_ann:#same parameters, lower accuracy because of initial weights 
                            j[i]=pspso.best_paricle_cost_ann #assign the global best accuracy
                    #check if there is a better solution in the swarm.
                    if min(j) <=pspso.best_paricle_cost_ann:
                        min_loss_index= j.index(min(j)) # get the index of the minimum value
                        pspso.best_paricle_cost_ann=min(j)        
                        pspso.best_model_ann=e[min_loss_index][1] #get the best model at index 1
                        pspso.best_history_ann = e[min_loss_index][2]# get the history which is at index 2
                        pspso.best_particle_position_ann=q[min_loss_index]#get the best particle position from the list of particles
            else:
                # this case is for the first iteration where no position,cost,model, or history are assigned yet
                min_loss_index= j.index(min(j)) # get the index of the minimum value
                pspso.best_paricle_cost_ann=min(j)  # set the best cost to best_particle_cost_ann
                pspso.best_model_ann=e[min_loss_index][1] #set the best model to best_model_ann
                pspso.best_history_ann=e[min_loss_index][2] # set the best history in best_history_ann
                pspso.best_particle_position_ann=q[min_loss_index]#set the best position to best_particle_position_ann
        #return the score of each particle in the population.
        return np.array(j) 


    @staticmethod
    def rebuildmodel(estimator,pos,task,score,X_train,Y_train,X_val,Y_val):
      """Used to rebuild the model after selecting the parameters.
      
      """
      if estimator=='xgboost':
          met,model=pspso.forward_prop_xgboost(pos,task,score,X_train,Y_train,X_val,Y_val)
      elif estimator == 'gbdt':
          met,model=pspso.forward_prop_gbdt(pos,task,score,X_train,Y_train,X_val,Y_val)
      elif estimator == 'svm':
          met,model=pspso.forward_prop_svm(pos,task,score,X_train,Y_train,X_val,Y_val)
      elif estimator == 'mlp' :# again, if the case is mlp,1dcnn, or 2dcnn we will load the best solution found in global variables of the class
          return pspso.best_paricle_cost_ann,pspso.best_model_ann
      return met,model
  
    def fitpspso(self, X_train=None, Y_train=None, X_val=None,Y_val=None,psotype='global',number_of_particles=5, number_of_iterations=10, options = {'c1':  1.49618, 'c2':  1.49618, 'w': 0.7298}):
        """Select the algorithm parameters based on PSO.
        
        Inputs
        
        X_train: numpy.ndarray of shape (a,b) 
            Contains the training input features, a is the number of samples, b is the number of features
            
        Y_train: numpy.ndarray of shape (a,1) 
            Contains the training target, a is the number of samples
            
        X_train: numpy.ndarray of shape (c,b) 
            Contains the validation input features, c is the number of samples, b is the number of features
            
        Y_train: numpy.ndarray of shape (c,1) 
            Contains the training target, c is the number of samples
            
        number_of_particles: integer
            number of particles in the PSO search space.
            
        number_of_iterations: integer
            number of iterations.

        options: dictionary
            A key,value dict of PSO parameters c1,c2, and w
            
        Returns
        
        pos: list
           The encoded parameters of the best solution
       
        cost: float
            The score of the best solution
            
        duration: float
            The time taken to conduct random search.
            
        model: 
            The best model generated via random search
            
        combinations: list of lists
            The combinations examined during random search
            
        results: list
            The score of each combination in combinations list
        
        """
        print("Running PSO Search .....")
        self.selectiontype= "PSO" # selection type
        self.number_of_particles=number_of_particles # the number of particles of the PSO
        self.number_of_iterations=number_of_iterations # the number of iterations in the pso
        self.psotype=psotype
        self.options=options # parameters of the PSO
        self.number_of_attempts=self.number_of_iterations *self.number_of_particles # max number of attempts to find a solution
        self.totalnbofcombinations= len(self.calculatecombinations())
        pspso.best_paricle_cost_ann =None
        pspso.best_model_ann=None
        pspso.best_history_ann=None
        pspso.best_particle_position_ann=None
        
        kwargs = {"estimator":self.estimator, "task":self.task, "score":self.score, "X_train" : X_train, "Y_train" : Y_train, 
                  "X_val" : X_val,"Y_val":Y_val}
        if psotype =='global':
            self.optimizer = ps.single.GlobalBestPSO(n_particles=self.number_of_particles, dimensions=self.dimensions, options=self.options,bounds=self.bounds)
        elif psotype =='local':
            self.optimizer = ps.single.LocalBestPSO(n_particles=self.number_of_particles, dimensions=self.dimensions, options=self.options,bounds=self.bounds)
        start=time.time()
        #Perform optimization by using the optimize class
        self.cost, self.pos = self.optimizer.optimize(pspso.f, iters=self.number_of_iterations,**kwargs)
        end=time.time()
        self.duration=end-start          
        self.met,self.model=pspso.rebuildmodel(self.estimator,self.pos,self.task,self.score,X_train,Y_train,X_val,Y_val)
        
        if self.estimator =='mlp' :# if the estimator is mlp, assign history variable 
            self.history=pspso.best_history_ann
        self.miniopt=self.save_optimizer_details()
        return self.pos,self.cost,self.duration,self.model,self.optimizer
    
    
    def fitpsgrid(self,X_train=None, Y_train=None, X_val=None,Y_val=None ):
        """ Select the algorithm parameters based on Grid search.
        
        Grid search was implemented to match the training process with pspso and for comparison purposes.
        I have to traverse each value between x_min, x_max. Create a list seperating rounding value.
        
        
        """
        print("Running Grid Search .....")
        self.selectiontype= "Grid"
        self.results=[]
        self.model=None
        self.pos=None
        self.cost=None
        self.combinations=self.calculatecombinations()
        self.totalnbofcombinations=len(self.combinations)
        self.number_of_attempts=self.totalnbofcombinations
        
        start=time.time()        
        for comb in self.combinations:#for each value, run the function associated with the estimator
            #run the combination
            if self.estimator=='xgboost':
                met,mo=pspso.forward_prop_xgboost(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
            elif self.estimator == 'gbdt':
                met,mo=pspso.forward_prop_gbdt(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
            elif self.estimator == 'svm':
                met,mo=pspso.forward_prop_svm(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
            elif self.estimator == 'mlp' :
                met,mo,hist=pspso.forward_prop_mlp(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
                            
            self.results.append(met)#record results
            if self.cost == None: #starting
                self.cost=met
                self.pos=comb
                self.model= mo
            else:
                if met <self.cost: #everything is treated as a minimization problem
                    self.cost=met
                    self.pos=comb
                    self.model=mo
                    if self.estimator =='mlp':
                        self.history =hist
        end=time.time()
        self.duration=end-start   
        return self.pos,self.cost,self.duration,self.model,self.combinations,self.results #return pos, cost, duration, model, combinations, results
    
    
    def fitpsrandom(self,X_train=None, Y_train=None, X_val=None,Y_val=None,number_of_attempts=20 ):
        """Select the algorithm parameters based on radnom search.
        
        With Random search, the process is done for number of times specified by a parameter in the function.
        
        Inputs
        
        X_train: numpy.ndarray of shape (a,b) 
            Contains the training input features, a is the number of samples, b is the number of features
            
        Y_train: numpy.ndarray of shape (a,1) 
            Contains the training target, a is the number of samples
            
        X_train: numpy.ndarray of shape (c,b) 
            Contains the validation input features, c is the number of samples, b is the number of features
            
        Y_train: numpy.ndarray of shape (c,1) 
            Contains the training target, c is the number of samples
            
        number_of_attempts: integer
            The number of times random search to be tried.
            
        Returns
        
        pos: list
           The encoded parameters of the best solution
       
        cost: float
            The score of the best solution
            
        duration: float
            The time taken to conduct random search.
            
        model: 
            The best model generated via random search
            
        combinations: list of lists
            The combinations examined during random search
            
        results: list
            The score of each combination in combinations list
        
        """
        print("Running Random Search .....")
        self.number_of_attempts=number_of_attempts
        self.selectiontype= "Random"
        self.combinations=self.calculatecombinations()
        self.totalnbofcombinations=len(self.combinations)#check the number of combinations we have
        start=time.time()
        self.results=[]
        self.model=None
        self.pos=None
        self.cost=None
        for z in list(range(0,number_of_attempts)):
            #generate a random number between zero and totalnbofcombinations-1
            a=random.randint(0,self.totalnbofcombinations-1)
            comb=self.combinations[a]
            #run the combination
            if self.estimator=='xgboost':
                met,mo=pspso.forward_prop_xgboost(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
            elif self.estimator == 'gbdt':
                met,mo=pspso.forward_prop_gbdt(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
            elif self.estimator == 'svm':
                met,mo=pspso.forward_prop_svm(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
            elif self.estimator == 'mlp' :
                met,mo,hist=pspso.forward_prop_mlp(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
                            
            self.results.append(met)
            if self.cost == None: #starting
                self.cost=met
                self.pos=comb
                self.model= mo
            else:
                if met <self.cost: #everything is treated as a minimization problem
                    self.cost=met
                    self.pos=comb
                    self.model=mo
                    if self.estimator =='mlp':
                        self.history =hist
        
        end=time.time()
        self.duration=end-start   
        return self.pos,self.cost,self.duration,self.model,self.combinations,self.results
    
    def print_results(self):
        """Print the results found in the pspso instance. Expected to print general details
        like estimator, task, selection type, number of attempts examined, total number of 
        combinations, position of the best solution, score of the best solution, parameters,
        details about the pso algorithm.
        
        """
        print("Estimator: " + self.estimator)
        print("Task: "+ self.task)
        print("Selection type: "+ str(self.selectiontype))
        print("Number of attempts:" + str(self.number_of_attempts))
        print("Total number of combinations: " + str(self.totalnbofcombinations))
        print("Parameters:")
        print(pspso.decode_parameters(self.pos))
        print("Global best position: " + str(self.pos))
        print("Global best cost: " +str(round(self.cost,4)))
        print("Time taken to find the set of parameters: "+ str(self.duration))
        if self.selectiontype == "PSO":
            print("Number of particles: " +str(self.number_of_particles))
            print("Number of iterations: "+ str(self.number_of_iterations))
            
    def calculatecombinations(self):
        """A function that will generate all the possible combinations in the search space. 
        Used mainly with grid search
        
        Returns
        
        combinations: list
            A list that contains all the possible combinations.
        
        """
        index=0
        thedict={}
        #I have to traverse each value between x_min, x_max. Create a list seperating rounding value.
        for i,j in zip(self.x_min,self.x_max):
            a=np.arange(i, j+0.000001, 10**(-1*self.rounding[index]))
            a=np.round(a,self.rounding[index])  #create a array
            thedict[pspso.parameters[index]]=a #add the list as a value and value at index parameters to a dictionary. 
            index=index+1
        #now thedict contains all the possible values. we want to create all the possible combinations
        allparams = thedict#sorted(thedict)
        combinations = it.product(*(thedict[Name] for Name in allparams))# use the dictionary to create all possible solutions.
        #for each combination in combinations, we need to run the pspso
        combinations=list(combinations)
        combinations=[list(row) for row in combinations]# change the values to list instead of tuple
        return combinations
    
    @staticmethod
    def predict(model,estimator,task, score,X_val,Y_val):
        """A function used to release the score of a model. 
        If the score is rmse, the value is released. 
        If the score is acc (accuracy), 1-acc is returned back since pso applies a minimization task.
        If the score is auc, 1-auc is returned back since pso applies a minization task
        
        This class is static and can be used to test the model accuracy over the hold-out sample once the selection process is finalized. 
        
        Inputs
        
        model: 
            A trained model
                    
        estimator: string value
            A string value that determines the estimator: 'mlp','xgboost','svm', or 'gbdt'
            
        task: string value
            A string value that determines the task under consideration: 'regression' or 'binary classification'
        
        score: string value
            Determines the score ('rmse','auc','acc')
            
        X_val: numpy.ndarray
            Input features
            
        Y_val: numpy.ndarray
            Target
            
            
        Returns
        
        met: float
           Score value of the model

        """
        
        if score=='rmse':
            preds_val=model.predict(X_val)# predict output
            met = np.sqrt(mean_squared_error(Y_val, preds_val))
            return met
        if task =='binary classification' and estimator =='mlp':
            #gbdt and mlp has same way of prediction.
            # since it is using the gbdt model, output will be one column
            #and for the mlp, since sigmoid, tanh, or relu then one output. No softmax here.
            preds_val=model.predict(X_val)# predict output
            if score == 'acc':#since it is a minimization task, we return 1-acc
                met = accuracy_score(Y_val,np.round(preds_val))# need to round since gbdtModel is used not LGBMClassifier
                return 1-met
            elif score == 'auc':
                fpr, tpr, thresholds = roc_curve(Y_val, preds_val)
                met = auc(fpr, tpr)
                return 1-met   
        elif task =='binary classification' and (estimator== 'xgboost' or estimator =='svm' or estimator =='gbdt'):
            # XGBOOST classifier and svm has same way of prediction.
            if score == 'acc': # if it is accuracy
                preds_val = model.predict(X_val)  # one column with acc, will generate labels with function predict() in xgboost
                met = accuracy_score(Y_val,preds_val) # measure accuracy
                return 1-met
            elif score == 'auc': # if the score is area under the curve of the recevier operating characteristic
                preds_val = model.predict_proba(X_val)# predict_proba() will return two columns representing each class probability
                fpr, tpr, thresholds = roc_curve(Y_val, preds_val[:,1]) 
                met = auc(fpr, tpr)# calculate auc using roc_curve and auc in sklearn.metrics class
                return 1-met
            
    def save_optimizer_details(self):#since the optimizer cant be save (pyswarms issue)
        opt={}
        opt['pos_history']=self.optimizer.pos_history
        opt['cost_history']=self.optimizer.cost_history
        
        opt['bounds']=self.optimizer.bounds
        opt['init_pos']=self.optimizer.init_pos
        opt['swarm_size']=self.optimizer.swarm_size
        opt['options']=self.optimizer.options
        opt['name']=self.optimizer.name
        opt['n_particles']=self.optimizer.n_particles
        opt['cost_history']=self.optimizer.cost_history
        return opt

        
        
        
    

    
    
