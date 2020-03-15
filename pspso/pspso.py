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
import lightgbm
import xgboost as xgb
from sklearn.svm import SVC,SVR
#keras MLP
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,ReduceLROnPlateau
from keras.layers import Dense
import pyswarms as ps# Import PySwarms
from sklearn.metrics import mean_squared_error,accuracy_score,auc,roc_curve


class psosearch:
    """
    This class searches for algorithm parameters by using the Particle Swarm Optimization (PSO) algorithm.
    """
    best_paricle_cost_ann =None
    best_model_ann=None
    best_history_ann=None
    best_particle_position_ann=None
       
    verbose=0
    early_stopping=60
    
    defaultparams= None #contains the default parameters of the algorithm that were not selected for optimization, or cant be selected for optimization
    parameters=None #contains the list of the parameters selected for optimization
    paramdetails= None #contains the dictionary given as input
    rounding=None # contains a list that determines to what extent the parameter will be loadad e.g. learning rate selected is 0.342 will be loaded to 0.34 if rounding value is 2 (two integers after the decimal point)
    
    def __init__(self, estimator='xgboost', params=None, task="regression",score= 'rmse'):
        """Construct an istance of the class psosearch.
        
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
        psosearch.parameters,psosearch.defaultparams,self.x_min,self.x_max,psosearch.rounding,self.bounds, self.dimensions,psosearch.paramdetails=psosearch.readparameters(params,self.estimator,self.task)


    @staticmethod
    def decode_parameters(particle):
        """Decodes the parameters of a list into a meaningful set of parameters.
        To decode a particle, we need the following global variables: 
            
            global variable parameters
            global variable defaultparameters
            global variable paramdetails
            global variable rounding
        """
        decodeddict={}
        # read each value in the particle
        for d in range(0,len(particle)):
            #get the key of the particle
            key=psosearch.parameters[d]# expected to save parameter name, like objective, neurons, learning_rate, etc.
            #get the value of the particle
            particlevalueatd=particle[d]
            # if it is a numerical variable, i want to round it
            # if it is a categorical variable, i want to check its meaning
            # to do that, i can check the first value in the list
            if all(isinstance(item, str) for item in psosearch.paramdetails[key]):
                #this means all the values are string, round the particlevalueatd and get the value
                index=int(round(particlevalueatd))
                decodeddict[key] = psosearch.paramdetails[key][index]
            else:
                #get the rounding for the parameter
                decodeddict[key] =round(particlevalueatd,psosearch.rounding[psosearch.parameters.index(key)])
                if psosearch.rounding[psosearch.parameters.index(key)] ==0:
                    decodeddict[key]=int(decodeddict[key])#neurons, max_depth, estimators should be integers
        return decodeddict
        
         
    @staticmethod
    def forward_prop_gbdt(particle,task,score,X_train,Y_train,X_val,Y_val):
        """Calculates the fitness value of the encoded parameters in variable particle.
        The particle is decoded into parameters of the gbdt. Then, The gbdt is trained and the score is sent back to the fitness function.
        
        Inputs
        ------
        particle: list of values (n dimensions)
            A particle in the swarm
        
        task: regression, binary classification, or binary classification r
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
        -------
        variable, model
            the score of the trained algorithm over the validation dataset, trained model

        """ 
        decodedparams = psosearch.decode_parameters(particle)
        modelparameters = {**psosearch.defaultparams,**decodedparams}
        try:
            if task =='regression':# if it is a regression task
                eval_metric ='rmse'
            elif task =='binary classification':
                eval_metric ='auc' 
            modelparameters['boosting_type'] = 'gbdt'
            modelparameters['metric'] = eval_metric
            modelparameters['verbose'] = psosearch.verbose
            if modelparameters['subsample'] ==1: # Note: to enable bagging, bagging_fraction(subsample_freq) should be set to value smaller than 1.0 as well
                modelparameters['subsample_freq'] =0 #disable bagging
            else:
                modelparameters['subsample_freq'] =1 #perform bagging each iteration 
            
            train_data = lightgbm.Dataset(X_train, label=np.squeeze(Y_train))
            val_data = lightgbm.Dataset(X_val, label=np.squeeze(Y_val))#evaluation set.
            gbm_n_estimators=modelparameters['n_estimators']
            del modelparameters['n_estimators']
            gbm = lightgbm.train(modelparameters,
                                 train_data,
                                 valid_sets=val_data,
                                 num_boost_round=gbm_n_estimators,verbose_eval=False,
                                 early_stopping_rounds=psosearch.early_stopping)
            preds_val=gbm.predict(X_val)# predict output
            if score =='rmse':  # if the score is rmse, calculate it and return it with the model
                met = np.sqrt(mean_squared_error(Y_val, preds_val))
                return met,gbm
            if score == 'acc':#since it is a minimization task, we return 1-acc
                preds_val = gbm.predict(X_val) # since it is using the gbdt model, output will be one columns
                met = accuracy_score(Y_val,np.round(preds_val))# need to round since gbdtModel is used not LGBMClassifier
                return 1-met,gbm
            elif score == 'auc':
                preds_val = gbm.predict(X_val)#generate the probabilities 
                fpr, tpr, thresholds = roc_curve(Y_val, preds_val)
                met = auc(fpr, tpr)
                return 1-met,gbm
        except Exception as e:
            print('Got an exception in training gbdt')
            print(e)
            return None,None
    

        
    @staticmethod
    def forward_prop_xgboost(particle,task,score,X_train,Y_train,X_val,Y_val):
        """This function accepts the particle from the PSO fitness function.
        The particle is decoded into parameters of the XGBoost.
        This function is similar to forward_prop_gbdt
        The gbdt is trained and the score is sent back to the fitness function.
        
        Inputs
        ------
        particle: list of values (n dimensions)
            A particle in the swarm
        
        task: regression, binary classification, or binary classification r
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
        -------
        variable, model
            the score of the trained algorithm over the validation dataset, trained model

        """         
        xgb_model=None
        eval_set = [(X_val, Y_val)]#eval set is the same in regression and classification
        try:
            if task =='regression':# if it is a regression task, will use the XGBRegressor
                eval_metric=["rmse"]
            elif task =='binary classification' or task == 'binary classification r':
                eval_metric=["aucpr","auc"]#since it is a classification task, we used auc. its always better to use auc
                
            decodedparams = psosearch.decode_parameters(particle)
            modelparameters = {**psosearch.defaultparams,**decodedparams}
            #print(modelparameters)
            #create the model
            if task !='binary classification':
                xgb_model = xgb.XGBRegressor(objective =modelparameters['objective'],  
                              learning_rate = modelparameters['learning_rate'],                        
                              max_depth = int(modelparameters['max_depth']) ,
                              n_estimators = int(modelparameters['n_estimators']),
                              subsample=modelparameters['subsample'])
            else : # if it is a binary classification task, will use XGBClassifier, note the different decoder since we have objective as fixed this time.
                xgb_model = xgb.XGBClassifier(objective =modelparameters['objective'],  
                              learning_rate = modelparameters['learning_rate'],                        
                              max_depth = int(modelparameters['max_depth']) ,
                              n_estimators = int(modelparameters['n_estimators']),
                              subsample=modelparameters['subsample'])
            
            xgb_model.fit(X_train,Y_train,early_stopping_rounds=psosearch.early_stopping,eval_set=eval_set,eval_metric=eval_metric,verbose=psosearch.verbose )
            
            if score =='rmse':# based on the score, measure the fitness of the solution
                preds_val = xgb_model.predict(X_val)# predict output in xgboost regression
                met = np.sqrt(mean_squared_error(Y_val, preds_val)) # calculate the rmse 
                return met,xgb_model # return rmse plus model
            elif score == 'acc' and task =='binary classification': # if it is accuracy
                preds_val = xgb_model.predict(X_val)  # one column with acc, will generate labels with function predict() in xgboost
                met = accuracy_score(Y_val,preds_val) # measure accuracy
                return 1-met,xgb_model # return accuracy with the model created.
            elif score == 'auc' and task =='binary classification': # if the score is area under the curve of the recevier operating characteristic
                preds_val = xgb_model.predict_proba(X_val)# predict_proba() will return two columns representing each class probability
                fpr, tpr, thresholds = roc_curve(Y_val, preds_val[:,1]) 
                met = auc(fpr, tpr)# calculate auc using roc_curve and auc in sklearn.metrics class
                return 1-met,xgb_model # return auc plus the model created. 
            elif score == 'acc' and task =='binary classification r': #calculate accuracy
                preds_val = xgb_model.predict(X_val)# expected one column as it is a refression explicitly
                met = accuracy_score(Y_val,np.round(preds_val))# I have to round the output to obtain labels. 
                return 1-met,xgb_model
            elif score =='auc' and task =='binary classification r':
                preds_val = xgb_model.predict(X_val)# expected one column as it is a refression explicitly
                fpr, tpr, thresholds = roc_curve(Y_val, preds_val) # outputs are actually probabilities
                met = auc(fpr, tpr)
                return 1-met,xgb_model
        except Exception as e:
            print('An exception occured in XGBoost training.')
            print(e)
            return None,None
        
    @staticmethod
    def forward_prop_svm(particle,task,score,X_train,Y_train,X_val,Y_val):
      """Train the SVM after decoding the parameters in variable particle.
      
      """
      print("SVM")
      try:
          decodedparams = psosearch.decode_parameters(particle)
          modelparameters = { **psosearch.defaultparams,**decodedparams}
          
          if task == 'regression': # if it is a regression task, use SVR
              if modelparameters['kernel']!='poly': # the fourth parameter is only usable with kernel being polynomial : 'poly'
                  sv = SVR(kernel=modelparameters['kernel'], C=modelparameters['C'],gamma=modelparameters['gamma']).fit(X_train, np.squeeze(Y_train))
              else:
                  sv = SVR(kernel=modelparameters['kernel'], C=modelparameters['C'],gamma=modelparameters['gamma'],degree=modelparameters['degree']).fit(X_train, np.squeeze(Y_train))
              preds_val = sv.predict(X_val)
              if score == 'rmse': # if score is rmse
                  met = np.sqrt(mean_squared_error(np.squeeze(Y_val), preds_val)) # calculate rmse
                  return met,sv # return rmse with the generated svm model
          elif task == 'binary classification': # if it is a binary classification task, use SVC
              if modelparameters['kernel']!='poly':
                  sv = SVC(kernel=modelparameters['kernel'], C=modelparameters['C'],gamma=modelparameters['gamma'],probability=True).fit(X_train, np.squeeze(Y_train))
              else:
                  sv = SVC(kernel=modelparameters['kernel'], C=modelparameters['C'],gamma=modelparameters['gamma'],degree=modelparameters['degree'],probability=True).fit(X_train, np.squeeze(Y_train))
              
              if score == 'acc':
                  preds_val = sv.predict(X_val)# expecting labels column
                  met = accuracy_score(Y_val,preds_val)# measure accuracy 
                  return 1-met,sv # retun accuracy with the generated svm model
              elif score == 'auc':
                  preds_val = sv.predict_proba(X_val)#generate the probabilities using the function predict_proba()
                  fpr, tpr, thresholds = roc_curve(Y_val, preds_val[:,1]) #apply to get the prob of class 1
                  met = auc(fpr, tpr)
                  return 1-met,sv # return the auc with the generated model
                         
      except Exception as e:
          print(e)
          print('An exception occured in SVM training.')
          return None,None
  

    @staticmethod
    def forward_prop_mlp(particle,task,score,X_train,Y_train,X_val,Y_val):
      """Train the MLP after the decoding the parameters in variable particle.
      
      """
      try:
          decodedparams = psosearch.decode_parameters(particle)
          modelparameters = {**psosearch.defaultparams,**decodedparams}
          if task == 'regression':
              loss='mse'
              metrics=['mse','mae']
              mode= 'auto'
          elif task == 'binary classification':
              loss = 'binary_crossentropy'
              metrics=['binary_accuracy']
              mode= 'auto'
          elif task == 'binary classification r':
              loss = 'mse'
              metrics=['mse']
              mode= 'auto'
              
          model=Sequential()
          model.add(Dense(int(modelparameters['neurons']), input_dim=X_train.shape[1], activation=modelparameters['hiddenactivation']))#particle,task='regression',score='rmse',X_train,Y_train,X_val,Y_val
          model.add(Dense(1, activation=modelparameters['activation']))#kernel_initializer='lecun_uniform',bias_initializer='zeros'
          model.compile(loss=loss, optimizer=modelparameters['optimizer'], metrics=metrics)
          model.optimizer.lr=modelparameters['learning_rate']
          #print(model.optimizer.lr)
          #checkpoint=ModelCheckpoint('mlp.h5',monitor='val_loss',verbose=psosearch.verbose,save_best_only=True,mode=mode)
          es = EarlyStopping(monitor='val_loss', mode=mode, verbose=psosearch.verbose,patience=psosearch.early_stopping)
          #callbacks_list=[checkpoint,es]   
          callbacks_list=[es] 
          history=model.fit(X_train,
                            Y_train,
                            batch_size=modelparameters['batch_size'],
                            epochs=modelparameters['epochs'],
                            shuffle=modelparameters['shuffle'],
                            validation_data=(X_val,Y_val),
                            callbacks=callbacks_list,
                            verbose=psosearch.verbose)
          #model.load_weights('C:/Users/AliHaidar/Desktop/python/ML_PSO/mlp.h5')
          #model.compile(loss=loss, optimizer=modelparameters['optimizer'], metrics=metrics)
          if task == 'regression':
              preds_val = model.predict(X_val)
              if score == 'rmse':
                  met = np.sqrt(mean_squared_error(Y_val, preds_val))
                  return met,model,history
          elif task == 'binary classification' or task == 'binary classification r':
              preds_val = model.predict(X_val)# predict output,one value expected since we have one neuron in last layer
              if score == 'acc':                  
                  met = accuracy_score(Y_val,np.round(preds_val))
              elif score == 'auc':
                  fpr, tpr, thresholds = roc_curve(Y_val, preds_val)
                  met = auc(fpr, tpr)
              return 1-met,model,history              
      except Exception as e:
          print("An exception occured in MLP training.")
          print(e)
          return None,None
      

    @staticmethod
    def f(q,estimator,task,score,X_train,Y_train,X_val,Y_val):
        """Higher-level method to do forward_prop in the
        whole swarm.
    
        Inputs
        ------
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search
    
        Returns
        -------
        numpy.ndarray of shape (n_particles, )
            The computed loss for each particle
        """
        
        n_particles = q.shape[0]
        if estimator=='xgboost':
            e = [psosearch.forward_prop_xgboost(q[i],task,score,X_train,Y_train,X_val,Y_val) for i in range(n_particles)]# forward_prop varies based on each classifier
            j=[e[i][0] for i in range(n_particles)]
        elif estimator == 'gbdt':
            e = [psosearch.forward_prop_gbdt(q[i],task,score,X_train,Y_train,X_val,Y_val) for i in range(n_particles)] 
            j=[e[i][0] for i in range(n_particles)]
        elif estimator == 'svm':
            e = [psosearch.forward_prop_svm(q[i],task,score,X_train,Y_train,X_val,Y_val) for i in range(n_particles)] 
            j=[e[i][0] for i in range(n_particles)]
        elif estimator == 'mlp': # as mentioned in paper, the mlp should be treated differently
            e = [psosearch.forward_prop_mlp(q[i],task,score,X_train,Y_train,X_val,Y_val) for i in range(n_particles)] 
            j=[e[i][0] for i in range(n_particles)]
            if psosearch.best_particle_position_ann is not None:
                #if a position has been already assigned to this global variable, then it implies that this is not the first iteration.
                #if the same particle is found as the global best, check if it has better solution and swap, otherwise, keep the previous one and update the particle best
                for i in range(n_particles):
                    # if it is the same particle as the global best
                    if psosearch.decode_parameters(q[i]) == psosearch.decode_parameters(psosearch.best_particle_position_ann):
                        #if error is higher than global best error
                        if j[i]>psosearch.best_paricle_cost_ann:#same parameters, lower accuracy because of initial weights 
                            j[i]=psosearch.best_paricle_cost_ann #assign the global best accuracy
                    #check if there is a better solution in the swarm.
                    if min(j) <=psosearch.best_paricle_cost_ann:
                        min_loss_index= j.index(min(j)) # get the index of the minimum value
                        psosearch.best_paricle_cost_ann=min(j)        
                        psosearch.best_model_ann=e[min_loss_index][1] #get the best model at index 1
                        psosearch.best_history_ann = e[min_loss_index][2]# get the history which is at index 2
                        psosearch.best_particle_position_ann=q[min_loss_index]#get the best particle position from the list of particles
            else:
                # this case is for the first iteration where no position,cost,model, or history are assigned yet
                min_loss_index= j.index(min(j)) # get the index of the minimum value
                psosearch.best_paricle_cost_ann=min(j)  # set the best cost to best_particle_cost_ann
                psosearch.best_model_ann=e[min_loss_index][1] #set the best model to best_model_ann
                psosearch.best_history_ann=e[min_loss_index][2] # set the best history in best_history_ann
                psosearch.best_particle_position_ann=q[min_loss_index]#set the best position to best_particle_position_ann
        #return the score of each particle in the population.
        return np.array(j) 


    @staticmethod
    def rebuildmodel(estimator,pos,task,score,X_train,Y_train,X_val,Y_val):
      """Used to rebuild the model after selecting the parameters. 
      """
      if estimator=='xgboost':
          met,model=psosearch.forward_prop_xgboost(pos,task,score,X_train,Y_train,X_val,Y_val)
      elif estimator == 'gbdt':
          met,model=psosearch.forward_prop_gbdt(pos,task,score,X_train,Y_train,X_val,Y_val)
      elif estimator == 'svm':
          met,model=psosearch.forward_prop_svm(pos,task,score,X_train,Y_train,X_val,Y_val)
      elif estimator == 'mlp' :# again, if the case is mlp,1dcnn, or 2dcnn we will load the best solution found in global variables of the class
          return psosearch.best_paricle_cost_ann,psosearch.best_model_ann
      return met,model
  
    def fitpsosearch(self, X_train=None, Y_train=None, X_val=None,Y_val=None,number_of_particles=2, number_of_iterations=2, options = {'c1': 0.5, 'c2': 0.3, 'w': 0.4}):
        print("Running PSO Search .....")
        self.selectiontype= "PSO" # selection type
        self.number_of_particles=number_of_particles # the number of particles of the PSO
        self.number_of_iterations=number_of_iterations # the number of iterations in the pso
        self.options=options # parameters of the PSO
        self.number_of_attempts=self.number_of_iterations *self.number_of_particles # max number of attempts to find a solution
        self.totalnbofcombinations= len(self.calculatecombinations())
        psosearch.best_paricle_cost_ann =None
        psosearch.best_model_ann=None
        psosearch.best_history_ann=None
        psosearch.best_particle_position_ann=None
        
        kwargs = {"estimator":self.estimator, "task":self.task, "score":self.score, "X_train" : X_train, "Y_train" : Y_train, 
                  "X_val" : X_val,"Y_val":Y_val}
        self.optimizer = ps.single.GlobalBestPSO(n_particles=self.number_of_particles, dimensions=self.dimensions, options=self.options,bounds=self.bounds)
        start=time.time()
        #Perform optimization by using the optimize class
        self.cost, self.pos = self.optimizer.optimize(psosearch.f, iters=self.number_of_iterations,**kwargs)
        end=time.time()
        self.duration=end-start          
        self.met,self.model=psosearch.rebuildmodel(self.estimator,self.pos,self.task,self.score,X_train,Y_train,X_val,Y_val)
        
        if self.estimator =='mlp' :# if the estimator is mlp, assign history variable 
            self.history=psosearch.best_history_ann
        
        return self.pos,self.cost,self.duration,self.model,self.optimizer
    
    
    def fitgridsearch(self,X_train=None, Y_train=None, X_val=None,Y_val=None ):
        """ Grid search was implemented to match the training process with psosearch and for comparison purposes.
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
                met,mo=psosearch.forward_prop_xgboost(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
            elif self.estimator == 'gbdt':
                met,mo=psosearch.forward_prop_gbdt(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
            elif self.estimator == 'svm':
                met,mo=psosearch.forward_prop_svm(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
            elif self.estimator == 'mlp' :
                met,mo,hist=psosearch.forward_prop_mlp(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
                            
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
    
    
    def fitrandomsearch(self,X_train=None, Y_train=None, X_val=None,Y_val=None,number_of_attempts=20 ):
        """With Random search, the process is done for number of times specified by a parameter in the function.
        
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
                met,mo=psosearch.forward_prop_xgboost(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
            elif self.estimator == 'gbdt':
                met,mo=psosearch.forward_prop_gbdt(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
            elif self.estimator == 'svm':
                met,mo=psosearch.forward_prop_svm(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
            elif self.estimator == 'mlp' :
                met,mo,hist=psosearch.forward_prop_mlp(comb,self.task,self.score,X_train,Y_train,X_val,Y_val)
                            
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
    
    def printresults(self):
        print("Estimator: " + self.estimator)
        print("Task: "+ self.task)
        print("Selection type: "+ str(self.selectiontype))
        print("Number of attempts:" + str(self.number_of_attempts))
        print("Maximum number of combinations: " + str(self.totalnbofcombinations))
        print("Global best position: " + str(self.pos))
        print("Global best cost: " +str(round(self.cost,4)))
        print("Parameters:")
        print(psosearch.decode_parameters(self.pos))
        print("Time taken to find the set of parameters: "+ str(self.duration))
        if self.selectiontype == "PSO":
            print("Number of particles: " +str(self.number_of_particles))
            print("Number of iterations: "+ str(self.number_of_iterations))
            
        
        
        
    def calculatecombinations(self):
        index=0
        thedict={}
        #I have to traverse each value between x_min, x_max. Create a list seperating rounding value.
        for i,j in zip(self.x_min,self.x_max):
            a=np.arange(i, j+0.000001, 10**(-1*self.rounding[index]))
            a=np.round(a,self.rounding[index])  #create a array
            thedict[psosearch.parameters[index]]=a #add the list as a value and value at index parameters to a dictionary. 
            index=index+1
        #now thedict contains all the possible values. we want to create all the possible combinations
        allparams = thedict#sorted(thedict)
        combinations = it.product(*(thedict[Name] for Name in allparams))# use the dictionary to create all possible solutions.
        #for each combination in combinations, we need to run the psosearch
        combinations=list(combinations)
        combinations=[list(row) for row in combinations]# change the values to list instead of tuple
        return combinations
    
    @staticmethod
    def readparameters(params=None,estimator=None, task=None):
        if params ==None:
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
                    params = {"objective": ["tweedie","gamma"],
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
                      "learning_rate":  [0.1,2,2],
                      "neurons": [1,40,0],
                      "hiddenactivation": ['relu','sigmoid','tanh'],
                      "activation":['relu','sigmoid','tanh']}
        x_min=[]
        x_max=[]
        rounding=[]
        
        #params to be searched for
        parameters=[]
        for key in params:
            if key == 'objective' or key == 'optimizer' or key =='hiddenactivation' or key == 'activation' or key == 'kernel': #categorical variables mainly
                of=params[key]
                x_min.append(0)
                x_max.append(len(of)-1)
                parameters.append(key)
                rounding.append(0)
            elif key == 'learning_rate' or key=='max_depth' or key=='n_estimators' or key== 'subsample' or key=='neurons' or key =='C' or key =='gamma' or key=='degree':
                thelist=params[key]
                x_min.append(thelist[0])
                x_max.append(thelist[1])
                parameters.append(key)
                rounding.append(thelist[2])
        bounds = (np.asarray(x_min), np.asarray(x_max))   
        dimensions=len(x_min)
        
        defaultparams= {}
        if estimator == 'xgboost':
            if task =='binary classification': # default activation
                defaultparams['objective'] = 'binary:logistic'
            else:
                defaultparams['objective'] = 'reg:tweedie' 
            defaultparams['learning_rate']=0.01# default learning rate
            defaultparams['max_depth']=6# default max depth
            defaultparams['n_estimators']=40 # default number of estimators
            defaultparams['subsample']=0.99 # default subsample 
        elif estimator == 'gbdt':
            if task =='binary classification': # default activation
                defaultparams['objective'] = 'binary'
            else:
                defaultparams['objective'] = 'tweedie' 
            defaultparams['learning_rate']=0.01# default learning rate
            defaultparams['max_depth']=6# default max depth
            defaultparams['n_estimators']=40 # default number of estimators
            defaultparams['subsample']=0.99 # default subsample 
        elif estimator == 'mlp':
            #steps_per_epoch=4000 // batch_size
            defaultparams['batch_size']=12 # batch size of neural network based models
            defaultparams['epochs']=50 # default number of epochs of neural network based models
            defaultparams['shuffle']=True # true to shuffle the training data
            if task =='binary classification': # set the optimizer based on the task
                defaultparams['optimizer']='Adam'
            else:
                defaultparams['optimizer']='RMSprop'
            defaultparams['neurons'] =13 # default nb of neurons
            defaultparams['hiddenactivation']='sigmoid' # default value of hidden activation functions
            defaultparams['activation']='sigmoid' #default value of output activation function
            defaultparams['learning_rate']=0.01 # default learning rate
        elif estimator == 'svm':
            defaultparams['kernel']= 'rbf'
            defaultparams['C']= 5
            defaultparams['gamma'] =5
                 
                
        return parameters,defaultparams, x_min,x_max,rounding,bounds, dimensions,params
    
    
