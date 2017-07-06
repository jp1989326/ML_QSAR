import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn import preprocessing, cross_validation, metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor as rfr
from bayes_opt import BayesianOptimization

'''
Valid scoring options are ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 
'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 
'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 
'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 
'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']

SCORERS = dict(r2=r2_scorer,
               neg_median_absolute_error=neg_median_absolute_error_scorer,
               neg_mean_absolute_error=neg_mean_absolute_error_scorer,
               neg_mean_squared_error=neg_mean_squared_error_scorer,
               median_absolute_error=median_absolute_error_scorer,
               mean_absolute_error=mean_absolute_error_scorer,
               mean_squared_error=mean_squared_error_scorer,
               accuracy=accuracy_scorer, roc_auc=roc_auc_scorer,
               average_precision=average_precision_scorer,
               log_loss=log_loss_scorer,
               neg_log_loss=neg_log_loss_scorer,
               adjusted_rand_score=adjusted_rand_scorer)
               
'''

class SVM_CV():
    
    def __init__(self): 
        pass
  #      print 'starting...'       

    
    
    def trainsvr (self, train, trainlabel, seed, Cmin, Cmax, numC, rmin, rmax, numr, degree=3,\
                  verbose = 1, method = 'roc_auc', rad_stat =2):
        C_range=np.logspace(Cmin, Cmax, num=numC, base=2,endpoint= True)
        gamma_range=np.logspace(rmin, rmax, num=numr, base=2,endpoint= True)
        
        scr = SVR(kernel=seed)
#        mean_score=[]
        df_C_gamma= pd.DataFrame({'gamma_range':gamma_range})
#        df_this = DataFrame({'gamma_range':gamma_range})
        count = 0 
        for C in C_range:    
            score_C=[]    
#            score_C_this = []
            count=count+1
            for gamma in gamma_range:                   
     
                scr.C = C
                scr.gamma = gamma
                scr.degree = degree
                scr.random_state = rad_stat
                this_scores = cross_val_score(scr, train, trainlabel, scoring=method, cv=10, n_jobs=-1 \
                                              )
                
                score_C.append(np.mean(this_scores))                                      

               #score_C_this.append(np.mean(this_scores))
            if verbose ==1:    
                print (np.mean(score_C) )
                print ("%r cycle finished, %r left" %(count, numC-count))
            df_C_gamma[C]= score_C
            #df_this[C] = score_C_this        
        
        return df_C_gamma 
