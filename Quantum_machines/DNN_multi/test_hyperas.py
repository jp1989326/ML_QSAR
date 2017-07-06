'''
Created by Peng 2017-07-03 for testing hyperas
'''
import numpy as np
import pandas as pd
import molml 
from molml.features import CoulombMatrix, BagOfBonds
import sys
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler 
import matplotlib.pyplot as plt

import timeit
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
import seaborn as sns
from IPython.core.pylabtools import figsize
from sklearn.cross_validation import StratifiedKFold

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe

from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model, load_model
from keras import regularizers
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

from keras.datasets import mnist
from keras.utils import np_utils

def get_pca_features(nb_pca=16, feature_list=None):
#choose n components   
    pca_fit = PCA(n_components=nb_pca).fit(feature_list[0])
    pca_train = pca_fit.transform(feature_list[0])
    
    if np.shape(feature_list)[-1] == 3:
        pca_vali = pca_fit.transform(feature_list[1])
        pca_test = pca_fit.transform(feature_list[2])
    
        return [pca_train, pca_vali, pca_test], pca_fit
    
    elif np.shape(feature_list)[-1] == 2:
        pca_test = pca_fit.transform(feature_list[-1])   
        return [pca_train, pca_test], pca_fit
    
def get_pca_scale(pca_set):
#abs scale the pca components     
    max_abs=MaxAbsScaler()
    
    if np.shape(pca_set)[-1] == 3 :
        
        pca_set_scale=[0,0,0]
        pca_set_scale[0]= max_abs.fit_transform(pca_set[0])
        pca_set_scale[1], pca_set_scale[2]= max_abs.transform(pca_set[1]), max_abs.transform(pca_set[2])

        return pca_set_scale
    
    if np.shape(pca_set)[-1] == 2 :
        pca_set_scale=[0,0]
        pca_set_scale[0]= max_abs.fit_transform(pca_set[0])
        pca_set_scale[-1] = max_abs.transform(pca_set[-1])

        return pca_set_scale 
    
def get_nb_pca(ratio_list, thres_pca):
    
    for i in xrange(1, len(ratio_list)):
        
        if sum(ratio_list[:i])>thres_pca:
            
            nb_pca = i
            break
            
    return nb_pca      

def get_fit_targets(target_list = None):
    columns_list = target_list[0].columns
    train_targets = []
    vali_targets = []
    for i in columns_list:
        train_targets.append(np.array(target_list[0][i]))
        vali_targets.append(np.array(target_list[1][i]))
        
    return train_targets, vali_targets    

def get_attributes(element_list, coord_list):
    # produce [element, coord] lists for applying coulomb matrix and bob convertion

    fit_list = []

    for i in range(0, len(element_list)):
        fit_list.append((element_list[i], coord_list[i]))
        
    return fit_list 


def get_train_vali_test(train_list, target_list, test_szie=0.2, random_state = 32):
    
    train, test, trainlabel, testlabel = train_test_split(train_list, target_list,\
                                                      test_size=0.2, random_state = 32)

    train_, train_vali, train_label, vali_label = train_test_split(train, trainlabel, \
                                                              test_size=0.2, random_state = 32)
    
    return [train_, train_vali, test], [train_label, vali_label, testlabel]


def get_train_test(train_list, target_list, test_szie=0.2, random_state = 32):
    
    train, test, trainlabel, testlabel = train_test_split(train_list, target_list,\
                                                      test_size=0.2, random_state = 32)
    
    return [train,  test], [trainlabel, testlabel]


def data():
    
    df_data = pd.read_pickle('/home/peng/git/ML_QSAR/Quantum_machines/DNN_multi/df_bob_target.pkl')
    feature_list = df_data['feature_list']
    train = feature_list[0]
    test = feature_list[1]
    return train, test


def hyperas_sdae_features(feature_list, encoding_dim = 92):
    
    noise_factor = 0.5
    
    train_noise = feature_list[0] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=feature_list[0].shape) 
    vali_noise = feature_list[1] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=feature_list[1].shape) 


    train_noisy = np.clip(train_noise, 0., 1.)
    vali_noisy = np.clip(vali_noise, 0., 1.)

    
    input_dim= Input(shape=(np.shape(feature_list[0])[-1],))
    

    
    encoded = Dense({{choice([500, 600, 700])}}, activation='tanh')(input_dim)
    
    encoded = Dense(92, activation='tanh', activity_regularizer=regularizers.activity_l1(10e-5))(encoded)

    decoded = Dense(np.shape(feature_list[0])[-1], activation='linear')(encoded)


                    
    autoencoder = Model(input=input_dim, output=decoded)

    encoder = Model(input=input_dim, output=encoded)
                     
    autoencoder.compile(optimizer='sgd', loss='mse')

    start = timeit.default_timer()
                     
    history = autoencoder.fit(train_noisy, feature_list[0],
                    nb_epoch=10,
                    batch_size=100,
                    shuffle=True,
                    verbose = 0,
                    validation_data=(vali_noisy, feature_list[1])
                    )

    stop = timeit.default_timer()

    print ("The running takes %r min" %((stop-start)/60))    
    
    sdae_train, sdae_test = encoder.predict(feature_list[0]), encoder.predict(feature_list[-1])
    
    score, mse = autoencoder.evaluate(vali_noisy, feature_list[1], verbose = 0)
    
    print ('test mse: ', mse)
                     
    return {'loss': mse, 'status':STATUS_OK, 'model':hyperas_sdae_features}

def model(train, test):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    model = Sequential()
    model.add(Dense(512, input_shape=(666,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(666))
    model.add(Activation('linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms, metrics=['mse'])

    model.fit(train, train,
              batch_size={{choice([64, 128])}},
              nb_epoch=1,
              verbose=2
              )
    score, mse = model.evaluate(test, test, verbose=0)
    print('Test accuracy for new:', mse)
    return {'loss': mse, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    
    train, test = data()

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials()                                         
                                         )

