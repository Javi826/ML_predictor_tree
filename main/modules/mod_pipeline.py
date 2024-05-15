#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""

import pandas as pd
import os
from main.paths.paths import path_base,folder_zinputs_model


def mod_pipeline(df_preprocess, start_train, endin_train, start_tests, endin_tests, lags, data_type):
          
    start_train_i = start_train[0]
    endin_train_i = endin_train[0]
    start_tests_i = start_tests[0]
    endin_tests_i = endin_tests[0]
    
    df_date_lag_dir = df_preprocess.copy()

          
    #DATA SPLIT
    #------------------------------------------------------------------------------     
    train_data = df_date_lag_dir[(df_date_lag_dir['date'] >= start_train_i) & (df_date_lag_dir['date'] <  endin_train_i)]
    tests_data = df_date_lag_dir[(df_date_lag_dir['date']  > start_tests_i) & (df_date_lag_dir['date'] <= endin_tests_i)]
    
    dlags_columns_selected = [col for col in df_date_lag_dir.columns if col.startswith('ret') or col.startswith('fet')]
    train_techi_name    = f"train_data_techi_{start_train_i}.xlsx"
    train_excel_path    = os.path.join(path_base, folder_zinputs_model, train_techi_name)
    train_data_selected = train_data[dlags_columns_selected]
    train_data_selected.to_excel(train_excel_path, index=False)
    
    
    #X_TRAIN_techi + dweek
    #------------------------------------------------------------------------------
    
    if data_type == 'X_train_techi':
           
        X_train_techi = train_data[dlags_columns_selected]
        #scaler        = StandardScaler()
        #scaler        = MinMaxScaler()
        #X_train_techi = scaler.fit_transform(X_train_techi)
        X_train_techi = pd.DataFrame(X_train_techi)
        
        return X_train_techi
    
    
    
    #X_TESTS
    #------------------------------------------------------------------------------        
    elif data_type == 'X_tests_techi':
        
        
        X_tests_techi = tests_data[dlags_columns_selected]

        #scaler        = StandardScaler()
        #scaler       = MinMaxScaler()
       # X_tests_techi = scaler.fit_transform(X_tests_techi)
        X_tests_techi = pd.DataFrame(X_tests_techi)
        #X_tests_techi = X_tests_techi.values.reshape(-1, n_features)
        
        return X_tests_techi

                    
    
    #y_train,valid,tests
    #------------------------------------------------------------------------------             
    elif data_type == 'y_train':
        
        y_train = train_data['direction']
        #y_train = y_train.values
               
        return y_train
    
        
    elif data_type == 'y_tests':
    
        y_tests = tests_data['direction']
        #y_tests = y_tests.values
        
        return y_tests
    