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
          
    start_train_i  = start_train[0]
    endin_train_i  = endin_train[0]
    start_tests_i  = start_tests[0]
    endin_tests_i  = endin_tests[0]
    start_market_i = start_tests[0]
    endin_market_i = endin_tests[0]
    
    df_date_lag_dir = df_preprocess.copy()
    

    #print(df_date_lag_dir)   
    #DATA SPLIT
    #------------------------------------------------------------------------------     
    train_data  = df_date_lag_dir[(df_date_lag_dir['date'] >= start_train_i) & (df_date_lag_dir['date'] <  endin_train_i)]
    tests_data  = df_date_lag_dir[(df_date_lag_dir['date']  > start_tests_i) & (df_date_lag_dir['date'] <= endin_tests_i)]
    market_data = df_date_lag_dir[(df_date_lag_dir['date']  > start_market_i) & (df_date_lag_dir['date'] <= endin_market_i)]
    
    dlags_columns_selected = [col for col in df_date_lag_dir.columns if col.startswith('rets') or col.startswith('fet')]
    train_excel_path    = os.path.join(path_base, folder_zinputs_model, f"train_data_techi_{start_train_i}.xlsx")
    train_data_selected = train_data[dlags_columns_selected]
    train_data_selected.to_excel(train_excel_path, index=False)
    
    
    #X_TRAIN_techi + dweek
    #------------------------------------------------------------------------------
    
    if data_type == 'X_train_techi':
           
        X_train_techi = train_data[dlags_columns_selected]
        X_train_techi = pd.DataFrame(X_train_techi)
        
        return X_train_techi
    
      
    #X_TESTS
    #------------------------------------------------------------------------------        
    elif data_type == 'X_tests_techi':
                
        X_tests_techi = tests_data[dlags_columns_selected]
        X_tests_techi = pd.DataFrame(X_tests_techi)
        
        return X_tests_techi
    
    #X_MARKET
    #------------------------------------------------------------------------------ 
    
    elif data_type == 'X_market_techi':
        
        dlags_columns_selected = [col for col in df_date_lag_dir.columns if col.startswith('rets') or col.startswith('fet')]
        market_excel_path    = os.path.join(path_base, folder_zinputs_model, f"market_data_techi_{endin_market_i}.xlsx")
        market_data_selected = market_data[dlags_columns_selected]
        market_data_selected.to_excel(market_excel_path, index=False)
                
        X_market_techi = market_data[dlags_columns_selected]
        X_market_techi = pd.DataFrame(X_market_techi)
        
        return X_market_techi
                       
    #y_train,valid,tests
    #------------------------------------------------------------------------------             
    elif data_type == 'y_train':
        
        y_train = train_data['y_target']
               
        return y_train
            
    elif data_type == 'y_tests':
    
        y_tests = tests_data['y_target']
        
        return y_tests
    