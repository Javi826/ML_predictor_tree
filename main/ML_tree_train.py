#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:33:28 2024
@author: javi
"""

#GOOGLE COLAB
#------------------------------------------------------------------------------
#ruta_directorio_clonado = '/content/ML_predictor'
#sys.path.append(ruta_directorio_clonado)

#GOOGLE JUPYTERd
#------------------------------------------------------------------------------
#nuevo_directorio = "/home/jupyter/ML_predictor_tree"
#os.chdir(nuevo_directorio)

import os
import time
import warnings
import pandas as pd
import yfinance as yf
from datetime import date,timedelta
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')

start_time = time.time()
#VISUALIZATION PRINTS
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

from main.functions.def_functions import feature_importance
from main.paths.paths import path_base,folder_csv
from main.modules.mod_data_build import mod_data_build
from main.modules.mod_preprocess import mod_preprocess
from main.modules.mod_proces_data import mod_process_data
from main.modules.mod_save_results import save_tests_results
from main.modules.mod_save_model import save_model
from main.modules.mod_backtesting import mod_backtesting
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

loops_backs_results =[]
loops_tests_results =[]

# YAHOO CALL + SAVE + READING file
#------------------------------------------------------------------------------
#symbol_ra = ["^GSPC","^IXIC","BBVA.MC","TEF.MC"]
#symbol_ra = ["^GSPC"]
symbol_ra = ["BBVA.MC"]
#symbol_ra = ["NVDA"]

for symbol in symbol_ra:
    
    start_date       = "2000-01-01"
    endin_date       = date.today().strftime("%Y-%m-%d")
    endin_date_yahoo = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    index_price_data = yf.download(symbol, start=start_date, end=endin_date_yahoo)
    
    index_file_name = f"{symbol}_train.csv"
    csv_path = os.path.join(path_base, folder_csv, index_file_name)
    index_price_data.to_csv(csv_path)
    
    #CALL BUILD
    #------------------------------------------------------------------------------
    df_data  = pd.read_csv(csv_path, header=None, skiprows=1, names=['date','open','high','low','close','adj_close','volume'])
    df_build = mod_data_build(symbol, df_data)
    
    e_features='MAVG'
    #e_features='returns'
    start_train   = ['2000-01-01']
    endin_train   = ['2023-12-30']
    
    start_tests   = ['2024-01-01']
    endin_tests   = [endin_date]
       
    MAES_ra=[1]
    for MAES in MAES_ra:
        
        #CALL PREPROCESSING
        #------------------------------------------------------------------------------
        df_preprocess = mod_preprocess(df_build,MAES, e_features,data_type='train')
            
        #n_estimators_ra    = [10,20,30,40,50,60,70]
        #min_samples_lf_ra  = [1,2,3,4,5,6,7,8,9,10]
        #min_samples_sp_ra  = [2,3,4,5,6,7,8,9,10]
        #max_depths_ra      = [1,2,3,4,5,6,7,8]
 
        n_estimators_ra    = [30]
        max_depths_ra      = [6]
        min_samples_sp_ra  = [2]
        min_samples_lf_ra  = [8]
                
        X_train, y_train = mod_process_data(df_preprocess, start_train, endin_train, start_tests, endin_tests, MAES, 'TRVAL')
        X_tests, y_tests = mod_process_data(df_preprocess, start_train, endin_train, start_tests, endin_tests, MAES, 'TESTS')
               
        for n_estimatorss in n_estimators_ra:
            for max_depths in max_depths_ra:
                for min_samples_sp in min_samples_sp_ra:
                    for min_samples_lf in min_samples_lf_ra:
                        
                        forrest_comb = [n_estimatorss,max_depths,min_samples_sp,min_samples_lf]
                        model = RandomForestClassifier(
                            n_estimators=n_estimatorss,            
                            criterion='gini',         
                            max_depth=max_depths,                
                            min_samples_split=min_samples_sp,         
                            min_samples_leaf=min_samples_lf,          
                            max_features=None,           
                            random_state=42)
                        
                        model.fit(X_train, y_train) 
                        
                        #feature_importance(model, X_train)
                        
                        y_pred = model.predict(X_tests)
                        df_predictions = pd.DataFrame({'y_target': y_tests, 'y_preds': y_pred})
                        
                        # Calcular la precisión del modelo
                        tests_accuracy = accuracy_score(y_tests, y_pred)
                        
                        save_tests_results(symbol, MAES, start_train, start_tests, endin_tests, n_estimatorss, max_depths, min_samples_sp, min_samples_lf, tests_accuracy, loops_tests_results)
                              
                        mod_backtesting(symbol,MAES,forrest_comb,tests_accuracy, df_preprocess, df_predictions, start_tests,endin_tests,loops_backs_results) 
                        print("Accuracy                :", tests_accuracy)                      
                        
                        save_model(model)
    
os.system("afplay /System/Library/Sounds/Ping.aiff")
elapsed_time   = time.time() - start_time
elapsed_hours, elapsed_minutes = divmod(elapsed_time / 60, 60)
print(f"Total time take to train: {int(elapsed_hours)} hours, {int(elapsed_minutes)} minutes")