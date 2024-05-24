#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:33:28 2024
@author: javi
"""
import os
import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date,timedelta


from main.modules.mod_data_build import mod_data_build
from main.modules.mod_preprocess import mod_preprocess
from main.modules.mod_proces_data import mod_process_data
from main.paths.paths import path_base,folder_csv


# YAHOO CALL + SAVE + READING file
#------------------------------------------------------------------------------
symbol     = ["BBVA.MC"]
   
#print('\n')
#print(symbol)
start_date = "2023-01-01"
endin_date       = date.today().strftime("%Y-%m-%d")
endin_date_yahoo = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
index_price_data = yf.download(symbol, start=start_date, end=endin_date_yahoo)
                              
index_file_name = f"{symbol}_market.csv"
csv_path = os.path.join(path_base, folder_csv, index_file_name)
index_price_data.to_csv(csv_path)

#CALL BUILD
#------------------------------------------------------------------------------
df_data  = pd.read_csv(csv_path, header=None, skiprows=1, names=['date','open','high','low','close','adj_close','volume'])
df_build = mod_data_build(symbol, df_data)

MAES = 1
e_features='MAVG'
#e_features='returns'
start_train  = ['2024-01-01']
endin_train  = ['2024-01-01']
start_market = [(date.today() - timedelta(days=5)).strftime("%Y-%m-%d")]
endin_market = [(date.today() + timedelta(days=1)).strftime("%Y-%m-%d")]
  
#CALL PREPROCESSING
#------------------------------------------------------------------------------
df_preprocess  = mod_preprocess(df_build,MAES , e_features,data_type='market')

print(df_preprocess.tail(1).T)
X_market = mod_process_data(df_preprocess, start_train, endin_train, start_market, endin_market, MAES, 'market')

model_file = '/Users/javi/Desktop/ML/ML_predictor_tree/tree_models/random_forest_model.pkl'

with open(model_file, 'rb') as f:
    model = pickle.load(f)
print("Model loaded correctly")

print("Prediction date:", np.datetime_as_string(df_preprocess.tail(1)['date'].values[0], unit='D'))

y_pred = model.predict(X_market)
print(y_pred)

