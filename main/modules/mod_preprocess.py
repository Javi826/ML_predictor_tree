#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""

import os
import numpy as np
import pandas as pd
from main.paths.paths import path_base,folder_preprocess
from main.functions.def_functions import filter_data_by_date_range, df_plots, diff_series
from main.functions.def_featuring import MAV, EMA, ROC, MOM, STK, STD

def mod_preprocess (df_build, MAES, e_features,data_type):
    
    aveg1 = MAES
    aveg2 = aveg1 * 5
    df_preprocess = df_build.copy() 
    
    if e_features == 'MAVG':
        
        #SHORT & LONG MAVG
        #----------------------------------------------------------------------------------------------       
        df_preprocess['short_MAVG'] = df_preprocess['close'].rolling(window=aveg1).mean()
        df_preprocess['longs_MAVG'] = df_preprocess['close'].rolling(window=aveg2).mean()
    
        df_preprocess['y_mavgs']   = np.where(df_preprocess['short_MAVG'] > df_preprocess['longs_MAVG'], 1, 0)
        df_preprocess['y_target']  = df_preprocess['y_mavgs'].shift(-1)

        #FEATURING   
        #----------------------------------------------------------------------------------------------    
        df_preprocess['fet_MAV010']  = MAV(df_preprocess,10)
        df_preprocess['fet_MAV030']  = MAV(df_preprocess,30)
        df_preprocess['fet_MAV0200'] = MAV(df_preprocess,200)
        df_preprocess['fet_ROC010']  = ROC(df_preprocess['close'], 10)
        df_preprocess['fet_ROC030']  = ROC(df_preprocess['close'], 30)
        df_preprocess['fet_MOM010']  = MOM(df_preprocess['close'], 10)
        df_preprocess['fet_MOM030']  = MOM(df_preprocess['close'], 30)
        df_preprocess['fet_EMA010']  = EMA(df_preprocess, 10)
        df_preprocess['fet_EMA030']  = EMA(df_preprocess, 30)
        df_preprocess['fet_EMA200']  = EMA(df_preprocess, 200)
        df_preprocess['fet_%K010']   = STK(df_preprocess['close'], df_preprocess['low'], df_preprocess['high'], 10)
        df_preprocess['fet_%D010']   = STD(df_preprocess['close'], df_preprocess['low'], df_preprocess['high'], 10)
        df_preprocess['fet_%K030']   = STK(df_preprocess['close'], df_preprocess['low'], df_preprocess['high'], 30)
        df_preprocess['fet_%D030']   = STD(df_preprocess['close'], df_preprocess['low'], df_preprocess['high'], 30)
        df_preprocess['fet_%D200']   = STD(df_preprocess['close'], df_preprocess['low'], df_preprocess['high'], 200)
        df_preprocess['fet_%K200']   = STK(df_preprocess['close'], df_preprocess['low'], df_preprocess['high'], 200)
        #df_preprocess['fet_d_week']  = df_build['day_week']
               
        
        #RETURNS
        #----------------------------------------------------------------------------------------------    
       
    if e_features == 'returns':
        
       df_preprocess['returns']    = df_preprocess['close'] - df_preprocess['close'].shift(1)
       df_preprocess['direction']  = np.where(df_preprocess['returns'] > 0, 1, 0)
       df_preprocess['y_target']   = df_preprocess['direction'].shift(-1)
       
       lags = 20
             
       cols = []
       for lag in range(1,lags+1):
           col = f'rets_lag_{str(lag).zfill(2)}'
           df_preprocess[col] = df_preprocess['returns'].shift(lag)
           cols.append(col)
    
    #df_preprocessing['date'] = pd.to_datetime(df_preprocessing['date'])
    df_plots(df_preprocess['date'],df_preprocess['close'],'date','close','lines')
    
    # SAVE Dataframe
    #---------------------------------------------------------------------------------------------- 
    if data_type == 'train':
        
        df_preprocess.dropna(inplace=True)        
        excel_file_path = os.path.join(path_base, folder_preprocess, f"df_preprocess_{str(aveg1).zfill(2)}_train.xlsx")
        df_preprocess.to_excel(excel_file_path, index=False)
        
    if data_type == 'market':
        
        excel_file_path = os.path.join(path_base, folder_preprocess, f"df_preprocess_{str(aveg1).zfill(2)}_market.xlsx")
        df_preprocess.to_excel(excel_file_path, index=False)
    
    return df_preprocess

#Generar las columnas de retraso para las características adicionales
#fet_cols = [col for col in df_preprocess.columns if col.startswith('fet_')]
#lagged_features = []

#for col in fet_cols:
#    lagged_feature_cols = pd.concat([df_preprocess[col].shift(lag).rename(f'fet_lag_{str(lag).zfill(2)}_{col[4:]}') for lag in range(1, lags + 1)], axis=1)
#    lagged_features.append(lagged_feature_cols)

# Concatenar las características adicionales retrasadas al dataframe principal
   #df_preprocess = pd.concat([df_preprocess] + lagged_features, axis=1)