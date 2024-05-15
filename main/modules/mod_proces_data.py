#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:24:05 2024
@author: javi
"""
from main.modules.mod_pipeline import mod_pipeline
import os
import pandas as pd
import numpy as np


def mod_process_data(df_preprocess, start_train, endin_train, start_tests, endin_tests, MAES, type_split):
    
    
    if type_split == 'TRVAL':
        
                
        X_train_techi = mod_pipeline(df_preprocess, start_train, endin_train, start_tests, endin_tests, MAES,'X_train_techi')  
              
        X_train = X_train_techi

        y_train = mod_pipeline(df_preprocess, start_train, endin_train, start_tests, endin_tests, MAES, 'y_train')
        
        return X_train, y_train
    
    elif type_split == 'TESTS':
            
        X_tests_techi = mod_pipeline(df_preprocess, start_train, endin_train,  start_tests, endin_tests, MAES,'X_tests_techi')
        
        X_tests = X_tests_techi
        
        y_tests = mod_pipeline(df_preprocess, start_train, endin_train,  start_tests, endin_tests, MAES, 'y_tests')
        
        return X_tests, y_tests