#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:05:07 2024
@author: javi
"""

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from modules.mod_backtesting import mod_backtesting


def tests_predictions(df_predictions, start_tests, endin_tests):
    
    start_tests_i = start_tests[0]
    endin_tests_i = endin_tests[0]

    
    mod_backtesting(tests_data,y_tests,y_pred_bin,start_tests_i,endin_tests_i)   

    # CTest accuracy
    tests_accuracy = accuracy_score(y_tests, y_pred_bin) 
    
    return tests_accuracy
                             