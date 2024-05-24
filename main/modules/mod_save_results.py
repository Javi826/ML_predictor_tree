#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:14:17 2024

@author: javi
"""
import os
import pandas as pd
from main.paths.paths import path_base,folder_tests_results
from main.functions.def_functions import tests_results,backs_results

def save_tests_results(symbol, MAES, start_train, start_tests, endin_tests,n_estimatorss,max_depths,min_samples_sp,min_samples_lf, tests_accuracy,loops_tests_results):
    
    start_train_strg = start_train[0]
    start_train_year = start_train_strg[:4]
    
    dc_tests_results       = tests_results(symbol, MAES, start_train, start_tests, endin_tests,n_estimatorss,max_depths,min_samples_sp,min_samples_lf, tests_accuracy)
    loops_tests_results.append(dc_tests_results)      
    df_loops_tests_results = pd.DataFrame(loops_tests_results)
    
    file_name        = f"df_tests_results_{start_train_year}.xlsx"
    excel_tests_path = os.path.join(path_base, folder_tests_results, file_name)
    df_loops_tests_results.to_excel(excel_tests_path, index=False)
    
    
def save_backs_results(symbol,MAES, forrest_comb,tests_accuracy, start_tests, initial_capital,last_capital,total_return_buy_hold,percentage_change,n_operations,sum_sginal,rent_op_mean, volatiliy_st, sharpe_ratio_st,max_drawdown_st, loops_backs_results):
    
    
    dc_backs_results       = backs_results(symbol,MAES,forrest_comb, tests_accuracy, initial_capital,last_capital,total_return_buy_hold,percentage_change,n_operations,sum_sginal,rent_op_mean,volatiliy_st, sharpe_ratio_st,max_drawdown_st)
    loops_backs_results.append(dc_backs_results)      
    df_loops_backs_results = pd.DataFrame(loops_backs_results)
    
    file_name        = f"df_backs_results_{start_tests[0]}.xlsx"
    excel_tests_path = os.path.join(path_base, folder_tests_results, file_name)
    df_loops_backs_results.to_excel(excel_tests_path, index=False)
    
    

