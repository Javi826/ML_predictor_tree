#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 21:08:29 2024

@author: javi

"""
import os
from main.paths.paths import path_base,folder_tests_results

def mod_operations(df_signals, initial_capital, comision, start_tests):
    
    df_operations = df_signals[df_signals['signal'] != 0].copy()
    
    df_operations.loc[df_operations['signal'] ==  1, 'oper_rent'] = 0
    df_operations.loc[df_operations['signal'] == -1, 'oper_rent'] = df_operations['close'].pct_change(fill_method=None)

    df_operations.loc[df_operations['oper_rent'] != 0, 'oper_rent_gross'] = df_operations['oper_rent'] 
    df_operations.loc[df_operations['oper_rent'] != 0, 'oper_rent_nets']  = df_operations['oper_rent'] - (comision/100)
    
    df_operations['gross_capital'] = initial_capital
    df_operations['nets_capital']  = initial_capital
    
    df_operations['gross_capital'] = (df_operations['oper_rent_gross'] + 1).cumprod() * initial_capital
    df_operations['nets_capital']  = (df_operations['oper_rent_nets']  + 1).cumprod() * initial_capital
    
    excel_back_path = os.path.join(path_base, folder_tests_results, f"df_back_operations_{start_tests[0]}.xlsx")
    df_operations.to_excel(excel_back_path, index=False)
    
    return df_operations










