#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:42:47 2024
@author: javi
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main.paths.paths import path_base,folder_tests_results
from main.functions.def_functions import filter_data_by_date_range,find_first_non_zero
from main.modules.mod_save_results import save_backs_results
from main.modules.mod_operations import mod_operations


def mod_backtesting(symbol, MAES, forrest_comb,tests_accuracy, df_preprocess, df_predictions, start_tests,endin_tests,loops_backs_results):
    
    start_tests_i = start_tests[0]
    endin_tests_i = endin_tests[0]
    
    df_tests = filter_data_by_date_range(df_preprocess,start_tests_i,endin_tests_i)
    df_recove_data = pd.DataFrame()
        
    df_recove_data['date']       = df_tests['date']
    df_recove_data['close']      = df_tests['close']
    df_recove_data['y_mavgs']    = df_tests['y_mavgs']
    df_recove_data.reset_index(drop=True, inplace=True)
    df_predictions.reset_index(drop=True, inplace=True)
    
    
    df_signals = pd.concat([df_recove_data, df_predictions], axis=1)
                 
    #SIGNALS
    #---------------------------------------------------------------------------------------------------   
    column_choose = 'y_mavgs'
    
    total_sum_next_five = df_signals[column_choose].iloc[:5].sum()
    
    if total_sum_next_five > 2:
        df_signals.at[0, column_choose] = 0
        
    df_signals['y_combi'] = df_signals['y_target'] + df_signals['y_preds']
    df_signals['signal']  = 0
    
    
    for i in range(len(df_signals)):
        
        if df_signals.loc[i, 'y_combi'] == 0:
            
            start_idx = max(0, i - 100)
            past_signals = df_signals['signal'][start_idx:i]
            
            last_non_zero = find_first_non_zero(past_signals)
            
            if df_signals.loc[i, 'y_combi'] == 0 and last_non_zero == -1:
                df_signals.loc[i, 'signal'] = 0
                
            elif df_signals.loc[i, 'y_combi'] == 0 and last_non_zero == 1:                
                df_signals.loc[i, 'signal'] = -1 
            
        elif df_signals.loc[i, 'y_combi'] == 2:
            
            start_idx = max(0, i - 100)
            past_signals = df_signals['signal'][start_idx:i]
            
            last_non_zero = find_first_non_zero(past_signals)
            
            if df_signals.loc[i, 'y_combi'] == 2 and last_non_zero == -1:
                df_signals.loc[i, 'signal'] = 1
                
            elif df_signals.loc[i, 'y_combi'] == 2 and last_non_zero == 1:                
                df_signals.loc[i, 'signal'] = 0 
                
            
        elif df_signals.loc[i, 'y_combi'] == 1:
            
            start_idx = max(0, i - 100)
            past_signals = df_signals['signal'][start_idx:i]
            
            last_non_zero = find_first_non_zero(past_signals)
    
            if df_signals.loc[i, 'y_preds'] == 1 and last_non_zero == -1:
               df_signals.loc[i, 'signal']  = 1
                
            elif df_signals.loc[i, 'y_preds'] == 1 and last_non_zero == 1:                
                 df_signals.loc[i, 'signal']  = 0     
                
            elif df_signals.loc[i, 'y_preds'] == 0 and last_non_zero == 1: 
                 df_signals.loc[i, 'signal']  = -1
                
            elif df_signals.loc[i, 'y_preds'] == 0 and last_non_zero == -1:                
                 df_signals.loc[i, 'signal']  = 0
                
            else:
                 df_signals.loc[i, 'signal'] = 1
    
                 
    #BACKTESTING-1
    #---------------------------------------------------------------------------------------------------              
                
    initial_capital = 10000
    comision = 0.2
                
    df_signals['day_g_returns'] = df_signals['close'].pct_change()
    df_signals['day_n_returns'] = df_signals['day_g_returns'] - (comision/100)
    df_signals['b&h_g_capital'] = initial_capital
    df_signals['b&h_g_capital'] = (df_signals['day_g_returns'] + 1).cumprod() * initial_capital
    
    df_signals['str_capital'] = initial_capital
    
    previous_str_capital = initial_capital
    
    for i in range(1, len(df_signals)):
        
        if df_signals.loc[i, 'signal'] == 0:
            
            start_idx = max(0, i - 100)
            past_signals  = df_signals['signal'][start_idx:i]
            last_non_zero = find_first_non_zero(past_signals)
            
            if last_non_zero == 1:
                current_str_capital = previous_str_capital * (1 + df_signals.loc[i, 'day_g_returns'])
            elif last_non_zero == -1:
                current_str_capital = previous_str_capital
            else:
                current_str_capital = previous_str_capital
                
        elif df_signals.loc[i, 'signal'] == 1:
             current_str_capital = previous_str_capital * (1 -(comision/100))
            
    
        elif df_signals.loc[i, 'signal'] == -1:
             current_str_capital = (previous_str_capital * (1 + df_signals.loc[i, 'day_g_returns'])) * (1 -(comision/100))
            
        else:
            current_str_capital = previous_str_capital
            
        current_str_capital = int(current_str_capital)
        
        df_signals.loc[i, 'str_capital'] = current_str_capital
        
        previous_str_capital = current_str_capital
        
        
    #MAXDRAWDOWN
    #---------------------------------------------------------------------------------------------------  
          
    peaks     = df_signals['str_capital'].cummax()
    drawdowns = (df_signals['str_capital'] - peaks) / peaks
    max_drawdown_st = drawdowns.min() * 100

    
    #VOLATILITY & SHARPE_RATIO
    #--------------------------------------------------------------------------------------------------- 
    df_signals['day_str_returns'] = df_signals['str_capital'].pct_change() 
    mean_str_returns   = df_signals['str_capital'].pct_change().mean() * 100   
    volatility_st      = df_signals['str_capital'].pct_change().std() *100
    volatility_st      = volatility_st * np.sqrt(252)
    excess_returns     = mean_str_returns - 0
    sharpe_ratio_st    = excess_returns / volatility_st
        
    #plt.figure(figsize=(12, 6))
    #plt.plot(df_signals['date'], df_signals['b&h_g_capital'], label='Buy & Hold Capital')
    #plt.plot(df_signals['date'], df_signals['str_capital'], label='Strategy Capital', linestyle='--')
    #plt.xlabel('Date')
    #plt.ylabel('Capital')
    #plt.title('Capital Over Time: Buy and Hold vs Strategy')
    #plt.legend()
    #plt.grid(True)
    #plt.show()
                               
      
    # SAVE SIGNALS FILE
    #---------------------------------------------------------------------------------------------------
    excel_signals_path = os.path.join(path_base, folder_tests_results, f"df_market_signals_{start_tests_i}.xlsx")
    df_signals.to_excel(excel_signals_path, index=False) 
    
    #OPERATIONS
    #---------------------------------------------------------------------------------------------------
    
    df_operations = mod_operations(df_signals, initial_capital, comision, start_tests)
    
    #BRIEF & SAVE
    #---------------------------------------------------------------------------------------------------
    
    last_capital    = df_operations['nets_capital'].dropna().iloc[-1]   
    return_strategy = ((last_capital - initial_capital) / initial_capital) * 100
    
    price_first     = df_tests['close'].iloc[0]
    price_lasts     = df_tests['close'].iloc[-1]
    return_buy_hold = (price_lasts - price_first) / price_first * 100
    n_operations    = (df_operations.shape[0])/2
    sum_sginal      = df_operations['signal'].sum()
    rent_op_mean    = df_operations['oper_rent'].mean()*100
    
    print('\n')
    print(f"{start_tests_i}-{endin_tests_i}")
    print("symbol back             :",symbol)
    print("Initl capital value     :", "{:.2f}".format(initial_capital))
    print("Final capital value     :", "{:.2f}".format(last_capital))
    print("Rentability buy&hold    :", "{:.2f}".format(return_buy_hold), "%")
    print("Rentability strategy    :", "{:.2f}".format(return_strategy), "%")
    print("Number of operations    :", n_operations)
    print("Sum signal              :",sum_sginal)
    print("Volatility strategy     :", "{:.2f}".format(volatility_st), "%")
    print("Sharpe ratio strategy   :", round(sharpe_ratio_st, 2))
    print("Max drawdown strategy   :", "{:.2f}".format(max_drawdown_st), "%")
    
    save_backs_results(symbol,MAES, forrest_comb,tests_accuracy, start_tests, initial_capital,last_capital,return_buy_hold,return_strategy,n_operations,sum_sginal,rent_op_mean,volatility_st,sharpe_ratio_st,max_drawdown_st, loops_backs_results)
         
    return 