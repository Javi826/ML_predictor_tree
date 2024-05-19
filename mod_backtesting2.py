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
from main.functions.def_functions import backs_results
from main.modules.mod_save_results import save_backs_results


def mod_backtesting2(symbol, MAES, forrest_comb, df_tests, df_predictions, start_tests,endin_tests,loops_backs_results):
    
    start_tests_i = start_tests[0]
    endin_tests_i = endin_tests[0]
    
    df_recove_data = pd.DataFrame()
        
    df_recove_data['date']  = df_tests['date']
    df_recove_data['close'] = df_tests['close']
    df_recove_data.reset_index(drop=True, inplace=True)
    df_predictions.reset_index(drop=True, inplace=True)
   
    df_signals = pd.concat([df_recove_data, df_predictions], axis=1)
    
    
    #SIGNAL
    #---------------------------------------------------------------------------------------------------   
    column_choose = 'direction'
    
    df_signals['signal'] = np.select(
        [(df_signals[column_choose] == 1) & (df_signals[column_choose].shift(1) == 0),
         (df_signals[column_choose] == 1) & (df_signals[column_choose].shift(1) == 1),
         (df_signals[column_choose] == 0) & (df_signals[column_choose].shift(1) == 0), 
         (df_signals[column_choose] == 0) & (df_signals[column_choose].shift(1) == 1)], 
        [1, 0, 0, -1], default=np.nan)
    
    df_signals.at[0, 'signal'] = 1
          
    # SAVE FILE
    #---------------------------------------------------------------------------------------------------
    excel_signals_path = os.path.join(path_base, folder_tests_results, f"df_market_signals_{start_tests_i}.xlsx")
    df_signals.to_excel(excel_signals_path, index=False) 
    
    #BACKTESTING
    #---------------------------------------------------------------------------------------------------
    
    df_backtesting = df_signals[df_signals['signal'] != 0].copy()
    
    df_backtesting.loc[df_backtesting['signal'] ==  1, 'oper_rent'] = 0
    df_backtesting.loc[df_backtesting['signal'] == -1, 'oper_rent'] = df_backtesting['close'].pct_change(fill_method=None)

    
    comision = 0.5
    
    df_backtesting.loc[df_backtesting['oper_rent'] != 0, 'oper_rent_gross'] = df_backtesting['oper_rent']
    df_backtesting.loc[df_backtesting['oper_rent'] != 0, 'oper_rent_nets']  = df_backtesting['oper_rent'] - (comision/100)
    
    initial_capital = 10000
    df_backtesting['gross_capital'] = initial_capital
    df_backtesting['nets_capital']  = initial_capital
    
    
    df_backtesting['gross_capital'] = (df_backtesting['oper_rent_gross'] + 1).cumprod() * initial_capital
    df_backtesting['nets_capital']  = (df_backtesting['oper_rent_nets']  + 1).cumprod() * initial_capital
    
    excel_back_path = os.path.join(path_base, folder_tests_results, f"df_back_operations_{start_tests[0]}.xlsx")
    df_backtesting.to_excel(excel_back_path, index=False)
    
    last_capital = df_backtesting['nets_capital'].dropna().iloc[-1]
    
    return_strategy = ((last_capital - initial_capital) / initial_capital) * 100
    
    price_first     = df_tests['close'].iloc[0]
    price_lasts     = df_tests['close'].iloc[-1]
    return_buy_hold = (price_lasts - price_first) / price_first * 100
    n_operations    = (df_backtesting.shape[0]-1)/2
    rent_op_mean    = df_backtesting['oper_rent'].mean()*100
    
    sharpe_ratio_st       = 0
    max_drawdown_st = 0
     
    # Imprimir los valores con dos decimales
    print('\n')
    print(f"{start_tests_i}-{endin_tests_i}")
    print("symbol back             :",symbol)
    print("Initl capital value     :", "{:.2f}".format(initial_capital))
    print("Final capital value     :", "{:.2f}".format(last_capital))
    print("Rentability buy&hold    :", "{:.2f}".format(return_buy_hold), "%")
    print("Rentability strategy    :", "{:.2f}".format(return_strategy), "%")
    print("Number of operations    :", n_operations)
    #print("Volatility for strategy   :", round(volatility_st, 2))
    #print("Sharpe ratio for strategy :", round(sharpe_ratio_st, 2))
    #print("Max drawdown for strategy :", round(max_drawdown_st, 2))
    
    save_backs_results(symbol,MAES, forrest_comb, start_tests, initial_capital,last_capital,return_buy_hold,return_strategy,n_operations,rent_op_mean,sharpe_ratio_st,max_drawdown_st, loops_backs_results)
         
    return 


    total_sum_next_five = df_signals[column_choose].iloc[:3].sum()
    
    if total_sum_next_five > 2:
        df_signals.at[0, column_choose] = 0

    # MAXDRWON
    #cum_returns = (1 + returns).cumprod()
    #max_drawdown_st = ((cum_returns.cummax() - cum_returns) / cum_returns.cummax()).max()

    # SHAPE RATIO
    #returns = df_backtesting['nets_capital']
    #risk_free_rate        = 0  # Supongamos una tasa libre de riesgo del 0% por simplicidad
    #volatility_st         = np.std(returns)
    #sharpe_ratio_st       = (np.mean(returns) - risk_free_rate) / volatility_st