#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def day_week(df_build):
       
    df_build['date']     = pd.to_datetime(df_build['date'])
    df_build['day_week'] = df_build['date'].dt.dayofweek   
    
    return df_build

def date_anio(df_build):
    
    df_build['date']      = pd.to_datetime(df_build['date'])    
    df_build['date_anio'] = df_build['date'].dt.year.astype(str).str[:4]
    
    return df_build

def one_hot_months(df_build):
    
    df_build['date'] = pd.to_datetime(df_build['date'])  
    
    months_columns = []
    for month in range(1, 13):
        month_name = f'month_{month:02d}'
        df_build[month_name] = (df_build['date'].dt.month == month).astype(int)
        months_columns.append(month_name)
    
    return df_build

def add_index_column(df_build):
    
    df_build.insert(0, 'index_id', range(1, len(df_build) + 1))
    df_build['index_id'] = df_build['index_id'].apply(lambda x: f'{x:05d}')
       
    return df_build

def rounding_data(df_build):

    columns_to_round           = ['open', 'high', 'low', 'close', 'adj_close']
    df_build[columns_to_round] = df_build[columns_to_round].astype(float)
    df_build['day_week']       = df_build['day_week'].astype(int)
    
    for column in columns_to_round:
      if column in df_build.columns:
          df_build[column] = df_build[column].round(4)
            
    return df_build

def sort_columns(df_build):

    month_columns = [col for col in df_build.columns if col.startswith('month_')]    
    desired_column_order = ['index_id', 'date_anio', 'date', 'day_week', 'close', 'open', 'high', 'low', 'adj_close', 'volume'] + month_columns
    df_build = df_build[desired_column_order]
    
    return df_build

def diff_series(series, diff):
    
    diff = diff
    diff_series = series.diff(periods=diff)
    
    return diff_series

def filter_data_by_date_range(df, filter_start_date, filter_endin_date):
        
    return df[(df['date'] >= filter_start_date) & (df['date'] <= filter_endin_date)]

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def class_weights(y_train):
    
    c0, c1 = np.bincount(y_train)   
    w0 = (1/c0) * (len(y_train)) / 2
    w1 = (1/c1) * (len(y_train)) / 2
        
    return {0: w0, 1: w1}
          
def tests_results(symbol, MAES, start_train, start_tests, endin_tests,n_estimatorss,max_depths,min_samples_sp,min_samples_lf, tests_accuracy):
    
    return {
        'symbol': symbol,
        'MAES': MAES,
        'Start_tests': start_tests[0],
        'endin_tests': endin_tests[0],
        'n_estimators': n_estimatorss,
        'Max_depths': max_depths,
        'min_samples_sp':min_samples_sp,
        'min_samples_lf': min_samples_lf,
        'tests_accuracy': tests_accuracy,
    }


def backs_results(symbol,MAES,forrest_comb, initial_capital,last_capital,total_return_buy_hold,return_strategy,n_operations,rent_op_mean, sharpe_ratio_st,max_drawdown_st):
    
    return {
        'symbol':symbol,
        'MAES':MAES,
        'forrest':forrest_comb,
        'initial_capital': initial_capital,
        'last_capital': round(last_capital, 0),  
        'return_buy_hold': round(total_return_buy_hold, 1),
        'return_strategy': round(return_strategy, 1),
        'n_operations': n_operations,
        'rent_op_mean': round(rent_op_mean,1),
        'sharpe_ratio_st': round(sharpe_ratio_st,1),
        'max_drawdown_st': round(max_drawdown_st,1)
    }


def feature_importance(model, X_train):

    importances = pd.DataFrame({'Importance': model.feature_importances_ * 100}, index=X_train.columns)
    importances_sorted = importances.sort_values(by='Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importances_sorted.index, importances_sorted['Importance'])
    plt.title('Feature Importance')
    plt.xlabel('Importance (%)')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()


def df_plots(x, y, x_label, y_label,plot_style):
    
    plt.figure(figsize=(10, 6))
    
    if plot_style == "lines":
        plt.plot(x, y, label=f'{x_label} vs {y_label}')  # Use a line plot
    elif plot_style == "points":
        plt.scatter(x, y, label=f'{x_label} vs {y_label}', marker='o')  # Use a scatter plot with markers
    else:
        raise ValueError("Invalid plot_style. Use 'lines' or 'points'.")

    plt.title(f'{x_label} vs {y_label} Plot')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()
            
def plots_histograms(dataframe, columns_of_interest):
    bins = 30
    figsize = (15, 5)
    
    # plot
    fig, axes = plt.subplots(nrows=1, ncols=len(columns_of_interest), figsize=figsize)
    
    # Columns
    for i, column in enumerate(columns_of_interest):
        axes[i].hist(dataframe[column], bins=bins, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Histogram de {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('frequency')
    
    # design
    plt.tight_layout()
    plt.show()
    
    
