#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 20:20:38 2024

@author: javi
"""
import pandas as pd
import numpy as np


def MAV(df_preprocess, n):

    
    MAV = pd.Series(df_preprocess['close'].rolling(n,min_periods=1).mean(), name='MAV_' + str(n))


    return MAV

def ROC(df_preprocess, n):
    

    M = df_preprocess.diff(n - 1)
    N = df_preprocess.shift(n - 1)
    
    ROC = pd.Series(((M / N)) * 100, name='ROC_' + str(n))

    return ROC


def MOM(df_preprocess, n):
    
    MOM = pd.Series(df_preprocess.diff(n), name = 'MOM_' + str(n))
    
    return MOM


def EMA(df_preprocess, n):
    
    EMA = pd.Series(df_preprocess['close'].ewm(span=n).mean(), name='EMA_' + str(n))
    
    return EMA

def STK(close, low, high, n):
    
    STK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    
    return STK

def STD(close, low, high, n):
    
    STK = ((close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min())) * 100
    STD = STK.rolling(3).mean()
    
    return STD


def RSI(series, period):
    
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period - 1]] = np.mean(u[:period])  # El primer valor es la suma de las ganancias promedio
    u = u.drop(u.index[:(period - 1)])
    d[d.index[period - 1]] = np.mean(d[:period])  # El primer valor es la suma de las pérdidas promedio
    d = d.drop(d.index[:(period - 1)])
    rs = u.ewm(com=period - 1, adjust=False).mean() / d.ewm(com=period - 1, adjust=False).mean()

    RSI = 100 - 100 / (1 + rs)
    
    
    return RSI




    