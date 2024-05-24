#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:21:03 2024
@author: javi
"""
import os
import pickle
from main.paths.paths import path_base,folder_save_tree


def save_model(model, model_file_name="random_forest_model.pkl"):
    
    
    model_file_name = "random_forest_model.pkl"
    model_file_path = os.path.join(path_base, folder_save_tree, model_file_name)
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Modelo guardado en: {model_file_path}")