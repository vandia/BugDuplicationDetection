'''
Created on Mar 5, 2017

@author: vandia
'''
import numpy as np
import pandas as pd 
import sys

def tolower(dataset):
    for i in dataset.columns:
        if type(dataset[i].iloc[0])==str :
            dataset[i]=dataset[i].str.lower()
            dataset
    return dataset

def clean_data(dataset):
    dataset.drop_duplicates(inplace = True)  #drop duplicate rows
    dataset.dropna(axis=0, thresh = 1/3, inplace = True) #drop rows with more than 1/10 null values
    return dataset
    
def drop_columns(dataset, list_columns):
    dataset.drop(labels=list_columns, axis=1, inplace = True) #drop columns with more nulls than the threshold
    return dataset

def fill_boolean_columns(dataset, columns, value=None):
    value = False if value==None else value
    dataset.fillna(value, inplace = True)
    return dataset  

def fill_nan_values (dataset, columns):
    if type(columns) == list:
        for i in columns:
            mean = int(round(dataset[dataset[i] != 0][i].mean()))
            dataset[i] = dataset[i].fillna(mean)
        return dataset
    elif type(columns) == str:
        mean = int(round(dataset[dataset[columns] != 0][columns].mean()))
        dataset[columns] = dataset[columns].fillna(mean, inplace=True)

    return dataset

def fill_zero_values(dataset, columns):
    if type(columns) == list:
        for i in columns:
            mean = int(round(dataset[dataset[i] != 0][i].mean()))
            dataset.replace(0,mean,inplace = True)
        return dataset
    elif type(columns) == dict:
        for i in columns.keys():
            dataset[i].replace(0, np.NaN, inplace = True) #groupby ignores NAN              
            for j in columns[i]:
                dataset[j].replace('0', np.NaN, inplace = True) 
                dataset[j].replace(0, np.NaN, inplace = True)           
            data = dataset.groupby(columns[i])[i]
            dataset[i] = data.transform(lambda x: x.fillna(round(x.mean())))
                
    elif type(columns) == str:
        mean = int(round(dataset[dataset[columns] != 0][columns].mean()))
        dataset.replace(0,mean,inplace = True)
        
    return dataset

def fill_categorical_values(dataset, columns):
    if type(columns) == list:
        for i in columns:
            dataset = dataset[i].fillna(dataset[i].value_counts().iloc[0])
        return dataset
    elif type(columns) == dict:
        for i in columns.keys(): 
            dataset[i].replace('0', None, inplace = True) 
            for j in columns[i]:
                dataset[j].replace('0', None, inplace = True)   
                dataset[j].replace(0, None, inplace = True)      
            data = dataset.groupby(columns[i])[i]
            try:
                dataset[i] = data.transform(lambda x: x.fillna(x.value_counts().index[0] if x.value_counts().size > 0 else np.NaN))
            except:
                print("The column "+i+" couldn't be filled"), sys.exc_info()[1]
                   
    elif type(columns) == str:
        dataset = dataset[columns].fillna(dataset[columns].value_counts().index[0])
        
    return dataset


def modify_column_types(dataset,types):
    return dataset.astype(types, copy=False, raise_on_error=False)