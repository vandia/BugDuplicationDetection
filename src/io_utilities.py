'''
Created on Mar 5, 2017

@author: vandia
'''
import numpy as np
import pandas as pd
import csv

def load_and_join_files(fnames, index=None, converters=None, how=None):
    na_values = ["None", "none"]
    h = "outer" if how == None else how
    response=pd.DataFrame()
    for s in fnames:
        df=pd.read_csv(s, index_col = index, converters=converters, infer_datetime_format=True, na_values=na_values, keep_default_na=True)
        response=df.join(response,how=h)
    return response


def load_and_join(fname1, fname2, index=None, converters=None, date_cols1=None, date_cols2=None, how=None):
    na_values=["None","none"]
    if (date_cols1 != None):
        dataset1 = pd.read_csv(fname1, index_col = index, converters=converters, parse_dates=date_cols1, infer_datetime_format=True, 
                               na_values=na_values, keep_default_na=True)
    else:
        dataset1 = pd.read_csv(fname1, index_col = index, converters=converters, na_values=na_values, keep_default_na=True)
        
    if (date_cols2 != None):
        dataset2 = pd.read_csv(fname2, index_col=index, converters=converters, parse_dates=date_cols2, infer_datetime_format=True,
                               na_values=na_values, keep_default_na=True)
    else:
        dataset2 = pd.read_csv(fname2,index_col=index, converters=converters, na_values=na_values, keep_default_na=True)

    h= "outer" if how == None else how
    
    dataset3 = dataset1.join(dataset2, how=h)
    return dataset3

def load (fname, index=None, date_cols=None, converters=None):   
    na_values=["None","none"] 
    dataset = pd.read_csv(fname, index_col = index, parse_dates=date_cols, infer_datetime_format=True, converters=converters, 
                          na_values=na_values, keep_default_na=True)
    return dataset

def save_csv (dataset, fname, index):
    dataset.to_csv(path_or_buf=fname, sep=",", mode="w", index=index, float_format='%.8f')

def get_weka_type(dataset, col):
    t=type(dataset[col].iloc[0]) 
    if dataset[col].dtype.name == 'category':              
        res="{"
        categ=dataset[col].cat.categories
        l=len(categ)
        for j in range(l):
            res+="'"+str(categ[j])+"'"
            if (j != (l-1)):
                res+=", "
        return res+"}"
    elif t == np.float_ or t == float or t == int or t== np.int_: 
        return "NUMERIC" 
    elif t == pd.tslib.Timestamp:
        return'DATE "yyyy-MM-dd"'
    elif t == bool or t==np.bool_:
        return "{True, False}"
    else:
        return'STRING'
    
    

def generate_arff(fname,name,dataset, tr_dataset=None):
    
    headers=dataset.columns.values.tolist() if tr_dataset is None else tr_dataset.columns.values.tolist()
    head_dataset=dataset if tr_dataset is None else tr_dataset

    with open(fname, "w") as ofile:
        ofile.write("@relation '"+name+"' \n")      
        for i in headers:
            ofile.write("@attribute "+i+"    "+get_weka_type(head_dataset,i)+"\n")       
        ofile.write("@data\n")   
    
  
    dataset.to_csv(path_or_buf=fname,index=False, header=None, sep=",", mode="a",quoting=csv.QUOTE_NONNUMERIC, na_rep='?')
    
    
