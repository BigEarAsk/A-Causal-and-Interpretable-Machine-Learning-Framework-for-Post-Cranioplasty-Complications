from dython.nominal import associations
import numpy as np
import xlrd
import pandas as pd


if __name__ == '__main__':
    # read data
    data = pd.read_excel('xxxx.xlsx',0)
    
    # get theil-U correlation matrix
    # [1,2,3,4,5,6,7,8,9,10,11,12,14,16,17,18,23] is the column_id of discrete variables
    res = associations(data[data.columns[
        [1,2,3,4,5,6,7,8,9,10,11,12,14,16,17,18,23]
    ]],nom_nom_assoc = 'theil',annot=False,vmin=0.0,plot=False)
    
    df = res['corr']

    # save data 
    df.to_excel('dis_vs_dis.xlsx')
