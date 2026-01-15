import xlrd
import numpy as np
import pandas as pd
from openpyxl import load_workbook

# input data path
path = 'xxxx.xlsx'

file = xlrd.open_workbook(path)
sheet = file.sheet_by_index(0)
index_del = []

def get_var_name():  # get all features' names
    return list(sheet.row_values(0))[1:]

def get_sum(col):  # sum data of the given column
    data = sheet.col_values(col)[1:]
    data = np.array(data)
    data = np.abs(data)
    return np.sum(data)

def get_id():  # get the index of maximum coef

    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(0)
    index = (1,1)
    maxr = 0

    for i in range(1,sheet.nrows):
        if i in index_del: continue
        for j in range(i+1,sheet.ncols):
            if j in index_del: continue
            if sheet.cell_value(i,j) > maxr:
                maxr = sheet.cell_value(i,j)
                index = (i,j)
    
    return maxr,index

def eliminate(threshold = 0.6):  # eliminate the colinear variables
    maxr,index = get_id()
    
    while maxr > threshold:
        # delete：
        sum1 = get_sum(index[0])
        sum2 = get_sum(index[1])

        id = 0 if sum1 > sum2 else 1

        index_del.append(index[id])

        maxr,index = get_id()

def del_ws_rows_cols(rowd):  # delete given rows and columns

    wb = load_workbook(path)
    ws = wb['Sheet1']   
    rowd = sorted(rowd, reverse=True)  # reverse the rows array
    
    for r in rowd:                     
        ws.delete_rows(r+1)
        ws.delete_cols(r+1)
    
    wb.save('Data(After Removing Multicolinearity).xlsx')  # save the data 
                              
if __name__ == '__main__':
    eliminate()
    print(index_del)
    del_ws_rows_cols(index_del)
