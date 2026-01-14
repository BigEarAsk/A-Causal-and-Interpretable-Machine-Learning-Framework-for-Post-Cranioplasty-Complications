import numpy as np
import xlrd,xlwt
import random

# set input data path
paths = ['xxxx.xlsx']


def get_var_name(path):  #  get all features' names
    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(0)

    return list(sheet.row_values(0))

def get_num(arr,cls): # get the number of the given class
    arr = np.array(arr)
    return np.sum(arr == cls)
 
def get_index(arr,cls): # get the index of the given class
    arr = np.array(arr)
    return np.where(arr == cls)[0]

def get_in_avg(arr,arr2,cls): # get the in-class average value 
    # arr2  continous arrays
    
    index = get_index(arr,cls)
    l = get_num(arr,cls)
    
    if l == 0: return 0

    avg = 0
    arr2 = np.array(arr2)
    for data in arr2[index]:
        avg += data / l
    return avg

def get_whole_avg(arr): # get total average value
    # arr: continous arrays
    avg = 0
    for i in arr:
        avg += i / len(arr)
    
    return avg

def get_weighted_res(in_avg,whole_avg,arr,cls): # get weighted value
    # cls array 
    # arr: discrete variables array
    # whole_avg  array
    # in_avg array

    res = 0
    for _ in list(zip(in_avg,whole_avg,cls)):
        in_,wh_,c = _
        n = get_num(arr,c)
        res += n * ((in_ - wh_) ** 2)  # (in_class - whole)^2 * class_num
    return res

def get_mean_square_sum(arr,whole_avg):  #  get the MSE
    # arr  continous array
    # whole_avg  single value
    res = 0
    for i in arr:
        res += (i - whole_avg) ** 2
    return res

def get_cor(arr1,arr2,cls):
    # arr1 discrete array
    # arr2 continous array
    # cls classes array

    in_avg_arr = []
    for c in cls:
        in_avg = get_in_avg(arr1,arr2,c) # in-class
        in_avg_arr.append(in_avg)

    whole_avg = get_whole_avg(arr2) # whole
    whole_avg_arr = [whole_avg] * len(arr2) # total average array
    weighted = get_weighted_res(in_avg_arr,whole_avg_arr,arr1,cls) # weighted error
    res = get_mean_square_sum(arr2,whole_avg) # MSE

    return pow(weighted / res,0.5)

def get_row_data(path,x_index = list(range(25))): # get data by rows
    # x_index  array of independent variables
    x = []
    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(0)
    if x_index == 'all':
        all = True
    for i in range(1,sheet.nrows):  # data rows
        info = list(sheet.row_values(i))
        user = []
        for id in x_index:
            user.append(info[id])
        x.append(user)
    return x

def get_col_data(path,x_index):  # get data by columns
    # x_index array of independent variables
    x = []
    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(0)

    for id in x_index:
        x.append(list(sheet.col_values(id))[1:])

    return x

def get_cor_matrix(path,con_id,dis_id,cls_arr): # get correlation matrix
    # con_id:   array of continous index
    # dis_id:  array of discrete index
    # cls_arr:  array of discrete variable class

    con_arr = get_col_data(path,con_id) #  column data of continuos variables
    dis_arr = get_col_data(path,dis_id) #  column data of discrete variables

    cor_matrix = []

    for con in con_arr:
        cor = []
        for (dis,cls) in zip(dis_arr,cls_arr):
            cor.append(round(get_cor(dis,con,cls),3))
        cor_matrix.append(cor)
    return cor_matrix

def save_data(path,data,con_id,dis_id):
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet = workbook.add_sheet('Sheet1')

    var_name = get_var_name(path)
    var_name = np.array(var_name)

    for r,name in zip(range(len(con_id)),var_name[con_id]):
        sheet.write(r+1,0,name)

    for c,name in zip(range(len(dis_id)),var_name[dis_id]):
        sheet.write(0,c+1,name)

    for r in range(len(data)):
        for c in range(len(data[0])):
            sheet.write(r+1,c+1,data[r][c])

    workbook.save('dis_vs_con.xls')
    
if __name__ == "__main__":

    for path in paths:
        data = get_row_data(path)

        continous_id = [0,13,15,19,20,21,22]
        discrete_id = list(set(range(24)) - set(continous_id))

        not_01_info = {
            14:[1,2],
            16:[1,2,3,4],
            18:[1,2,3],
        }

        cls_arr = [[0,1] if i not in not_01_info.keys() else not_01_info[i] for i in discrete_id]
        cor_matrix = get_cor_matrix(path,continous_id,discrete_id,cls_arr)
        save_data(path,cor_matrix,continous_id,discrete_id)

