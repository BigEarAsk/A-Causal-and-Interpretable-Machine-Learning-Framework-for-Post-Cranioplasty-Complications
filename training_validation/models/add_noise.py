import numpy as np
import xlrd
import random

def get_data(path = 'AUC/Derivation Cohort(After Genetic Algorithm).xlsx',id = 0,x_index = -1,y_index = -1):
    x,y = [],[]
    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(id)
    for i in range(1,sheet.nrows):  
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])

    return np.array(x),np.array(y)

def add_noise_to_features(id,path = 'models/Derivation Cohort(After Genetic Algorithm).xlsx',
                          noise_level=0.02, random_state=42):
    
    np.random.seed(random_state)
    random.seed(random_state)

    # the numerical indices of each variable
    cols = [
        [0,3,4,6,7],
        [0,1,2,3,4],
        [0,1,2,4,5],
        [0,3,4,5,6],
        [0,4,5,9,10],
        [0,2,3,5,6],
        [0,3,4,8,9]
    ]

    X,y = get_data(path,id)

    for col in cols[id]:
        std = X[:,col].std()
        noise = np.random.normal(0, std * noise_level, size=len(X))
        X[:,col] += noise

    return X,y
