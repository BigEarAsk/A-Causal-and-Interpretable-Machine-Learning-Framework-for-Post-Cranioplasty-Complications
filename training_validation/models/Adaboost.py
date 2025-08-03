import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import xlrd,random
import joblib
from var_name import var_name,path
from add_noise import add_noise_to_features


def get_data(path = 'Hyperparameter/Derivation Cohort(After Genetic Algorithm).xlsx',
             id = 0,x_index = -1,y_index = -1): # get data 
    x,y = [],[]
    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(id)
    for i in range(1,sheet.nrows): 
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])

    return np.array(x),np.array(y)

def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)


def save_model(model,name):  # save model
    joblib.dump(model,name+'.pkl',compress=3)

def train():  # train model
    lr = [1,1.,1.,1,1,1,.5,.5]
    n = [200,200,200,200,200,200,50,200]

    for i in range(1):

        print(var_name[i]+":")
        
        X,y = add_noise_to_features(path = path,id = i)
            
        # set model
        model = AdaBoostClassifier(learning_rate=lr[i],n_estimators=n[i])

        # train model
        model.fit(X, y)

        save_model(model,'model_parameters/'+f'ada_{var_name[i]}')
        print('---------------------------------')


if __name__ == '__main__':
    set_seed()
    train()
