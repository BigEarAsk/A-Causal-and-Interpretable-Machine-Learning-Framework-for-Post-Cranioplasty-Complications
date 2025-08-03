from sklearn import tree 
import random
import numpy as np
import xlrd
import joblib
from var_name import var_name,path
from add_noise import add_noise_to_features

def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)

def get_data(path,id = 0,x_index = -1,y_index = -1):
    x,y = [],[]
    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(id)
    for i in range(1,sheet.nrows):  
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])

    return np.array(x),np.array(y)

def save_model(model,name):  
    joblib.dump(model,name+'.pkl',compress=3)

def train():

    cri = ['gini' if i in [0,1,3,5] else 'entropy' for i in range(8)]
    depth = [None,10,15,15,None,15,5,None]
    min_leaf = [1,1,1,1,1,1,2,1]

    for i in range(1):

        print(var_name[i]+":")

        X,y = add_noise_to_features(path=path,id = i)

        model = tree.DecisionTreeClassifier(criterion=cri[i],
                                            min_samples_leaf=min_leaf[i],
                                            max_depth=depth[i])

        # train model 
        model.fit(X, y)

        save_model(model,'model_parameters/'+f'dt_{var_name[i]}')

        print('---------------------------------')

if __name__ == '__main__':
    set_seed()
    train()


