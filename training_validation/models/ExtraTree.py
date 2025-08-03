# 导入所需的库
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import xlrd,random
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
    depth = [None,None,None,20,None,None,20,None]
    min_spl = [2,2,2,2,2,5,2,2]
    n = [50,200,200,100,200,100,200,50]
    fea = ['log2' if i in [1,3,4,6] else 'sqrt' for i in range(8)]

    for i in range(1):

        X,y = add_noise_to_features(path=path,id = i)
    
        model = ExtraTreesClassifier(
            max_depth=depth[i],
            max_features=fea[i],
            min_samples_leaf=1,
            min_samples_split=min_spl[i],
            n_estimators=n[i]
        )

        # train model
        model.fit(X, y)

        save_model(model,'model_parameters/'+f'extra_tree_{var_name[i]}')

        print('---------------------------------')



if __name__ == "__main__":

    set_seed()
    train()
