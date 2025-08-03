import numpy as np
import lightgbm as lgb
import xlrd,random,joblib
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
    lr = [0.5,.5,.5,.5,.1,.5,.1,.5]
    n = [50,50,100,200,200,200,100,200]
    leave = [50,15,15,31,15,50,15,15]
    

    for i in range(1):

        print(var_name[i]+":")
        X,y = add_noise_to_features(path=path,id = i)

        model = lgb.LGBMClassifier(learning_rate=lr[i],
                                    num_leaves=leave[i],
                                    n_estimators=n[i])
        
        # train model 
        model.fit(X, y)

        save_model(model,'model_parameters/'+f'lgb_{var_name[i]}')

        print('---------------------------------')


if __name__ == '__main__':
    set_seed()
    train()
