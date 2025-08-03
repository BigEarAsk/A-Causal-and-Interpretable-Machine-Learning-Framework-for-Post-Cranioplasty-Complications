import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xlrd,random,joblib
from add_noise import add_noise_to_features
from var_name import var_name,path


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
    depth = [None,20,20,None,None,20,None,None]
    fea = ['sqrt' if i in [0,1,3,4,5,6,7] else 'log2' for i in range(8)]

    min_spl = [2 if i in [2,3,4,5,6,7] else 5 for i in range(8)]
    n = [100,100,100,50,100,50,200,100]

    for i in range(1):

        print(var_name[i]+":")
        X,y = add_noise_to_features(path=path,id = i)

        model = RandomForestClassifier(
            n_estimators=n[i],
            max_depth=depth[i],
            min_samples_leaf=2,
            min_samples_split=min_spl[i], 
            max_features=fea[i],
            random_state=42)

        # train model 
        model.fit(X, y)

        save_model(model,'model_parameters/'+f'rf_{var_name[i]}')

        print('---------------------------------')

if __name__ == '__main__':

    set_seed()
    train()

