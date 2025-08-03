from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import xlrd,joblib,random
import numpy as np
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
    n_com = [7,5,7,7,7,7,5,7]
    depth = [7 for i in range(8)]
    n = [200,200,50,200,50,100,100,200]

    for i in range(1):

        print(var_name[i]+":")
        X,y = add_noise_to_features(path=path,id = i)

        model = Pipeline([
            ('pca', PCA(n_components=n_com[i])),  
            ('clf', RandomForestClassifier(n_estimators=n[i], 
                        max_depth=depth[i],random_state=42))  
        ])

        # train model 
        model.fit(X, y)

        save_model(model,'model_parameters/'+f'rotation_{var_name[i]}')

        print('---------------------------------')

if __name__ == '__main__':
    set_seed()
    train()

