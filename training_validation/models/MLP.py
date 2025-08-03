import random
from sklearn.neural_network import MLPClassifier
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

def train():

    for i in range(1):

        print(var_name[i]+":")
        X,y = add_noise_to_features(path=path,id = i)

        model = MLPClassifier(hidden_layer_sizes=(100, 50),  
                        activation='relu',  
                        solver='adam',  
                        max_iter=1000, 
                        random_state=42)  


        # train model 
        model.fit(X, y)

        save_model(model,'model_parameters/mlp_'+var_name[i])

        print('---------------------------------')


def save_model(model,name):  # save model
    joblib.dump(model,name+'.pkl',compress=3)

if __name__ == '__main__':
    set_seed()
    train()
