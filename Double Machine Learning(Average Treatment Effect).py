# 导入必要的库
import random
import numpy as np
import pandas as pd
from doubleml import DoubleMLData, DoubleMLPLR
from econml.dml import LinearDML,NonParamDML
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

np.random.seed(1000)
random.seed(1000)

def get_data(path, id, t_name, y_name):
    df = pd.read_excel(path, id)

    T = df[t_name]
    Y = df[y_name]
    X = df.drop(columns=[t_name, y_name])

    return X, T, Y


if __name__ == '__main__':

    T_names = ['Surgery.time','N.P.drainage','Titanium']

    Y_names = ['Infection', 'Fluid.collections', 'Pneumocephalus', 'Intracranial.hemorrhage', 
               'Hydrocephalus', 'Seizures','Total.overall.complications', 'Reoperations']
    
    models = ['rf','extra_tree','rotation','rf','extra_tree','extra_tree','rf','rf']

    path = 'xxxx.xlsx'
    
    name_id = 2
    for i in range(1):

        X,T,Y = get_data(path,i,T_names[name_id],Y_names[i],0,pca_flag=True)

        # n_estimators=100,learning_rate=0.005,loss='absolute_error'  ->  Titanium && NP
        # n_estimators=100,learning_rate=0.5,loss='quantile' -> Surgery.time

        # n_estimators=100,learning_rate=0.5,loss='absolute_error',min_samples_leaf=10,max_features='log2'  -> NP
        # n_estimators=100,learning_rate=0.001,loss='squared_error',min_samples_leaf=12,max_features=0.1 -> Titanium
        model_T = GradientBoostingClassifier(n_estimators=100,learning_rate=0.001,loss='squared_error',min_samples_leaf=12,max_features=0.1)
        model_Y = GradientBoostingClassifier(n_estimators=100,learning_rate=0.001,loss='squared_error',min_samples_leaf=12,max_features=0.1)

        # surgery.time:
        dml = LinearDML(model_y=model_Y, model_t=model_T,discrete_treatment=True)
        
        dml.fit(Y,T,X=X)

        # if variable is binary,  T0 = 0,T1 = 1
        # Surgery.time -> T0 = T,T1 = T+60/77.6
        
        if T_names[name_id] == 'Surgery.time':
            ate = dml.ate(X,T0 = T,T1 = T+60/77.6)
            ate_ci = dml.ate_interval(X,T0 = T,T1 = T+60/77.6)
        else:
            ate = dml.ate(X,T0 = 0,T1 = 1)
            ate_ci = dml.ate_interval(X,T0 = 0,T1 = 1,alpha=0.05)

        print(f"{Y_names[i]} vs {T_names[name_id]} ATE : {ate:.4f}")
        print(f"{Y_names[i]} vs {T_names[name_id]} ATE 95%CI: ({ate_ci[0]:.4f},{ate_ci[1]:.4f})")
