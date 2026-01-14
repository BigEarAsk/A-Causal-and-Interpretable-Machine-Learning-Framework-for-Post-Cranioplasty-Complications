import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import xlrd

file = xlrd.open_workbook("xxxx.xlsx") # input data path

def get_data(id = 0,x_index = -1,y_index = -1):
    x,y = [],[]

    sheet = file.sheet_by_index(id) 
    for i in range(1,sheet.nrows): 
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])

    return np.array(x),np.array(y)

def get_var_name(id = 0):
    sheet = file.sheet_by_index(id)
    return list(sheet.row_values(0))


if __name__ == '__main__':

    var_name = ['xxxx'] # outcome variables
    
    model_names = ['rf','extra_tree',
                   'rotation','rf',
                   'extra_tree',
                   'extra_tree','rf','rf',]
    
    for i,(model_name,vars) in enumerate(zip(model_names,var_name)):

        features = get_var_name(i)[:-1]

        print(model_name)

        X,y = get_data(id = i)

        model = joblib.load('xxxx.pkl') # model path

        
        if model_name == 'rotation':
            explainer = shap.ExactExplainer(model.predict,X)
            shap_values = explainer(X)
        else:
            explainer = shap.Explainer(model,X)
            shap_values = explainer(X,check_additivity=False)

        shap.initjs()
        
        if model_name == 'rotation':
            force_plot = shap.force_plot(shap_values.values.mean(),shap_values.values, 
                                     X, feature_names=features)
        else:
            force_plot = shap.force_plot(explainer.expected_value[1], shap_values.values[:,:,1], 
                                     X, feature_names=features)
        

        html_file = vars + "_force_plot_all.html"
        shap.save_html(html_file, force_plot)
