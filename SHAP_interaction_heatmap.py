import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import xlrd

file = xlrd.open_workbook('xxxx.xlsx') # input data path

def get_var_name(id = 0):  # get all features' names
    sheet = file.sheet_by_index(id)
    return list(sheet.row_values(0))

def get_data(id = 0,x_index = -1,y_index = -1):  # get data
    
    x,y = [],[]
    
    sheet = file.sheet_by_index(id)
    var_name = list(sheet.row_values(0))[:-1]
    for i in range(1,sheet.nrows):  
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])

    return np.array(x),np.array(y),var_name


if __name__ == '__main__':

    var_name = ['xxx'] # outcome path
    
    model_name = ['rf','extra_tree',
                   'rotation','rf',
                   'extra_tree',
                   'extra_tree','rf','rf',]
    
    for i,(vars,models) in enumerate(zip(var_name,model_name)):

        print('-'*6 + vars + ":" + '-'*6)
        
        model = joblib.load('xxxx.pkl') # model path

        X,y,feature_name = get_data(id = i) 

        if models in ['rf','extra_tree','gbdt']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X)[1]
        
        else:
            explainer = shap.KernelExplainer(model.predict,X,shap.kmeans(X,10))
            shap_values = explainer(X)
        

        shap.plots.heatmap(shap_values, show=False,)
        plt.savefig(vars+'_shap_heatmap.pdf',bbox_inches = 'tight', dpi=500)
        plt.show()
