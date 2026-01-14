import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import xlrd

file = xlrd.open_workbook("xxxx.xlsx") # input data path

def get_data(id = 0,x_index = -1,y_index = -1):
    x,y = [],[]

    pos_id = -1  # record the first positive sample index
    neg_id = -1  # record the first negative sample index

    sheet = file.sheet_by_index(id) 
    for i in range(1,sheet.nrows):  
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])

        if pos_id == -1 and user[y_index] == 1: pos_id = i
        if neg_id == -1 and user[y_index] == 0: neg_id = i

    return np.array(x),np.array(y),pos_id,neg_id

def get_var_name(id = 0):
    sheet = file.sheet_by_index(id)
    return list(sheet.row_values(0))

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    
    var_name = ['xxx'] # outcome variables

    model_names = ['rf','extra_tree',
                   'rotation','rf',
                   'extra_tree',
                   'extra_tree','rf','rf',]
    
    for i,(model_name,vars) in enumerate(zip(model_names,var_name)):

        features = get_var_name(0)[:-1]

        print(vars)

        X,y,pos_index,neg_index = get_data(id = 0)

        model = joblib.load('xxxxx.pkl') # model path

        pred = model.predict(X)
        pos_index = np.where(pred == 1)[0][0]

        explainer = shap.ExactExplainer(model.predict,X)
        shap_values = explainer(X)

        # draw the first positive sample force figure
        shap.force_plot(shap_values.values.mean(),np.reshape(shap_values.values[pos_index],(1,-1)), 
                                    X[pos_index], feature_names=features,matplotlib=True,show=False)
        
        plt.savefig(vars + "_force_plot_positive.pdf", format='pdf', bbox_inches='tight')
        plt.show()

        # draw the first negative sample force figure

        shap.force_plot(shap_values.values.mean(),np.reshape(shap_values.values[neg_index],(1,-1)), 
                                    X[neg_index], feature_names=features,matplotlib=True,show=False)
        
        plt.savefig(vars + "_force_plot_negative.pdf", format='pdf', bbox_inches='tight')
        plt.show()

            
            
        