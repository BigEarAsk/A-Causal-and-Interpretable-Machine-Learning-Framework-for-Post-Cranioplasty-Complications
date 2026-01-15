import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import xlrd

file = xlrd.open_workbook('xxxx.xlsx') # input data path

def get_var_name(id = 0):  # get all features' names
    sheet = file.sheet_by_index(id)
    return list(sheet.row_values(0))

def get_data(id = 0,x_index = -1,y_index = -1):
    x,y = [],[]

    pos_id = -1  # record the first positive sample index
    neg_id = -1  # record the first negative sample index

    sheet = file.sheet_by_index(id) 
    var_name = list(sheet.row_values(0))[:-1]
    for i in range(1,sheet.nrows):  
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])

        if pos_id == -1 and user[y_index] == 1: pos_id = i
        if neg_id == -1 and user[y_index] == 0: neg_id = i

    return np.array(x),np.array(y),pos_id,neg_id,var_name

plt.rcParams['figure.figsize']=(19.6, 12.8)

if __name__ == '__main__':

    var_name = ['xxxx'] # outcome variables
    
    var_name = ['total']

    model_name = ['rf','extra_tree',
                   'rotation','rf',
                   'extra_tree',
                   'extra_tree','rf','rf',]
    
          
    for i,(vars,models) in enumerate(zip(var_name,model_name)):

        print('-'*6 + vars + ":" + '-'*6)
        
        model = joblib.load('xxxx.pkl') # model path

        X,y,pos_id,neg_id,feature_name = get_data(id = 0)  # get data

        # compute shap values
        if models in ['rf','extra_tree','gbdt']:

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[[pos_id,neg_id]])
            shap_values = np.array(shap_values)[1]

        else:
            explainer = shap.KernelExplainer(model.predict,X[[pos_id,neg_id]])
            shap_values = explainer.shap_values(X[[pos_id,neg_id]])
            shap_values = np.array(shap_values)
        
        if models == 'rotation':
            shap.decision_plot(explainer.expected_value, shap_values[0], X[pos_id], feature_names=feature_name,show=False)
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(vars + '_decision_positive.pdf',bbox_inches = 'tight', dpi=500)
            plt.show()

            shap.decision_plot(explainer.expected_value, shap_values[1], X[neg_id], feature_names=feature_name,show=False)
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(vars + '_decision_negative.pdf',bbox_inches = 'tight', dpi=500)
            plt.show()

        else:
            shap.decision_plot(explainer.expected_value[1], shap_values[0], X[pos_id], feature_names=feature_name,show=False)
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(vars + '_decision_positive.pdf',bbox_inches = 'tight', dpi=500)
            plt.show()

            shap.decision_plot(explainer.expected_value[1], shap_values[1], X[neg_id], feature_names=feature_name,show=False)
            plt.rcParams['pdf.fonttype'] = 42
            plt.savefig(vars + '_decision_negative.pdf',bbox_inches = 'tight', dpi=500)
            plt.show()
        
        


