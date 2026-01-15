import joblib
from matplotlib import rcParams
import shap
import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = xlrd.open_workbook("xxxxx.xlsx") # input data path

def get_var_name(id = 0):  # get all features' names
    sheet = file.sheet_by_index(id)
    return list(sheet.row_values(0))

def get_data(id = 0,x_index = -1,y_index = -1):  # get data
    x,y = [],[]

    sheet = file.sheet_by_index(id) 
    for i in range(1,sheet.nrows):  
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])

    return np.array(x),np.array(y)

# draw combine figure
def draw_combine(dir = 'xxxx',var_name = '',shap_values=None,sign = True):
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1200)

    if sign: shap_values = shap_values[1]

    shap.summary_plot(shap_values, X, feature_names=get_var_name(id=i)[:-1], plot_type="dot", show=False, color_bar=True)
    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
    ax1 = plt.gca()
    ax2 = ax1.twiny()

    # draw bar
    shap.summary_plot(shap_values, X, feature_names=get_var_name(id=i)[:-1],plot_type="bar", show=False)

    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
    ax2.axhline(y=13, color='gray', linestyle='-', linewidth=1)
    bars = ax2.patches

    for bar in bars:    
        bar.set_alpha(0.2)

    # set lables and style
    ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=12)
    ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=12)
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax1.set_ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.rcParams['pdf.fonttype'] = 42
    # save figure
    plt.savefig(dir + var_name + "_SHAP_combined_0728.pdf", format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    var_name = ['xxxx'] # outcome variables

    model_names = ['rf','rf',
                   'rotation','logit',
                   'gam',
                   'logit','rf',]

    for i,(model_name,vars) in enumerate(zip(model_names,var_name)):
        
        if vars not in ['Seizures']: continue

        print(model_name)

        X,y = get_data(id = i)

        model = joblib.load('xxxxx.pkl') # model path

        if model_name in ['rf','extra_tree','gbdt']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap_values = np.array(shap_values)

            draw_combine(var_name=vars,shap_values=shap_values)
            
        else:
            explainer = shap.KernelExplainer(model.predict,X,shap.kmeans(X,10))
            shap_values = explainer.shap_values(X)
            shap_values = np.array(shap_values)

            draw_combine(var_name=vars,shap_values=shap_values,sign=False)
           

            

