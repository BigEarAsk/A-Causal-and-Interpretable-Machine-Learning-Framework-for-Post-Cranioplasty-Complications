import joblib
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import fsolve
import pandas as pd
import shap
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

def draw(X,fea_id,shap_values_df,feature,var_name):

    # draw scatter
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(X[:,fea_id], shap_values_df[feature], s=20, label='SHAP values', alpha=0.7)
   
    plt.axhline(y=0, color='black', linestyle='-.', linewidth=1, label='SHAP = 0')
    # add legend
    plt.legend()

    if feature in ['Pre.op.V.P','Pre.op.seizures','Pre.op.infection','Titanium']:
        plt.xlim(-0.3,1.3)
    # set title and label
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('SHAP value for\n' + feature, fontsize=12)
    # delete upper and righter border
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.rcParams['pdf.fonttype'] = 42

    # save figure
    plt.savefig(var_name + "_" + feature + " SHAP Dependence Plot_with_Multiple_Intersections.pdf")


if __name__ == '__main__':
    
    var_name = ['xxxx'] # outcome variables
    
    model_name = ['rf','extra_tree',
                   'rotation','rf',
                   'extra_tree',
                   'extra_tree','rf','rf',]
    
    features_to_plot = {
        'Fluid':['Skull.defect.area','Surgery.time','N.P.drainage'],
        'Hydro':['GCS','Pre.op.V.P','Skull.defect.area'],
        'Seizures':['Pre.op.seizures','GCS','Titanium'],
        'intra_hem':['GCS','Skull.defect.area','Surgery.time'],
        'Reop':['GCS','Surgery.time','Skull.defect.area'],
        'total':['Surgery.time','Skull.defect.area','GCS'],
        'infection':['Surgery.time','Pre.op.infection','DC.CP.interval'],
        'Penu':['Surgery.time','Titanium','Age']
    }

    for i,(vars,models) in enumerate(zip(var_name,model_name)):

        print('-'*6 + vars + ":" + '-'*6)
        
        model = joblib.load('xxxx.pkl') # model path

        feature_plot = features_to_plot[vars]   # get features to be plot

        X,y,feature_name = get_data(id = 0)                 

        fea_id1 = feature_name.index(feature_plot[0])    # get index of feature1
        fea_id2 = feature_name.index(feature_plot[1])    # get index of feature2
        fea_id3 = feature_name.index(feature_plot[2])    # get index of feature3

        shap_values = None

        if models in ['rf','extra_tree','gbdt']:

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap_values = np.array(shap_values)[1]

        else:
            explainer = shap.KernelExplainer(model.predict,X,shap.kmeans(X,10))
            shap_values = explainer.shap_values(X)
            shap_values = np.array(shap_values)
            sign = False
        
        values_df = pd.DataFrame(shap_values,columns=feature_name)

        # draw 
        draw(X,fea_id1,values_df,feature_plot[0],vars)
        draw(X,fea_id2,values_df,feature_plot[1],vars)
        draw(X,fea_id3,values_df,feature_plot[2],vars)
