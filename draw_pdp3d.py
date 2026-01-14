import joblib
from matplotlib import rcParams
import numpy as np
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import xlrd
from pdpbox import pdp
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def get_data(path,id = 0,x_index = -1,y_index = -1):  # get data
    x,y = [],[]
    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(id)
    var_name = list(sheet.row_values(0))[:-1]
    for i in range(1,sheet.nrows):  
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])

    return np.array(x),np.array(y),var_name 

def plot_3d_pdp_scatter(model, X, features, var_name,grid_resolution=20):
    feature_1, feature_2, feature_3 = features
    feature_1_vals = np.linspace(X[feature_1].min(), X[feature_1].max(), grid_resolution)
    feature_2_vals = np.linspace(X[feature_2].min(), X[feature_2].max(), grid_resolution)
    feature_3_vals = np.linspace(X[feature_3].min(), X[feature_3].max(), grid_resolution)

    grid_1, grid_2, grid_3 = np.meshgrid(feature_1_vals, feature_2_vals, feature_3_vals)
    grid_points = np.c_[grid_1.ravel(), grid_2.ravel(), grid_3.ravel()]
    X_grid = np.zeros((grid_points.shape[0], X.shape[1]))
    X_grid[:, X.columns.get_loc(feature_1)] = grid_points[:, 0]
    X_grid[:, X.columns.get_loc(feature_2)] = grid_points[:, 1]
    X_grid[:, X.columns.get_loc(feature_3)] = grid_points[:, 2]

    preds = model.predict_proba(X_grid)[:,1]
    fig = plt.figure(figsize=(10, 8),dpi=1200)
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(grid_1.ravel(), grid_2.ravel(), grid_3.ravel(), c=preds, 
                    cmap='viridis', alpha=0.8)
    plt.colorbar(sc, ax=ax, label='Prediction')
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_zlabel(feature_3)
    rcParams['pdf.fonttype'] = 42
    plt.title(f'3D Partial Dependence Scatter Plot for {feature_1}, {feature_2}, and {feature_3}')
    plt.savefig(f"{var_name}_3D.pdf", format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    
    var_name = ['xxxx'] # outcome variables

    model_name = ['rf','extra_tree',
                   'rotation','rf',
                   'extra_tree',
                   'extra_tree','rf','rf',]

    # define which three features need to be painted for each dependent variables
    features_to_plot = {
        'Fluid':['Skull.defect.area','Surgery.time','N.P.drainage'],
        'Hydro':['GCS','Pre.op.V.P','Skull.defect.area'],
        'Seizures':['Pre.op.seizures','GCS','Titanium'],
        'intra_hem':['GCS','Skull.defect.area','Surgery.time'],
        'Reop':['GCS','Surgery.time','Skull.defect.area'],
        'Total':['Surgery.time','Skull.defect.area','GCS'],
        'infection':['Surgery.time','Pre.op.infection','DC.CP.interval'],
        'Penu':['Surgery.time','Titanium','Age']
    }

    data_path = ["xxxx.xlsx"] # input data path

    for i,(vars,models) in enumerate(zip(var_name,model_name)):

        print('-'*6 + vars + ":" + '-'*6)
        
        model = joblib.load('xxx.pkl') # model path
        feature_plot = features_to_plot[vars]

        for path in data_path:

            df = pd.read_excel(path,0)
            X = df.iloc[:,:-1]
            n_classes = 2 if vars == 'Penu' else None

            # get pdp image
            plot_3d_pdp_scatter(model,X,feature_plot,vars)
            
        print('-----------------------------------')