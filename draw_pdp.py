import joblib
from matplotlib import rcParams
import numpy as np
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import xlrd
from pdpbox import pdp
import pandas as pd
from itertools import *

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


if __name__ == '__main__':

    var_name = ['infection','Fluid','Penu','intra_hem','Hydro','Seizures',
                'Total','Reop']
    
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
        'Total':['Surgery.time','Skull.defect.area','GCS'],
        'infection':['Surgery.time','Pre.op.infection','DC.CP.interval'],
        'Penu':['Surgery.time','Titanium','Age']
    }

    data_path = ["xxxxx.xlsx"] # input data path

    for i,(vars,models) in enumerate(zip(var_name,model_name)):

        print('-'*6 + vars + ":" + '-'*6)
        
        model = joblib.load('xxxx.pkl') # model path

        for feature_plot in combinations(features_to_plot[vars],2):
            for path in data_path:
                df = pd.read_excel(path,i)
                X = df.iloc[:,:-1]
                n_classes = 2 if vars == 'Penu' else None

                # get pdp image
                inter = pdp.PDPInteract(model,X,X.columns,feature_plot,
                            feature_plot,n_classes=n_classes).plot(dpi=100,engine='matplotlib')
                plt.title(label = vars + ' pdp',loc = 'center')
                rcParams['pdf.fonttype'] = 42
                plt.savefig(vars + f'_pdp{feature_plot}.pdf') 
                plt.show()
            
        print('-----------------------------------')