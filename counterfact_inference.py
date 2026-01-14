import random
import dice_ml
from dice_ml import Dice
import joblib
from matplotlib import rcParams
import shap
import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file = xlrd.open_workbook("xxxx.xlsx") # input data path

plt.rcParams['pdf.fonttype'] = 42

seed = 500
random.seed(seed)
np.random.seed(seed)

def get_var_name(id = 0):  # get all features' names
    sheet = file.sheet_by_index(id)
    return list(sheet.row_values(0))

def get_data(id = 0):  # get data
    x,y = [],[]

    feature_name = get_var_name(id)

    sheet = file.sheet_by_index(id) 
    for i in range(1,sheet.nrows):  
        user = sheet.row_values(i)
        x.append(user)
        y.append(user[-1])

    x = np.array(x)
    x = pd.DataFrame(x,columns=feature_name)
    return x,np.array(y)

def draw_rador(original_instance,counterfactuals_df,var_name):

    labels = counterfactuals_df.columns[:-1]
    num_vars = len(labels)

    # 创建雷达图的角度 create rador's angle
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # close the shape
    original_values = original_instance.to_numpy().flatten().tolist()
    original_values += original_values[:1]
    angles += angles[:1]

    # draw rador
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # draw original sample
    ax.fill(angles, original_values, color='#cecece', alpha=0.25, label='Original Instance')
    ax.plot(angles, original_values, color='#cecece', linewidth=2)

    print(counterfactuals_df)

    # draw counterfactuals
    for i in range(counterfactuals_df.shape[0]):
        if i == 3 or i == 4: continue
        values = counterfactuals_df.iloc[i, :-1].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, alpha = 0.8 - 0.05*i,label=f'Counterfactual {i+1}')

    # set label and title
    ax.set_yticklabels([])  # no show pole lable
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title('Comparison of Original Instance and Counterfactuals of '+ var_name)
    plt.legend(loc='upper right')
    plt.savefig(var_name + ' Comparison of Original Instance and Counterfactuals.pdf',bbox_inches = 'tight', dpi=500)
    plt.show()

def draw_heatmap2(original_values, counterfactuals_df, var_name):
    # get counterfactuals_values
    counterfactuals_values = counterfactuals_df.iloc[:, :-1].values
    labels = counterfactuals_df.columns[:-1]

    # combine original values and counterfactuals_values
    comparison_data = np.vstack([original_values, counterfactuals_values])
    comparison_df = pd.DataFrame(comparison_data, columns=labels)

    # set heatmap style
    plt.figure(figsize=(12, 8))  
    sns.set_theme(font_scale=1.2)  
    sns.set_style("whitegrid")  

    # draw heatmap
    heatmap = sns.heatmap(
        comparison_df, 
        annot=True, 
        cmap='YlGnBu',  
        cbar=True, 
        xticklabels=labels, 
        yticklabels=['Original Instance'] + [f'Counterfactual {i + 1}' for i in range(counterfactuals_values.shape[0])],
        linewidths=0.5,  
        linecolor='gray',  
        fmt='.2f'  
    )

    # add title and label
    plt.title(f'Heatmap of Original Instance and Counterfactuals for {var_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Instances', fontsize=14)

    # save figure
    plt.savefig(f'{var_name}_Heatmap_Original_vs_Counterfactuals.pdf', bbox_inches='tight', dpi=500)
    plt.show()

def draw_purple_secret_plot(id,original_df,counterfactual_df,var_name):

    conti_var = ['Skull.defect.area','DC.CP.interval','GCS','Surgery.time']

    combined_df = pd.concat([original_df.assign(Type='Original'), counterfactual_df.assign(Type='Counterfactual')])
    

    for var in conti_var:
        if var in combined_df.columns:
            combined_df[var] = combined_df[var] * std[var] + mean[var]
            combined_df[var] = combined_df[var].astype('int')

    combined_df_copy = combined_df.copy(deep=True)
    for var in conti_var:
        combined_df_copy[var] /= max_v[var]

    features = combined_df.columns.drop('Type')
    num_features = len(features)
    combined_df.loc[id,features[-1]] = 1.0
    print(vars)
    print(combined_df)
    combined_df.to_excel(f'{var_name}.xlsx')


def get_id(X,feature_name,var_name):  # random choose a sample
     
    for fea in feature_name:
        X = X.loc[X[fea] == 0]
    
    X = X.loc[X[var_name] == 1]

    return X.index[random.randint(0,len(X.index))]



mean = {            
    'Age':42.09072351,
    'Skull.defect.area':119.69336779,
    'DC.CP.interval':5.96628424,
    'GCS':13.02067183,
    'Surgery.time':168.59689922
}

std = {
    'Age':15.53508887,
    'Skull.defect.area':57.23446224,
    'DC.CP.interval':13.97955779,
    'GCS':3.3306842,
    'Surgery.time':77.60619325
}

max_v = {
    'Age':78,
    'Skull.defect.area':310,
    'DC.CP.interval':360,
    'GCS':15,
    'Surgery.time':670
}

if __name__ == '__main__':
    
    var_name = ['xxxxx'] # outcome variables

    model_names = ['rf','extra_tree',
                   'rotation','rf',
                   'extra_tree',
                   'extra_tree','rf','rf',]
    
    # set features to be changed
    features_to_vary = {
        'infection':[
            'Surgery.time','N.P.drainage','Titanium'
        ],
        'Fluid':[
            'Surgery.time','N.P.drainage'
        ],
        'Penu':[
            'Surgery.time','N.P.drainage','Titanium'
        ],
        'intra_hem':[
            'Surgery.time','N.P.drainage'
        ],
        'Hydro':[
            'Surgery.time','N.P.drainage','Titanium'
        ],
        'Seizures':[
            'Surgery.time','N.P.drainage','Titanium'
        ],
        'total':[
            'N.P.drainage','Titanium'
        ],
        'Reop':[
            'Surgery.time','N.P.drainage','Titanium'
        ]
    }

    continue_feature = ['Surgery.time','DC.CP.interval','Skull.defect.area','GCS']

    for i,(model_name,vars) in enumerate(zip(model_names,var_name)):
        
        print(vars)

        # get data
        X,y = get_data(id = i)

        feature_name = get_var_name(id = i)

        # set choose feature
        feature_set = ['N.P.drainage']

        # create model
        model = joblib.load('xxxx.pkl') # model path
        data = dice_ml.Data(dataframe=X,continuous_features = continue_feature,outcome_name=feature_name[-1])

        model = dice_ml.Model(model=model,backend='sklearn')
        dice = Dice(data,model)

        if 'Titanium' in feature_name:
            feature_set.append('Titanium')

        id = get_id(X,feature_set,feature_name[-1])

        query_instance = X.loc[[id]].to_dict()  # get random sample

        permitted_range = {
                "Surgery.time": [-1.6, X.loc[id,'Surgery.time']],  
            }

        fixed_feature_name = list(set(feature_name[:-1]) - set(features_to_vary[vars]))

        fixed_features = dict()

        for fea in fixed_feature_name:
            fixed_features[fea] = query_instance[fea]

        # generate counterfactuals
        counterfactuals = dice.generate_counterfactuals(X.loc[[id],feature_name[:-1]], total_CFs=3, 
                                                        features_to_vary=features_to_vary[vars],permitted_range=permitted_range,
                                                        )
        
        counterfactuals_df = counterfactuals.cf_examples_list[0].final_cfs_df
        counterfactuals_df.reset_index(drop=True, inplace=True)

        draw_rador(X.loc[[id],feature_name[:-1]],counterfactuals_df,vars)

        draw_heatmap2(X.loc[[id],feature_name[:-1]],counterfactuals_df,vars)

        draw_purple_secret_plot(id,X.loc[[id],feature_name[:-1]],counterfactuals_df,vars)

        
