from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier
from bootstrap import bootstrap_metrics_bca
import xlwt,xlrd
import numpy as np
import joblib
import random
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold
from pygam import LogisticGAM
import lightgbm as lgb

np.random.seed(100)
random.seed(100)

# Only  Training, Internal Validation and External Validation
# External Validation: AUC/External Validation Cohort(After Genetic Algorithm).xlsx
# Training and Internal Validation: AUC/Derivation Cohort(After Genetic Algorithm).xlsx

def get_data(path = 'AUC/Derivation Cohort(After Genetic Algorithm).xlsx',id = 0,rows = None,x_index = -1,y_index = -1):
    x,y = [],[]
    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(id)
    for i in range(1,sheet.nrows):  
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])

    return np.array(x),np.array(y)

def write_xls_evaluate(data,model_name,var_name,value_name,excel_name):
    # data : 1 * 16 * 14
    workbook = xlwt.Workbook(encoding='utf-8')
    data = np.array(data)

    data = data.reshape(data.shape[1],data.shape[0],-1)

    for i in range(len(data)):
        sheet = workbook.add_sheet(var_name[i])

        for r in range(len(value_name)):
            sheet.write(0,r+1,value_name[r])
        for c in range(len(model_name)):
            sheet.write(c+1,0,model_name[c])
        
        sheet_data = data[i]

        for r in range(len(sheet_data)):
            model_data = sheet_data[r]
            for c in range(len(model_data)):
                sheet.write(r+1,c+1,model_data[c])
    workbook.save('置信区间/'+excel_name+'_result.xls')

def write_index(data,model_name,var_name,suffix):
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet = workbook.add_sheet('youden_index')

    for i in range(len(var_name)):
        sheet.write(i+1,0,var_name[i])

    for i in range(len(model_name)):
        sheet.write(0,i+1,model_name[i])

    for r in range(len(var_name)):
        for c in range(len(model_name)):
            sheet.write(r+1,c+1,str(data[c][r]))
    
    workbook.save('cutoff/youden_index_'+suffix+'.xls')


if __name__ == '__main__':

    model_name = ['gam', 'logit', 'gbdt', 'knn', 'ada', 'lgb',
                  'rotation', 'rf',
                  'xgboost', 
                  'gauss_cls', 'extra_tree',
                  'dt', 'bayes', 'mlp', 'svm']
    
    model_dict = {
        'gam':LogisticGAM(lam=0.1,n_splines = 10),
        'logit':LogisticRegression(C = 1,
                                    penalty='l2',
                                    solver='liblinear',
                                    max_iter=1000
                                    ),
        'gbdt':GradientBoostingClassifier(n_estimators=100,  
                                        learning_rate=0.5,  
                                        max_depth=7,  
                                        random_state=42),
        'knn':KNeighborsClassifier(metric='manhattan',
                                        n_neighbors=9,
                                        weights='distance'),
        'ada':AdaBoostClassifier(learning_rate=0.1,n_estimators=200),

        'lgb':lgb.LGBMClassifier(learning_rate=0.01,
                                    num_leaves=15,
                                    n_estimators=200),
        'rotation':Pipeline([
            ('pca', PCA(n_components=7)),  
            ('clf', RandomForestClassifier(n_estimators=200, 
                        max_depth=7,random_state=42))  
        ]),
        'rf':RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            min_samples_split=2, 
            max_features='sqrt',
            random_state=42),
        'xgboost':XGBClassifier(learning_rate=0.5,
                                max_depth = 7,
                                n_estimators = 50),
        'gauss_cls':GaussianProcessClassifier(kernel=None, n_restarts_optimizer=0
                                            ,random_state=0),
        'extra_tree':ExtraTreesClassifier(
            max_depth=20,
            max_features='log2',
            min_samples_leaf=1,
            min_samples_split=5,
            n_estimators=100
        ),

        'dt':tree.DecisionTreeClassifier(criterion='entropy',
                                            min_samples_leaf=1,
                                            max_depth=5),
        'bayes':GaussianNB(),
        'mlp':MLPClassifier(hidden_layer_sizes=(100, 50),  
                            activation='relu',  
                            solver='adam',  
                            max_iter=1000, 
                            random_state=42),
        'svm': SVC(C = 3,degree=3,kernel='rbf',probability=True)
    }

    var_name = ['total']
    
    value_name = ['acc','auc','tp','tn','fp','fn','sensitivity','specificity',
                  'pos_pred','neg_pred','f1','brier_score','youden_index']

    
    # Training need to run individually 
    # Internal Validation and External Validation can run together

    '''
    if run Internal Validation and External Validation together, 
    add 'AUC/External Validation Cohort(After Genetic Algorithm).xlsx' into data_path list
    delete 'Training' and add 'Internal Validation' and 'External Validation' into names list
    '''

    data_path = ["models/Derivation Cohort(After Genetic Algorithm).xlsx",
                 "models/Derivation Cohort(After Genetic Algorithm).xlsx",
                 "models/External Validation Cohort(After Genetic Algorithm).xlsx"]
    
    names = ['Training','Internal Validation','External Validation']

    file = xlrd.open_workbook('cutoff/youden_index_Training.xls')
    youden_sheet = file.sheet_by_index(0)

    cols = [
        [0,3,4,6,7],
        [0,1,2,3,4],
        [0,1,2,4,5],
        [0,3,4,5,6],
        [0,4,5,9,10],
        [0,2,3,5,6],
        [0,3,4,8,9]
    ]

    prob_standard = {
        'infection':0.605,  
        'Fluid':0.569,
        'Penu':0.310, 
        'intra_hem':0.462,  
        'Hydro':0.470,
        'Seizures':0.534,
        'total':0.366,
        'Reop':0.673
    }

    for path,name in zip(data_path,names):

        data_total = [] # 1 sheets' total data 
        best_thresholds_total = []

        suffix = name
        
        sheet_data = [] # each sheet's data

        best_thresholds = []  # each var's youden_index

        for i in range(15):

            model_data  = []  # each model's data

            print('*'*6+model_name[i]+'*'*6+":")

            best_tmp = []
            
            for j,vars in enumerate(var_name):

                score = []

                print('\t',var_name[j])
                X,y = get_data(path = path,id=j)

                y_true = [] if 'External' not in path and name != 'Training' else y

                if name != 'Training' and 'External' not in path:

                    k_folds = 5  # set K value 
                    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                        
                    for train_index, test_index in kf.split(X):
                        
                        if model_name[i] == 'gam':
                            model = LogisticGAM(lam=0.1,n_splines = 10)

                        elif var_name[j] in ['Fluid','Reop'] and model_name[i] == 'rotation':
                            model = Pipeline([
                                ('pca', PCA(n_components=5)),  
                                ('clf', RandomForestClassifier(n_estimators=200, 
                                            max_depth=7,random_state=42))  
                            ])

                        else:
                            model = model_dict[model_name[i]]

                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                        for col in cols[j]:
                            std = X_train[:,col].std()
                            noise = np.random.normal(0, std * 0.08, size=len(X_train))
                            X_train[:,col] += noise
                        
                        model.fit(X_train,y_train)

                        score_ = model.predict_proba(X_test)

                        if model_name[i] != 'gam':
                            score_ = score_[:,1]

                        score.extend(list(score_))
                        y_true.extend(list(y_test))

                    score = np.array(score)
                    y_true = np.array(y_true)

                else:
                    model = joblib.load('model_parameters_0.02/'+model_name[i]+f'_{vars}.pkl')
                    print(model_name[i]+f'_{vars}.pkl',model.get_params())
                    
                    score = model.predict_proba(X)

                    if model_name[i] != 'gam':
                        score = score[:,1]

                yoden_index = prob_standard[vars] # if 'External' not in path and name != 'Training' else youden_index_list[i]

                metrics, metrics_ci,best_threshold = bootstrap_metrics_bca(y_true, score,
                                                                        yoden_index=yoden_index)

                best_tmp.append(best_threshold)

                var_data = []

                for key, value in metrics.items():
                    
                    v1 = metrics_ci[key][0]
                    v2 = metrics_ci[key][1]

                    if key not in ['tp','tn','fp','fn']:
                        value = min(value,1)    
                        v1 = min(metrics_ci[key][0],1)
                        v2 = min(metrics_ci[key][1],1)

                    if key in ['tp','tn','fp','fn']:
                        v_range = f"{value:.3f}"
                    else: v_range = f"{value:.3f}({v1:.3f},{v2:.3f})"

                    var_data.append(v_range)
                
                var_data.append(str(best_threshold))
                model_data.append(var_data)   # 7 * 13
            sheet_data.append(model_data)  # 15 * 7 * 13

            best_thresholds.append(best_tmp)

        write_index(best_thresholds,model_name,var_name,suffix=suffix)
        write_xls_evaluate(sheet_data,model_name,var_name,value_name,name)

