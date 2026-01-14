import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
import xlrd
from data import data_index,get_rows
from matplotlib import rcParams
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold, train_test_split
from pygam import LogisticGAM, s
import lightgbm as lgb
import random

random.seed(100)
np.random.seed(100)

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def plot_DCA(ax, thresh_group, net_benefit_model, net_benefit_all,data_name,color,
             none_color,all_color):

    for i in range(len(thresh_group)):

        # Plot
        ax.plot(thresh_group[i], net_benefit_model[i], color[i], label = data_name[i])
        ax.plot(thresh_group[i], net_benefit_all[i], color = all_color[i],label = data_name[i] + ' Treat all')
        ax.plot((0, 1), (0, 0), color = none_color[i], linestyle = ':', label = data_name[i] + ' Treat none')

        # Fill
        y2 = np.maximum(net_benefit_all[i], 0)
        y1 = np.maximum(net_benefit_model[i], y2)
        ax.fill_between(thresh_group[i], y1, y2, color = color[i], alpha = 0.2)

        # Figure Configuration
        ax.set_xlim(0,1)
        ax.set_ylim(net_benefit_model[i].min() - 0.15, net_benefit_model[i].max() + 0.15)

    ax.set_xlabel(
        xlabel = 'Threshold Probability', 

        fontdict= {'fontsize': 15}
        )
    
    ax.set_ylabel(
        ylabel = 'Net Benefit', 

        fontdict= {'fontsize': 15}
        )
    
    ax.grid('major')
    ax.spines['right'].set_color((0.8, 0.8, 0.8))
    ax.spines['top'].set_color((0.8, 0.8, 0.8))
    ax.legend(loc = 'upper right')

    return ax

def get_data(path,id = 0,x_index = -1,y_index = -1):
    x,y = [],[]
    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(id)
    for i in range(1,sheet.nrows): 
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])

    return np.array(x),np.array(y)

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

if __name__ == '__main__':

    var_name = ['xxxx'] # outcome variables

    model_names = ['rf', 'extra_tree', 'rotation', 'logit', 'gam', 'logit', 'rf']

    data_path = ["xxxx.xlsx"]  # input data path

    data_name = ['Internal Validations Cohort','External Validations Cohort']
    color = ['crimson','#4CAF50','blue']
    none_color = ['black','#996633','#7f7f7f']
    all_color = ['black','#996633','#7f7f7f']

    for i,(vars,model_name) in enumerate(zip(var_name,model_names)):

        cali_X,cali_y = get_data(path="xxxx.xlsx",id=i) # calibration data path

        thresh_groups = []  
        net_benefit_models = []
        net_benefit_alls = []

        for j,path in enumerate(data_path):
            X, y = get_data(path=path, id=i)

            if 'External' not in path and j == 0:   

                y_t = []
                y_s = []
                k_folds = 5  # set K value 
                kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    if model_names[i] == 'gam':
                        model = LogisticGAM(lam=0.1,n_splines = 10)

                    elif var_name[i] in ['Fluid','Reop'] and model_names[i] == 'rotation':
                        model = Pipeline([
                            ('pca', PCA(n_components=5)),  
                            ('clf', RandomForestClassifier(n_estimators=200, 
                                        max_depth=7,random_state=42))  
                        ])

                    else:
                        model = model_dict[model_names[i]]

                    model.fit(X_train,y_train)

                    if model_names[i] != 'gam':
                        score_ = model.predict_proba(X_test)[:,1]
                    else:
                        score_ = model.predict_proba(X_test)
                        
                    y_s.extend(list(score_))
                    y_t.extend(list(y_test))


                thresh_group = np.arange(0,1,0.01)
                net_benefit_model = calculate_net_benefit_model(thresh_group, np.array(y_s), np.array(y_t))
                net_benefit_all = calculate_net_benefit_all(thresh_group, np.array(y_t))
                
                thresh_groups.append(thresh_group)
                net_benefit_alls.append(net_benefit_all)
                net_benefit_models.append(net_benefit_model)

            else:
                model = joblib.load('xxx.pkl') # model path

                y_prob = model.predict_proba(X)
                if model_name != 'gam':
                    y_prob = y_prob[:,1]
                thresh_group = np.arange(0,1,0.01)
                net_benefit_model = calculate_net_benefit_model(thresh_group, y_prob, y)
                net_benefit_all = calculate_net_benefit_all(thresh_group, y)

                thresh_groups.append(thresh_group)
                net_benefit_alls.append(net_benefit_all)
                net_benefit_models.append(net_benefit_model)

        fig, ax = plt.subplots()
        plt.title(vars + ' DCA Curve')
        ax = plot_DCA(ax, thresh_groups, net_benefit_models, net_benefit_alls,data_name,color,
                      none_color,all_color)
         
        rcParams['pdf.fonttype'] = 42   
        fig.savefig(var_name + ' DCA.pdf')  # save path
        plt.show()
