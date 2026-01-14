import joblib
from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
import xlrd, random
from data import data_index, get_rows
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

random.seed(100)
np.random.seed(100)

def smooth_roc_curve(fpr, tpr, num_points=100):
    fpr_smooth = np.linspace(0, 1, num_points)
    tpr_smooth = interp1d(fpr, tpr)(fpr_smooth)
    return fpr_smooth, tpr_smooth


def draw(y_score, y_true, var_name, model_name, color):
    data_name = ['Internal Validation Cohort', 'External Validation Cohort']

    plt.figure(figsize=(10, 8))
    plt.title(var_name + ' ROC', fontsize=14)

    lw = 2  
    plt.plot([-0.02, 1], [-0.02, 1], color='#7f7f7f', lw=lw, linestyle='--', alpha=0.7)  # 基线
    plt.xlim([-0.02, 1.0])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('1 - Specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    for n in range(len(y_score)):
        fpr, tpr, thresholds = roc_curve(y_true[n], y_score[n])
        fpr_smooth, tpr_smooth = smooth_roc_curve(fpr, tpr)
        fpr_smooth[0] = 0
        tpr_smooth[0] = 0
        roc_auc = roc_auc_score(y_true[n], y_score[n])
        plt.plot(fpr_smooth, tpr_smooth, color=color[n], lw=lw, alpha=0.7,
                 label=f'{model_name} {data_name[n]} (AUC = {roc_auc:.3f})')


    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    rcParams['pdf.fonttype'] = 42
    plt.savefig(f'{var_name}_ROC.pdf') # save path
    plt.show()

def get_data(path, id=0, x_index=-1, y_index=-1):
    x, y = [], []
    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(id)
    for i in range(1, sheet.nrows):
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])
    return np.array(x), np.array(y)

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
    var_name = ['xxx','xxxx'] # outcome variables

    model_names = ['dt', 'gbdt', 'dt', 'extra_tree', 'svm', 'logit', 'bayes'] 

    data_path = ["xxxxx.xlsx",] # input data path

    color = ['#f16c23', '#2b6a99', '#1b7c3d', 'lime', 'violet', 'yellow', 'blue', 'orange']

    for i, (vars, models) in enumerate(zip(var_name, model_names)):

        y_true = []
        y_score = []

        print('-' * 6 + vars + ":" + '-' * 6)
       
        for j,path in enumerate(data_path):
            X, y = get_data(path=path, id=i)

            if 'External' not in path and j == 1:   

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
                    score_ = model.predict_proba(X_test)[:,1]

                    y_s.extend(list(score_))
                    y_t.extend(list(y_test))

                y_true.append(np.array(y_t))
                y_score.append(np.array(y_s))

            else:
                model = joblib.load(f'xxxxx.pkl')  # model path
                y_prob = model.predict_proba(X)[:,1]
                y_true.append(y)
                y_score.append(y_prob)

        print('-----------------------------------')
        draw(y_score, y_true, vars, models, color)
