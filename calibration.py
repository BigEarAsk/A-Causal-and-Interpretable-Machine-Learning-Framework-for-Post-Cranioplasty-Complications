import joblib
from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC
from sklearn.utils import resample
import xlrd
import random
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import splev, splrep
from data import data_index,get_rows
import statsmodels.api as sm
from sklearn.calibration import CalibratedClassifierCV

np.random.seed(100)
random.seed(100)

from scipy.interpolate import interp1d
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

def get_data(path,id = 0,x_index = -1,y_index = -1):  # get data 
    x,y = [],[]
    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(id)
    for i in range(1,sheet.nrows):  
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])

    return np.array(x),np.array(y)

def smooth_calibration_curve(mean_values, frac_values, frac_std=None, smooth_frac_std=False): # smooth curve
    # use Loess curve
    lowess = sm.nonparametric.lowess
    smooth_frac = lowess(frac_values, mean_values, frac_std if frac_std is not None else 0.05)
    smooth_mean, smooth_frac = smooth_frac[:, 0], smooth_frac[:, 1]

    return smooth_mean, smooth_frac

def draw(mean_values,frac_values,var_name,model_name,color,train = 0): #  draw ROC curve

    line_name = ['Internal Validation Cohort',
                 'External Validation Cohort',
                 ]

    plt.figure(figsize=(10,8))  # set figure size
    plt.title(var_name + ' Calibration Curve', fontsize=14)  # set title
    lw = 2  # set line-width
    plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=lw, linestyle='--', alpha=0.8,
             label='Ideal')  # draw diag

    plt.ylim([0.0, 1.05])  # set y axis range
    plt.xlabel('Mean predicted probability', fontsize=12)  # set x axis label
    plt.ylabel('Fraction of positives', fontsize=12)  # set y axis lable
    
    plt.grid(True, linestyle='--', alpha=0.5)  # show gird 
    plt.tight_layout()  # adjust the layout 

    for i in range(len(mean_values)):

        smooth_mean,smooth_frac = smooth_calibration_curve(mean_values[i],
                                                           frac_values[i],0.85,0.85)
        smooth_frac = np.clip(smooth_frac,0,1)

        plt.plot(smooth_mean, smooth_frac, color=color[i], lw=lw, 
                marker = 's', linestyle='-', label = line_name[i], )
        
    plt.legend(loc="center right", fontsize=12)  # set icon position and font size
    rcParams['pdf.fonttype'] = 42
    plt.savefig('results_img/Calibration Curve/'+var_name+' Calibration Curve(No smooth).pdf')
    plt.show()

def equal_frequency_binning(y_true, probas_pred, n_bins=5):  # get equal_frequency_binning
    
    indices = np.argsort(probas_pred)
    probas_pred_sorted = probas_pred[indices]
    y_true_sorted = y_true[indices]

    bin_edges = np.interp(np.linspace(0, len(probas_pred), n_bins + 1), 
                          np.arange(len(probas_pred)), probas_pred_sorted)

    prob_true = np.zeros(n_bins)
    prob_pred = np.zeros(n_bins)

    for i in range(n_bins):
        bin_mask = (probas_pred >= bin_edges[i]) & (probas_pred < bin_edges[i + 1])
        if bin_mask.any():
            prob_true[i] = y_true[bin_mask].mean()
            prob_pred[i] = probas_pred[bin_mask].mean()

    return prob_true, prob_pred

def kmeans_binning(y_true, y_prob, n_clusters=10):  # use k-means_binning 
    
    y_prob_reshaped = y_prob.reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(y_prob_reshaped)
    
    clusters = kmeans.predict(y_prob_reshaped)
    
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob, 'cluster': clusters})

    cluster_means = df.groupby('cluster')['y_prob'].mean()
    true_rate = df.groupby('cluster')['y_true'].mean()
    
    return cluster_means, true_rate


def adaptive_binning(y_true, y_prob, n_bins=10, n_clusters=3): # adaptive_binning
    
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values('y_prob')
    df['bin'] = pd.qcut(df['y_prob'], q=n_bins, labels=False,duplicates='drop')
    
    refined_bins = []
    for _, group in df.groupby('bin'):
        y_prob_reshaped = group['y_prob'].values.reshape(-1, 1)
        if len(y_prob_reshaped) > n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            group['sub_bin'] = kmeans.fit_predict(y_prob_reshaped)
            refined_bins.append(group)
        else:
            group['sub_bin'] = 0
            refined_bins.append(group)
    
    refined_df = pd.concat(refined_bins)
    
    bin_means = refined_df.groupby(['bin', 'sub_bin'])['y_prob'].mean()
    true_rate = refined_df.groupby(['bin', 'sub_bin'])['y_true'].mean()
    
    return bin_means, true_rate


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
    
    
    model_names = ['rf','rf',
                   'rotation','logit',
                   'gam',
                   'logit','rf',]


    color = [
        'brown','#4CAF50','#2b6a99',
        'gold','lime','violet',
    ]

    
    data_path = ["xxxx.xlsx"] # input data path
    
    # define use which process
    kmeans = False  
    equi = True
    origin = not False
    n_bins = 15

    # hydro and Reop need draw individually!
    for i,(vars,model_name) in enumerate(zip(var_name,model_names)):
        
        mean = []
        frac = []

        print(model_name)
        
        print('-'*6 + vars +":" + '-'*6)

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
                    
                    score_ = model.predict_proba(X_test)

                    if model_name != 'gam':
                        score_ = score_[:,1]

                    y_s.extend(list(score_))
                    y_t.extend(list(y_test))
                    
                if origin:
                    true_rate, bin_means = \
                    calibration_curve(np.array(y_t), np.array(y_s), n_bins=n_bins)
                elif kmeans:
                    bin_means, true_rate = adaptive_binning(np.array(y_t), np.array(y_s),
                                                            n_bins=n_bins, 
                                                      n_clusters=5)
                else:
                    true_rate, bin_means = equal_frequency_binning(np.array(y_t), np.array(y_s), 
                                                                   n_bins=n_bins)

                mean.append(bin_means)
                frac.append(true_rate)

            else:
                if 'External' in path:

                    if model_name not in ['gam','logit']:
                            model = CalibratedClassifierCV(base_estimator=model, 
                                                            method='sigmoid', cv='prefit')
                            model.fit(X, y)
                    else:
                        model = model
 
                model = joblib.load('xxxx.pkl')  # model path

                y_prob = model.predict_proba(X)

                if model_name != 'gam':
                    y_prob = y_prob[:,1]
                
                if origin:
                    true_rate, bin_means = \
                    calibration_curve(y, y_prob, n_bins=n_bins)
                elif kmeans:
                    bin_means, true_rate = adaptive_binning(y, 
                                                      y_prob, n_bins=n_bins, 
                                                      n_clusters=5)
                else:
                    true_rate, bin_means = equal_frequency_binning(y, 
                                                 y_prob, n_bins=n_bins)
                    
                mean.append(bin_means)
                frac.append(true_rate)

            print('-----------------------------------')

        draw(mean,frac,vars,model_name,color)

        