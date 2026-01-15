from scipy.stats import bootstrap
import joblib
from matplotlib import rcParams
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import xlrd
from scipy.interpolate import interp1d
import xlwt
from sklearn.utils import resample

def calculate_bca_ci(data, stat_func, alpha=0.05): # use bootstrap
    res = bootstrap(data, stat_func, confidence_level=1-alpha, method='BCa')
    return res.confidence_interval.low, res.confidence_interval.high

def smooth_pr_curve(fpr, tpr, num_points=100):
    fpr_smooth = np.linspace(0, 1, num_points)
    tpr_smooth = interp1d(fpr, tpr)(fpr_smooth)
    return fpr_smooth, tpr_smooth

def draw(y_score,y_true,var_name,model_name,color):

    data_name = ['Internal Validation Cohort','External Validation Cohort']
    # draw pr curve
    plt.figure(figsize=(8, 6))
    plt.title(var_name + ' PR', fontsize=16)  # set title
    lw = 2  # set line width
    plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=lw, linestyle='--', alpha=0.8)  # draw diag
    plt.xlim([0.0, 1.0])  # set x axis range
    plt.ylim([0.0, 1.05])  # set y axis range

    plt.xlabel('Recall',fontsize = 14)   
    plt.ylabel('Precision',fontsize = 14)
    
    plt.grid(True, linestyle='--', alpha=0.5)  # show grid 
    plt.tight_layout()  # adjust layout

    # draw 
    for n in range(len(y_score)):

        # compute precision and recall value 
        precision, recall, thresholds = precision_recall_curve(y_true[n], y_score[n])
        auc_score = auc(recall, precision)
        auc_score = min(auc_score,1)

        plt.plot(recall, precision, color=color[n], lw=lw, 
                 label=model_name + ' ' + data_name[n] + '(area = {:.2f})'.format(min(auc_score,1)), alpha=0.8)
        plt.legend(loc="lower right", fontsize=12)  # set icon position and font size

    rcParams['pdf.fonttype'] = 42
    plt.savefig(var_name+' PR.pdf')  # save figure
    plt.show()

def get_data(path,id = 0,x_index = -1,y_index = -1):
    x,y = [],[]
    file = xlrd.open_workbook(path)
    sheet = file.sheet_by_index(id)
    for i in range(1,sheet.nrows):  
        user = sheet.row_values(i)
        x.append(user[:x_index])
        y.append(user[y_index])

    return np.array(x),np.array(y)

def get_conf(y_prob,y_true,n_bootstrap):

    prauc = []
    for i in range(n_bootstrap):
        indices = resample(np.arange(len(y_true)), replace=True)

        while (1 not in y_true[indices]) or (0 not in y_true[indices]):
            indices = resample(np.arange(len(y_true)), replace=True)
            
        y_true_bootstrap = y_true[indices]
        y_prob_bootstrap = y_prob[indices]

        precision, recall, thresholds = precision_recall_curve(y_true_bootstrap, y_prob_bootstrap)
        auc_score = auc(recall, precision)
        auc_score = min(auc_score,1)

        prauc.append(auc_score)
    
    low, high = calculate_bca_ci((prauc,),np.mean)
    return round(low,3),round(high,3)

def write_xls(data,var_name,dataset):
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet = workbook.add_sheet('Sheet1')

    for i in range(len(var_name)):
        sheet.write(0,i+1,var_name[i])
    
    for i in range(len(dataset)):
        sheet.write(i+1,0,dataset[i])
    
    for i,dic in enumerate(data.values()):
        for j,val in enumerate(dic.values()):
            sheet.write(j+1,i+1,str(val))
    
    workbook.save('xxxx.xls') # save path

rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            min_samples_split=2, 
            max_features='sqrt',
            random_state=42)

if __name__ == '__main__':
    
    var_name = ['xxxx'] # outcome variables

    dataset = [
        'Internal Validation Cohort','External Validation Cohort'
    ]

    model_names = ['rf','extra_tree',
                   'rotation','rf',
                   'extra_tree',
                   'extra_tree','rf','rf',]
    
    
    
    data_path = ['xxxx.xlsx']  # input data path
     
    color = [
        '#f16c23','#2b6a99',
        '#1b7c3d','lime','violet',
        'yellow','blue','orange',
        'pink','cyan','grey',
        'yellowgreen','purple','black'
    ]

    conf_range = {key:{keys:None for keys in dataset} for key in var_name}
    
    for i,(vars,models) in enumerate(zip(var_name,model_names)):
        
        # record label and score
        y_true = []
        y_score = []

        print('-'*6 + vars +":" + '-'*6)
        
        for j,path in enumerate(data_path):
            X, y = get_data(path=path, id=0)

            if 'External' not in path and j == 0:   

                y_t = []
                y_s = []
                k_folds = 5  # set K value 
                kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    model = rf
                    model.fit(X_train,y_train)
                    score_ = model.predict_proba(X_test)[:,1]

                    y_s.extend(list(score_))
                    y_t.extend(list(y_test))

                y_true.append(np.array(y_t))
                y_score.append(np.array(y_s))

                low,high = get_conf(np.array(y_s),np.array(y_t),1000)
                conf_range[vars]['Internal Validation Cohort'] = (low,high)

            else:
                model = joblib.load('xxx.pkl') # model_path
                if models == 'gam':
                    y_prob = model.predict_proba(X)
                else:
                    y_prob = model.predict_proba(X)[:,1]
                
                low,high = get_conf(y_prob,y,1000)

                name = 'xxxxx' if j == 1 else 'External Validation Cohort'
                conf_range[vars][name] = (low,high)

                y_true.append(y)
                y_score.append(y_prob)

        print('-----------------------------------')
        draw(y_score,y_true,vars,models,color)
    write_xls(conf_range,var_name,dataset)


    
