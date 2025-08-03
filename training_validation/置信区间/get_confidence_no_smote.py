from bootstrap import bootstrap_metrics_bca
import xlwt,xlrd
import numpy as np
import joblib
import random
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, auc
from sklearn.utils import resample
from scipy.stats import bootstrap

np.random.seed(1000)
random.seed(1000)

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
    # data : 1 * 8 * 14
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet = workbook.add_sheet('result')

    for r in range(len(value_name)):
        sheet.write(0,r+1,value_name[r])
    for c in range(len(var_name)):
        sheet.write(c+1,0,var_name[c])
    
    sheet_data = np.array(data).reshape(len(data),len(value_name))

    for r in range(len(sheet_data)):
        model_data = sheet_data[r]
        for c in range(len(model_data)):
            sheet.write(r+1,c+1,model_data[c])
    workbook.save('置信区间/'+excel_name+' result.xls')

def write_index(data,model_name,var_name,suffix):
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet = workbook.add_sheet('youden_index')

    for i in range(len(var_name)):
        sheet.write(i+1,0,var_name[i])

    for i in range(len(model_name)):
        sheet.write(0,i+1,model_name[i])

    for r in range(len(var_name)):
        for c in range(len(model_name)):
            sheet.write(r+1,c+1,data[r][c])
    
    workbook.save('cutoff/youden_index_'+suffix+'.xls')


def calculate_bca_ci(data, stat_func, alpha=0.05): # use bootstrap
    res = bootstrap(data, stat_func, confidence_level=1-alpha, method='BCa')
    return res.confidence_interval.low, res.confidence_interval.high

def get_conf(y_prob,y_true,n_bootstrap):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    original_auc = auc(recall, precision)
    original_auc = min(original_auc,1)

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
    return round(original_auc,3), round(low,3), round(high,3)  


if __name__ == '__main__':
    
    model_name = ['rf']

    
    var_name = ['total']
    
    value_name = ['acc','auc','tp','tn','fp','fn','sensitivity','specificity',
                  'pos_pred','neg_pred','f1','brier_score','prauc']

    
    # Training need to run individually 
    # Internal Validation and External Validation can run together

    '''
    if run Internal Validation and External Validation together, 
    add 'AUC/External Validation Cohort(After Genetic Algorithm).xlsx' into data_path list
    delete 'Training' and add 'Internal Validation' and 'External Validation' into names list
    '''

    data_path = ['置信区间/时序验证-总_标准化后.xlsx']

    names = ['时序验证-总 _External Validation (No Smote)']

    prob_standard = {
        'infection':0.605,  
        'Fluid':0.569,
        'Penu':0.310, 
        'intra_hem':0.462,  
        'Hydro':0.470,
        'Seizures':0.534,
        'Reop':0.673,
        'total':0.366
    }

    for path,name in zip(data_path,names):
        data_total = [] # 8 sheets' total data 
        best_thresholds_total = []

        suffix = 'internal validation' if 'External' not in path else 'external validation'

        for i in range(len(var_name)):
            
            print('*'*6+var_name[i]+'*'*6+":")

            X,y = get_data(path = path,id=i)

            sheet_data = [] # each sheet's data
 
            best_thresholds = []  # each var's youden_index

            model = joblib.load('model_parameters_0.02/'+model_name[i]+'_'+var_name[i]+'.pkl')

            model_data  = []  # each model's data 

            print(var_name[i],end=':')

            score = model.predict_proba(X)

            if model_name[i] != 'gam':
                score = score[:,1]

            metrics, metrics_ci,best_threshold = bootstrap_metrics_bca(y, score,yoden_index=prob_standard[var_name[i]])
            
            print(best_threshold)
            best_thresholds.append(best_threshold)

            score = (score >= prob_standard[var_name[i]])
            print(np.where(score == 1))

            prauc,low,high = get_conf(score,y,2000)

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

                model_data.append(v_range)

            model_data.append(f"{prauc:.3f}({low:.3f},{high:.3f})")
            sheet_data.append(model_data)   # 8 * 14
            
            data_total.append(sheet_data)  # 1 * 8 * 14

        write_index(best_thresholds,model_name,var_name,suffix=suffix)
    write_xls_evaluate(data_total,model_name,var_name,value_name,name)

