from matplotlib import rcParams
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import xlrd
import joblib 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from bootstrap import calculate_metrics,bootstrap_metrics,bootstrap_metrics_bca
import xlwt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
from sklearn.utils import resample

def get_data(df,col,op,val,x_index,y_index):  # get data
    
    if op not in ['=','<','>=']:
        print('op not valid！')
        return None , None
    
    # choose different data by given op
    if op == '=':
        data = df.loc[df[col] == val]
    elif op == '<':
        data = df.loc[df[col] < val]
    elif op == '>=':
        data = df.loc[df[col] >= val]

    X = data.loc[:,x_index].to_numpy()

    y = data.loc[:,y_index].to_numpy()

    return X,y

def evaluate(model_name,calibrated_model,X,y,yoden_index=None): # evaluate the model

    tmp = []
    auc = None
    acc = None

    if model_name not in ['gam','logit']:
        calibrated_model.fit(X ,y)

    score = calibrated_model.predict_proba(X)
    pred = calibrated_model.predict(X)
    
    total_num = len(X)
    true_positive = np.sum(y)
    
    if model_name != 'gam':
        score = score[:,1]
    
    if yoden_index != None:
        pred = (score >= yoden_index)

    pred_positive = np.sum(pred)

    metrics, metrics_ci = bootstrap_metrics_bca(y,score,yoden_index=yoden_index)

    for key, value in metrics.items():
        
        v1 = metrics_ci[key][0]
        v2 = metrics_ci[key][1]

        if key not in ['tp','tn','fp','fn']:
            value = min(value,1)    
            v1 = min(metrics_ci[key][0],1)
            v2 = min(metrics_ci[key][1],1)

        if key == 'auc':
            auc = (round(value,3),round(v1,3),round(v2,3))
        if key == 'acc':
            acc = round(value,3)
        if key in ['tp','tn','fp','fn']:
            v_range = f"{value:.3f}"
        else: v_range = f"{value:.3f}({v1:.3f},{v2:.3f})"

        tmp.append(v_range)

    return tmp,auc,total_num,true_positive,pred_positive,acc

def write_xls_analysis(data,var_name,value_name,row_names): # save data 
    global workbook

    for i,name in enumerate(var_name):
        sheet = workbook.add_sheet(name)
        sheet_data = data[i]
        
        for j in range(len(value_name)):
            sheet.write(0,j+1,value_name[j])
    
        for j in range(len(row_names)):
            sheet.write(j+1,0,row_names[j])

        for j in range(len(sheet_data)):
            var_data = sheet_data[j]
            for k in range(len(var_data)):
                sheet.write(j+1,k+1,str(var_data[k]))

def draw2(auc,name,y_min = 0.6,y_max = 1.5,acc = None,sample_size = None):
    auc = np.array(auc)
    acc = np.array(acc)
    sample_size = np.array(sample_size)

    data = pd.DataFrame({
    'Group': ['All', '<40', '≥40', 'Male', 'Female'],
    'AUROC': [auc[0][0],auc[1][0],auc[2][0],auc[3][0],auc[4][0]],
    'SampleSize': [sample_size[0],sample_size[1],sample_size[2],sample_size[3],sample_size[4]],
    'ACCURACY': [acc[0],acc[1],acc[2],acc[3],acc[4]] 
    })


    plt.figure(figsize=(10, 6))


    scatter = plt.scatter(
        x=data['AUROC'],
        y=data['Group'],
        s=data['SampleSize'],                
        c=data['ACCURACY'],                 
        cmap='viridis',
        alpha=0.7,
        edgecolors='black'
    )

    for i, row in data.iterrows():
        plt.text(row['AUROC'], row['Group'], f"{row['SampleSize']}", 
                va='center', ha='center', fontsize=9, color='white')


    cbar = plt.colorbar(scatter)
    cbar.set_label('ACCURACY')

    plt.xlabel('AUROC')
    plt.title('Bubble Chart of Subgroup Performance\n(AUROC vs Sample Size vs ACCURACY)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    rcParams['pdf.fonttype'] = 42
    plt.savefig(name+' subgroup analysis.pdf')  # save path
    plt.show()
    

def draw(auc,name,y_min = 0.6,y_max = 1.5):  # draw auc figure

    auc = np.array(auc)

    data = {
    'Group': ['All', '<40', '≥40', 'Male','Female'],
    'Category': ['Total', 'Age', 'Age','Sex','Sex'],
    'AUROC': [auc[0][0],auc[1][0],auc[2][0],auc[3][0],auc[4][0]],
    'CI_Lower': [auc[0][1],auc[1][1],auc[2][1],auc[3][1],auc[4][1]],
    'CI_Upper': [auc[0][2],auc[1][2],auc[2][2],auc[3][2],auc[4][2]]
    }

    df = pd.DataFrame(data)

    # compute error line
    df['Error'] = df['CI_Upper'] - df['AUROC']

   
    sns.set_theme(style="whitegrid")

    # draw bar figure
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(
        x='Group', 
        y='AUROC', 
        hue='Category', 
        data=df, 
        palette=['#FF5575', '#FFD36A', '#6299FF','#A3D4C5'], 
        dodge=False,
        ci=None,
        edgecolor='black'
    )

    # add error line
    for index, row in df.iterrows():
        plt.errorbar(index, row['AUROC'], 
                    yerr=[[max(row['AUROC'] - row['CI_Lower'],0)], 
                          [max(row['CI_Upper'] - row['AUROC'],0)]], 
                    fmt='none', 
                    color='black', 
                    capsize=5)

    # beautiful 
    plt.title(name+' AUROC by Group', fontsize=16)
    plt.axhline(y=auc[0][0],linestyle='--',color='gray')
    plt.xlabel('', fontsize=14)

    plt.ylabel('AUROC ('+'%'+' of All)', fontsize=14)
    plt.ylim(y_min, y_max)
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12)
    plt.legend(title='', loc='upper left')

    sns.despine()
    rcParams['pdf.fonttype'] = 42
    plt.savefig(+name+' subgroup analysis.pdf') # save path
    plt.show()

def calculate_bca_ci(data, stat_func, alpha=0.05): # use bootstrap
    res = bootstrap(data, stat_func, confidence_level=1-alpha, method='BCa')
    return res.confidence_interval.low, res.confidence_interval.high

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

def get_prauc(model,model_name,X,y):
    # prauc:
    y_prob = model.predict_proba(X)

    if model_name != 'gam':
        y_prob = y_prob[:,1]

    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    auc_score = auc(recall, precision)
    auc_score = min(auc_score,1)

    low,high = get_conf(y_prob,y,1000)

    return f"{auc_score:.3f}({low},{high})"


if __name__ == '__main__':

    # woman 1 man 0
    
    path = 'xxxx.xlsx' # input data path
    
    var_name = ['xxx'] # outcome variables 
    
    value_name = ['acc','auc','tp','tn','fp','fn','sensitivity','specificity',
                  'pos_pred','neg_pred','f1','brier_score','prauc','total_num','true_positive','pred_positive']
    

    y_indexes = ['xxxxx'] # outcome variables 

    model_names = ['rf','extra_tree',
                   'rotation','rf',
                   'extra_tree',
                   'extra_tree','rf','rf',]


    # the corresponding independent variables of each outcome variables
    choose_feas = [

        ['Age','Alcohol','Pre.op.infection','Skull.defect.area','DC.CP.interval',
         'Titanium','GCS' ,'Surgery.time','N.P.drainage'],

        ['Age','Skull.defect.area','DC.CP.interval' ,'GCS',
         'Surgery.time','N.P.drainage'],

        ['Age','Skull.defect.area',
         'DC.CP.interval','Titanium','GCS','Surgery.time','N.P.drainage'],

        ['Age','Alcohol','Pre.op.infection','Skull.defect.area',
         'DC.CP.interval','GCS','Surgery.time','N.P.drainage'],

        ['Age','Smoke','Alcohol','Pre.op.infection','Skull.defect.area',
         'DC.CP.interval','ICH','Pre.op.V.P','Titanium','GCS','Surgery.time','N.P.drainage'],

        ['Age','Pre.op.seizures','Skull.defect.area','DC.CP.interval',
         'Titanium','GCS','Surgery.time','N.P.drainage'],

        ['Age','Skull.defect.area','DC.CP.interval',
         'GCS','Surgery.time','N.P.drainage'],

        ['Age','Sex','Alcohol',
         'Skull.defect.area','DC.CP.interval','Trauma','Pre.op.V.P',
         'Titanium','GCS','Surgery.time','N.P.drainage'],
    ]

    analysis_feas = {
        'Age':40,
        'Sex':[0,1],
    }

    row_names = [
        'All',
        'Age < 40',
        'Age >= 40',
        'Sex = Male',
        'Sex = Female',
    ]

    prob_standard = {
        'infection':0.277,
        'Fluid':0.396,
        'Penu':0.452,
        'intra_hem':0.370,
        'Hydro':0.554,
        'Seizures':0.633,
        'Total':0.580,
        'Reop':0.423,
        'total':0.3657407318118504
    }
    
    sheet_data = []
    workbook = xlwt.Workbook(encoding='utf-8')

    auc_list = []
    acc_list = []
    sample_size = []
    for i,(vars,choose_fea,model_name,y_index) in \
        enumerate(zip(var_name,choose_feas,model_names,y_indexes)):
        
        yoden_index = prob_standard[vars]

        print(vars)

        model = joblib.load('xxxx.pkl') # model path

        if model_name not in ['gam','logit']:
            calibrated_model = CalibratedClassifierCV(estimator=model, 
                                                  method='sigmoid', cv='prefit')
        else:
            calibrated_model = model

        # calibrated_model = model
        df = pd.read_excel(path,i)

        var_auc = []
        var_acc = []
        var_evaluate_data = []

        data = df.copy()

        X = data.loc[:,choose_fea].to_numpy()
        y = data.loc[:,y_index].to_numpy()
        tmp1, auc1,length,true_positive,pred_positive,acc = evaluate(model_name,calibrated_model,X,y,yoden_index)

        prauc = get_prauc(calibrated_model,model_name,X,y)
        tmp1.append(prauc)
        tmp1.append(length)
        sample_size.append(length)
        tmp1.append(true_positive)
        tmp1.append(pred_positive)
        var_evaluate_data.append(tmp1)

        print(f"All:\t{length}\t{true_positive}\t{pred_positive}")

        var_auc.append(auc1)
        var_acc.append(acc)

        for k,v in analysis_feas.items():
            
            if type(v) == type(1):  # >=  <

                if 'External' not in path:
                    v = (v-41.98747409) / 15.68930025
                else:
                    v = (v-42.29562982) / 15.22234545

                X,y = get_data(
                    df,k,'<',v,choose_fea,y_index
                )   

                tmp,auc1,length,tp,pred_p,acc = evaluate(model_name,calibrated_model,X,y,yoden_index)
                sample_size.append(length)
                var_auc.append(auc1)
                var_acc.append(acc)
                prauc = get_prauc(calibrated_model,model_name,X,y)
                tmp.append(prauc)
                tmp.append(length)
                tmp.append(tp)
                tmp.append(pred_p)
                print(f'{k} < 40:\t{length}\t{tp}\t{pred_p}')
                var_evaluate_data.append(tmp)

                X,y = get_data(
                    df,k,'>=',v,choose_fea,y_index
                ) 

                tmp,auc1,length,tp,pred_p,acc = evaluate(model_name,calibrated_model,X,y,yoden_index)
                var_auc.append(auc1)
                var_acc.append(acc)
                sample_size.append(length)
                prauc = get_prauc(calibrated_model,model_name,X,y)
                tmp.append(prauc)
                tmp.append(length)
                tmp.append(tp)
                tmp.append(pred_p)
                print(f'{k} >= 40:\t{length}\t{tp}\t{pred_p}')
                var_evaluate_data.append(tmp)

            else:  # == 
                for i in range(len(v)):
                    X,y = get_data(
                        df,k,'=',v[i],choose_fea,y_index
                    )   
                    tmp,auc1,length,tp,pred_p,acc = evaluate(model_name,calibrated_model,X,y,yoden_index)
                    var_auc.append(auc1)
                    var_acc.append(acc)
                    sample_size.append(length)
                    prauc = get_prauc(calibrated_model,model_name,X,y)
                    tmp.append(prauc)
                    tmp.append(length)
                    tmp.append(tp)
                    tmp.append(pred_p)
                    print(f'{k} = {v[i]}:\t{length}\t{tp}\t{pred_p}')
                    var_evaluate_data.append(tmp)

        auc_list.append(var_auc)
        acc_list.append(acc)
        sheet_data.append(var_evaluate_data)

    write_xls_analysis(sheet_data,var_name,value_name,row_names)

    workbook.save('Subgroup Analysis/subgroup analysis result_total_all.xls')

    y_min = [0.85,0.6,0.8,0.6,0.6,0.6,0.6,0.6] 
    y_max = [1.]*8

    for i in range(len(auc_list)):
        draw2(auc_list[i],var_name[i],y_min[i],y_max[i],acc_list[i],sample_size)


    
    

   

    
    




    
