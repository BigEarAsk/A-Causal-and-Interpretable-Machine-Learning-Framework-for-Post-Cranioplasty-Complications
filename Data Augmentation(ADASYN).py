import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

data_path = 'xxxx.xlsx' # input data path

data = pd.read_excel(data_path,0)

list = ['xxx','xxxxx','xxxx'] # the column names of imbalanced variables

# get data
for id in range(len(list)):

    X_train, y_train = data.iloc[:, :-1], data.iloc[:, -1] 
    
    counts_1 = y_train.value_counts()
    print('Imbalanced Data Distribution')
    print(counts_1)

    # create ADASYN 
    adasyn = ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=5)
    X_train_sm, y_train_sm = adasyn.fit_resample(X_train, y_train)


    counts_sm = y_train_sm.value_counts()
    print('Balanced Data Count')
    print(counts_sm)
    data_sm = pd.concat([X_train_sm, y_train_sm], axis=1)
    data_sm.to_excel(list[id]+'.xlsx', index=False) # save path
        
