# 导入必要的库
import random
import numpy as np
import pandas as pd
from econml.grf import RegressionForest, CausalForest
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from econml.dml import LinearDML
from scipy.stats import mannwhitneyu


np.random.seed(1000)
random.seed(1000)


def check(cate,X_train,T_binary_train,Y_train,vars,max_features=None):

    if 'N' in vars:
        causal_forest = CausalForest(n_estimators=100, min_samples_leaf=10, random_state=42)
    else:
        causal_forest = CausalForest(n_estimators=100, min_samples_leaf=12, max_features = max_features,
                                     random_state=42,max_depth=2)
        
    random_col = np.random.randn(len(X_train))  
    X_augmented = X_train.copy()
    X_augmented["random_feature"] = random_col
    causal_forest.fit(X_augmented, T_binary_train[vars], Y_train)

    random_col = np.random.randn(len(X_test))
    X_augmented = X_test.copy()
    X_augmented["random_feature"] = random_col
    cate_random = causal_forest.predict(X_augmented)

    stat, p_value = mannwhitneyu(cate, cate_random, alternative='two-sided')

    print(f"Mann-Whitney U-check p: {p_value.mean():.4f}")

    if p_value < 0.05:
        print("The original ATE and the ATE after adding random variables show a significant difference.")
    else:
        print("The original ATE and the ATE after adding random variables do not show a significant difference.")

def get_data(df):

    X = df

    return X

if __name__ == '__main__':

    T_names = ['N.P.drainage','Titanium']

    Y_names = ['Total.overall.complications']

    models = ['rf']
    path = 'xxxx.xlsx' # input data path

    for i in range(1):

        df = pd.read_excel(path,i)  

        df['Age_group'] = df['Age'].apply(lambda x: 'Over40' if x > -0.128741 else 'Under40')
        df['Gender_group'] = df['Sex'].apply(lambda x: 'Male' if x == 0 else 'Female')
        
        T_binary = df[['N.P.drainage', 'Titanium']]  
        X = df.drop(columns=['N.P.drainage', 'Titanium', Y_names[i], 'Age', 'Sex','Age_group','Gender_group'])
        Y = df[Y_names[i]]  
        
        X = get_data(X)

        groups = df.groupby(['Age_group', 'Gender_group'])
        Age_groups = df.groupby(['Age_group'])
        Gender_groups = df.groupby(['Gender_group'])

        for (age_group,gender_group), group_df in groups:

            print(f"\n=== CATE ({age_group} {gender_group}) ===")

            if (age_group,gender_group) != ('Under40','Female'): 
                max_features = 0.02
                
            else: max_features = 0.1

            X_group = X.loc[group_df.index]
            T_binary_group = T_binary.loc[group_df.index]
            Y_group = Y.loc[group_df.index]

            X_train, X_test, T_binary_train, T_binary_test, Y_train, Y_test = train_test_split(
                X_group, T_binary_group, Y_group, test_size=0.2, random_state=42
            )
            
            causal_forest = CausalForest(n_estimators=100, min_samples_leaf=10, random_state=42)
            
            causal_forest.fit(X_train, T_binary_train['N.P.drainage'], Y_train)
            cate_binary = causal_forest.predict(X_test)
            cate_lower, cate_upper = causal_forest.predict_interval(X_test)

            print(f"{Y_names[i]} CATE 估计（N.P.drainage mean）: {cate_binary.mean():.4f}")
            print(f"{Y_names[i]} CATE 估计（N.P.drainage) 95% CI: [{cate_lower.mean():.4f}, {cate_upper.mean():.4f}]")

            check(cate_binary,X_train,T_binary_train,Y_train,'N.P.drainage')

            causal_forest = CausalForest(n_estimators=100, min_samples_leaf=12, max_features = max_features,random_state=42,
                                            max_depth=2)
            causal_forest.fit(X_train, T_binary_train['Titanium'], Y_train)
            cate_binary = causal_forest.predict(X_test)
            cate_lower, cate_upper = causal_forest.predict_interval(X_test)

            print(f"{Y_names[i]} CATE （Titanium mean）: {cate_binary.mean():.4f}")
            print(f"{Y_names[i]} CATE （Titanium) 95% CI: [{cate_lower.mean():.4f}, {cate_upper.mean():.4f}]")

            check(cate_binary,X_train,T_binary_train,Y_train,'Titanium')