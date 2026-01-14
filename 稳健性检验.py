# 导入必要的库
import numpy as np
import pandas as pd
# from econml.dml import DoubleMLRegressor
from econml.grf import RegressionForest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# 导入必要的库
import random
import numpy as np
import pandas as pd
# from econml.dml import DoubleMLRegressor
from doubleml import DoubleMLData, DoubleMLPLR
from econml.dml import LinearDML,NonParamDML
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import mannwhitneyu

np.random.seed(1000)
random.seed(1000)

def check(ate,ate_random):

    stat, p_value = mannwhitneyu([ate], [ate_random], alternative='two-sided')

    print(f"Original ATE: {ate:.4f}")
    print(f"ATE after adding random treatment variables: {ate_random:.4f}")
    print(f"Mann-Whitney U test p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("The original ATE and the ATE after adding random variables show a significant difference.")
    else:
        print("The original ATE and the ATE after adding random variables do not show a significant difference.")

def get_data(path,id, t_name, y_name):
    df = pd.read_excel(path, id)
    T = df[t_name]
    Y = df[y_name]
    X = df.drop(columns=[t_name, y_name])

    return X, T, Y

if __name__ == '__main__':

    T_names = ['Surgery.time','N.P.drainage','Titanium']
    
    Y_names = ['Total.overall.complications']


    path='xxxx.xlsx'  # input data path
    
    name_id = 2
    for i in range(1):

        X,T,Y = get_data(path,i,T_names[name_id],Y_names[i],0,pca_flag=True)

        # n_estimators=100,learning_rate=0.005,loss='absolute_error'
        # n_estimators=100,learning_rate=0.5,loss='quantile' -> Surgery.time
        # n_estimators=100,learning_rate=0.001,loss='squared_error',min_samples_leaf=12,max_features=0.1 -> Titanium
        # n_estimators=100,learning_rate=0.5,loss='absolute_error',min_samples_leaf=10,max_features='log2' -> NP
        model_T = GradientBoostingRegressor(n_estimators=100,learning_rate=0.001,loss='squared_error',min_samples_leaf=12,max_features=0.1)
        model_Y = GradientBoostingRegressor(n_estimators=100,learning_rate=0.001,loss='squared_error',min_samples_leaf=12,max_features=0.1)

        if T_names[name_id] == 'N.P.drainage':
            model_T = GradientBoostingRegressor(n_estimators=100,learning_rate=0.5,loss='absolute_error',min_samples_leaf=10,max_features='log2')
            model_Y = GradientBoostingRegressor(n_estimators=100,learning_rate=0.5,loss='absolute_error',min_samples_leaf=10,max_features='log2')
        # surgery.time:
        dml = LinearDML(model_y=model_Y, model_t=model_T)
        dml.fit(Y, T, X=X)
        ate = dml.ate(X,T0=0,T1=1)

        dml = LinearDML(model_y=model_Y, model_t=model_T)
        random_col = np.random.randn(len(X))  
        X_augmented = X.copy()
        X_augmented["random_feature"] = random_col
        dml.fit(Y, T, X=X_augmented)
        ate_random = dml.ate(X_augmented,T0=0,T1=1)

        check(ate,ate_random)