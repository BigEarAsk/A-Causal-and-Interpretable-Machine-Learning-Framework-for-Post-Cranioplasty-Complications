import random
import numpy as np
import pandas as pd
from econml import TMLELearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

np.random.seed(1000)
random.seed(1000)

def check_cate_stability(cate, X_test):
    random_col = np.random.randn(len(X_test))
    cate_random = cate + random_col * 0.01  

    from scipy.stats import mannwhitneyu
    stat, p_value = mannwhitneyu(cate, cate_random, alternative='two-sided')
    print(f"[Check] Mann-Whitney U-test p value: {p_value:.4f}")
    if p_value < 0.05:
        print("The original CATE and the perturbed CATE show a significant difference.")
    else:
        print("The original CATE and the perturbed CATE does not show a significant difference.")


def get_data(path, id, t_name, y_name):
    df = pd.read_excel(path, id)

    T = df[t_name]
    Y = df[y_name]
    X = df.drop(columns=[t_name, y_name])

    return X, T, Y

if __name__ == '__main__':
    T_names = ['N.P.drainage','Titanium']
    Y_names = ['Total.overall.complications']
    path = 'xxxx.xlsx' # input data path

    for i in range(1):
        df = pd.read_excel(path, i)

        df['Age_group'] = df['Age'].apply(lambda x: 'Over40' if x > -0.128741 else 'Under40')
        df['Gender_group'] = df['Sex'].apply(lambda x: 'Male' if x == 0 else 'Female')

        T_binary = df[['N.P.drainage', 'Titanium']]
        X = df.drop(columns=['N.P.drainage', 'Titanium', Y_names[i], 'Age', 'Sex', 'Age_group', 'Gender_group'])
        Y = df[Y_names[i]]

        groups = df.groupby(['Age_group', 'Gender_group'])

        for (age_group, gender_group), group_df in groups:
            print(f"\n=== TMLE ({age_group} {gender_group}) ===")

            X_group = X.loc[group_df.index]
            T_group = T_binary.loc[group_df.index]
            Y_group = Y.loc[group_df.index]

            X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
                X_group, T_group, Y_group, test_size=0.2, random_state=42
            )

            for t_var in T_train.columns:
                print(f"\n---- {t_var} ----")

                model_y = GradientBoostingRegressor()
                model_t = GradientBoostingClassifier()

                tmle = TMLELearner(model_y=model_y, model_t=model_t, discrete_treatment=True)
                tmle.fit(Y_train, T_train[t_var], X=X_train)

                cate = tmle.effect(X_test)
                ate = np.mean(cate)
                ci_lower, ci_upper = np.percentile(cate, [2.5, 97.5])

                print(f"[{t_var}] CATE mean: {ate:.4f}")
                print(f"[{t_var}] CATE 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

                check_cate_stability(cate, X_test)
