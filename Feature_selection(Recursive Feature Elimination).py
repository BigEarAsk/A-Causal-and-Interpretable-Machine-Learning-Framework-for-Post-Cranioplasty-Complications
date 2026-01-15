import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE

from tqdm import tqdm

files = ['xxxx'] # file names

plt.rcParams['pdf.fonttype'] = 42


data = pd.read_excel(f'xxxxx.xlsx') # input data path

X = data.iloc[:, :-1]  
y = data.iloc[:, -1]

if y.dtypes == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf.fit(X, y)
feature_importances = rf.feature_importances_

# sort by the importance
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_features = X.columns[sorted_idx]

auc_scores = []
reverse = True

for i in tqdm(range(X.shape[1],0,-1)):

    selector = RFE(estimator=rf, n_features_to_select=i, step=1)
    X_selected = selector.fit_transform(X, y)

    # evaluate AUC by cross-validation
    auc_fold_scores = []
    for train_idx, test_idx in cv.split(X_selected, y):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf.fit(X_train, y_train)
        
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_fold_scores.append(auc)
    
    # compute the mean of auc
    auc_scores.append(np.mean(auc_fold_scores))

if reverse:
    auc_scores = auc_scores[::-1]

plt.figure(figsize=(10, 6))
plt.plot(range(1, X.shape[1] + 1), auc_scores, marker='o', color='b', label='AUC')

# find the index of the optimal features
optimal_num_features = np.argmax(auc_scores) + 1

# draw feature points
plt.scatter(optimal_num_features, auc_scores[optimal_num_features - 1], color='red', zorder=5)
plt.annotate(f'Optimal Features: {optimal_num_features}', 
            xy=(optimal_num_features, auc_scores[optimal_num_features - 1]),
            xytext=(optimal_num_features + 1, auc_scores[optimal_num_features - 1] - 0.05),
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.title(f'AUC vs Number of Features for Total.overall.complications')
plt.xlabel('Number of Features')
plt.ylabel('AUC (Area Under Curve)')
plt.grid(True)
plt.legend()
plt.savefig(f'xxxx.pdf')

plt.show()
optimal_features = sorted_features[:optimal_num_features]
print(f"the number of the optimal features: {optimal_num_features}")
print(f"the optimal features: {optimal_features}")
