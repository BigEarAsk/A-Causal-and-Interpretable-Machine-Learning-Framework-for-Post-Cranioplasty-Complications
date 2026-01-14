import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from econml.grf import RegressionForest, CausalForest
from sklearn.model_selection import train_test_split

def draw_violin(df):

    sns.set_style("whitegrid")

    g = sns.FacetGrid(df, col="Age_group", row="Gender_group", height=4, aspect=1.2, margin_titles=True)  # 按 Age & Sex 分面
    g.map_dataframe(
        sns.violinplot, 
        x="Treatment", y="CATE", 
        palette={"N.P.drainage": "#2C7BB6", "Titanium": "#D7191C"}, 
        scale = 'width',inner="box" 
    )

    for ax, (age, sex) in zip(g.axes.flat, [(a, s) for a in df["Age_group"].unique() for s in df["Gender_group"].unique()]):
        subset = df[(df["Age_group"] == age) & (df["Gender_group"] == sex)]
        for i, row in subset.iterrows():
            x = ["N.P.drainage", "Titanium"].index(row["Treatment"])  # X 轴索引
            ax.errorbar(
                x, row["CATE"], 
                fmt='o', color="black", capsize=5, 
                alpha=0
            )

    g.set_axis_labels("Treatment", "CATE Estimate")
    g.set_titles(row_template="{row_name}", col_template="{col_name} Age Group")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("CATE Distribution by Treatment, Age, and Sex with 95% Confidence Intervals",
                    fontsize=14, fontweight="bold")
    
    plt.rcParams['pdf.fonttype'] = 42
    plt.savefig('results_img/Violin/CATE_violin.pdf',format='pdf', bbox_inches='tight')

    plt.show()

from sklearn.decomposition import PCA

def get_data(df):

    X = df

    return X

if __name__ == '__main__':

    T_names = ['N.P.drainage','Titanium']
    Y_names = ['Infection', 'Fluid.collections', 'Pneumocephalus', 'Intracranial.hemorrhage', 
               'Hydrocephalus', 'Seizures','Total.overall.complications', 'Reoperations']
    models = ['rf','extra_tree','rotation','rf','extra_tree','extra_tree','rf','rf']
    path = 'xxx.xlsx'

    for i in range(1):

        df = pd.read_excel(path,0)  
        df['CATE'] = 0
        df['Lower_CI'] = 0
        df['Upper_CI'] = 0
        df['Treatment'] = 'N.P.drainage'
        
        
        df['Age_group'] = df['Age'].apply(lambda x: 'Over40' if x > -0.128741 else 'Under40')
        df['Gender_group'] = df['Sex'].apply(lambda x: 'Male' if x == 0 else 'Female')

        df2 = df.copy(deep=True)
        df2['Treatment'] = 'Titanium'

        T_binary = df[['N.P.drainage', 'Titanium']]  
        X = df.drop(columns=['N.P.drainage', 'Titanium', Y_names[i], 'Age', 'Sex','Age_group','Gender_group',
                             'Treatment','CATE','Lower_CI','Upper_CI'])
        Y = df[Y_names[i]]  
        
        X = get_data(X,True)

        groups = df.groupby(['Age_group', 'Gender_group'])
        Age_groups = df.groupby(['Age_group'])
        Gender_groups = df.groupby(['Gender_group'])

        test_index = []

        for (age_group,gender_group), group_df in groups:
            print(f"\n=== CATE ({age_group} {gender_group}) ===")

            if (age_group,gender_group) == ('Over40','Male'):
                np_max_features = 0.12
            else:
                np_max_features = 'auto'
            
            if (age_group,gender_group) != ('Under40','Female'): 
                max_features = 0.02
                
            else: max_features = 0.1


            X_group = X.loc[group_df.index]
            T_binary_group = T_binary.loc[group_df.index]
            Y_group = Y.loc[group_df.index]

            X_train, X_test, T_binary_train, T_binary_test, Y_train, Y_test = train_test_split(
                X_group, T_binary_group, Y_group, test_size=0.2, random_state=42
            )

            test_index.extend(list(X_test.index))

            causal_forest = CausalForest(n_estimators = 100, min_samples_leaf=10, 
                                         max_features = np_max_features,random_state=42)

            causal_forest.fit(X_train, T_binary_train['N.P.drainage'], Y_train)
            cate_binary = causal_forest.predict(X_test)
            cate_lower, cate_upper = causal_forest.predict_interval(X_test)
            
            df.loc[X_test.index,'CATE'] = cate_binary
            df.loc[X_test.index,'Lower_CI'] = cate_lower
            df.loc[X_test.index,'Upper_CI'] = cate_upper

            causal_forest = CausalForest(n_estimators=100, min_samples_leaf=12, max_features = max_features,random_state=42,
                                             max_depth=2)
            causal_forest.fit(X_train, T_binary_train['Titanium'], Y_train)
            cate_binary = causal_forest.predict(X_test)
            cate_lower, cate_upper = causal_forest.predict_interval(X_test)

            df2.loc[X_test.index,'CATE'] = cate_binary
            df2.loc[X_test.index,'Lower_CI'] = cate_lower
            df2.loc[X_test.index,'Upper_CI'] = cate_upper

        df_total = pd.concat([df.loc[test_index],df2.loc[test_index]],axis = 0) 
        draw_violin(df_total)
