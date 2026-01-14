import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import numpy as np

def scale(df,col,mean,std):
    return (df[col] - mean) / std

if __name__ == '__main__':

    # set continuous varibles
    var_name = ['Age','Skull.defect.area','DC.CP.interval','GCS','Surgery.time']

    paths = ['xxxxx.xlsx'] # input data path

    scaler = StandardScaler()

    mean = dict()  # record average value
    std = dict()   # record standard value

    # Standardize the data in the file directory
    for path in paths:
        if 'single' in path:   # individual complications
            df1 = pd.read_excel(path,0)
            df2 = pd.read_excel(path,1)
            df3 = pd.read_excel(path,2)
            df4 = pd.read_excel(path,3)
            df5 = pd.read_excel(path,4)
            df6 = pd.read_excel(path,5)
            df7 = pd.read_excel(path,6)
            
            df1['Age'] = scale(df1,'Age',42.09,15.53)
            df1['Surgery.time'] = scale(df1,'Surgery.time',168.59,77.61)
            df1['Skull.defect.area'] = scale(df1,'Skull.defect.area',119.69,57.23)
            df1['GCS'] = scale(df1,'GCS',13.02,3.33)
            df1['DC.CP.interval'] = scale(df1,'DC.CP.interval',5.96,13.98)

            df2['Age'] = scale(df2,'Age',42.09,15.53)
            df2['Surgery.time'] = scale(df2,'Surgery.time',168.59,77.61)
            df2['Skull.defect.area'] = scale(df2,'Skull.defect.area',119.69,57.23)
            df2['GCS'] = scale(df2,'GCS',13.02,3.33)
            df2['DC.CP.interval'] = scale(df2,'DC.CP.interval',5.96,13.98)

            df3['Age'] = scale(df3,'Age',42.09,15.53)
            df3['Surgery.time'] = scale(df3,'Surgery.time',168.59,77.61)
            df3['Skull.defect.area'] = scale(df3,'Skull.defect.area',119.69,57.23)
            df3['GCS'] = scale(df3,'GCS',13.02,3.33)
            df3['DC.CP.interval'] = scale(df3,'DC.CP.interval',5.96,13.98)

            df4['Age'] = scale(df4,'Age',42.09,15.53)
            df4['Surgery.time'] = scale(df4,'Surgery.time',168.59,77.61)
            df4['Skull.defect.area'] = scale(df4,'Skull.defect.area',119.69,57.23)
            df4['GCS'] = scale(df4,'GCS',13.02,3.33)
            df4['DC.CP.interval'] = scale(df4,'DC.CP.interval',5.96,13.98)

            df5['Surgery.time'] = scale(df5,'Surgery.time',168.59,77.61)
            df5['Age'] = scale(df5,'Age',42.09,15.53)
            df5['Skull.defect.area'] = scale(df5,'Skull.defect.area',119.69,57.23)
            df5['GCS'] = scale(df5,'GCS',13.02,3.33)
            df5['DC.CP.interval'] = scale(df5,'DC.CP.interval',5.96,13.98)

            df6['Surgery.time'] = scale(df6,'Surgery.time',168.59,77.61)
            df6['Age'] = scale(df6,'Age',42.09,15.53)
            df6['Skull.defect.area'] = scale(df6,'Skull.defect.area',119.69,57.23)
            df6['GCS'] = scale(df6,'GCS',13.02,3.33)
            df6['DC.CP.interval'] = scale(df6,'DC.CP.interval',5.96,13.98)

            df7['Surgery.time'] = scale(df7,'Surgery.time',168.59,77.61)
            df7['Age'] = scale(df7,'Age',42.09,15.53)
            df7['Skull.defect.area'] = scale(df7,'Skull.defect.area',119.69,57.23)
            df7['GCS'] = scale(df7,'GCS',13.02,3.33)
            df7['DC.CP.interval'] = scale(df7,'DC.CP.interval',5.96,13.98)

            with pd.ExcelWriter('标准化/时序验证-分_标准化后.xlsx') as writer:
                df1.to_excel(writer,sheet_name='Infection',index=False)
                df2.to_excel(writer,sheet_name='Fluid.collections',index=False)
                df3.to_excel(writer,sheet_name='Pneumocephalus',index=False)
                df4.to_excel(writer,sheet_name='Intracranial.hemorrhage',index=False)
                df5.to_excel(writer,sheet_name='Hydrocephalus',index=False)
                df6.to_excel(writer,sheet_name='Seizures',index=False)
                df7.to_excel(writer,sheet_name='Reoperations',index=False)

        else:  # total complication

            df1 = pd.read_excel(path)

            df1['Surgery.time'] = scale(df1,'Surgery.time',168.59,77.61)
            df1['Skull.defect.area'] = scale(df1,'Skull.defect.area',119.69,57.23)
            df1['GCS'] = scale(df1,'GCS',13.02,3.33)
            df1['DC.CP.interval'] = scale(df1,'DC.CP.interval',5.96,13.98)

            df1.to_excel('xxxx.xlsx') # save path 
