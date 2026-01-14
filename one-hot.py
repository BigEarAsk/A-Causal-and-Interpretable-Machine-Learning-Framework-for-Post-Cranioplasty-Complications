import pandas as pd
import os

def check(df,name):
    
    if name == 'Materials':
        df.loc[df[name] == 1,'PEEK'] = 1
        df.loc[df[name] == 2,'Composite material'] = 1
        df.loc[df[name] == 3,'Titanium'] = 1
    elif name == 'Craniectomy.indication':
        df.loc[df[name] == 1,'Trauma'] = 1
        df.loc[df[name] == 2,'ICH'] = 1
        df.loc[df[name] == 3,'Infraction'] = 1
        df.loc[df[name] == 4,'Tumor'] = 1
    else:
        df.loc[df[name] == 1,'Bilateral.skull.defect'] = 0
        df.loc[df[name] == 2,'Bilateral.skull.defect'] = 1
    
    return df

if __name__ == '__main__':

    paths = ['xxxxx.xlsx']  # input data path

    cols = [18,16,13] # insert columns_id
    iters = [3,4,1]  # define insert times
    check_names = ['Materials','Craniectomy.indication','Skull.defect.side']

    col_names = [
        ['PEEK','Composite material','Titanium'],
        ['Trauma','ICH','Infraction','Tumor'],
        ['Bilateral.skull.defect']
    ]

    for path in paths:

        df = pd.read_excel(path)
        rows = len(df.index)  # get rows 

        for num,col in enumerate(cols):  # get the start position
            name = check_names[num]     # get where to insert
            print(name)
            for dif,_ in enumerate(range(iters[num])):  # get insert times
                df.insert(col,col_names[num][dif],[0]*(rows))

            df = check(df,name)

        df.pop('Materials')
        df.pop('Skull.defect.side')
        df.pop('Craniectomy.indication')
        df.loc[df['Tumor'] != 1,'Tumor'] = 0
        
        if 'External' not in path:
            df.to_excel('xxxx.xlsx')
        else:
            df.to_excel('xxxx.xlsx')