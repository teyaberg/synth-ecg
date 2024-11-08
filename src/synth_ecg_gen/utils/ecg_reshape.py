import pandas as pd 
import numpy as np
import os

df = pd.read_csv('./ecg_dataset_small/ecg_dataset_small_gt.csv', index_col = [0])
df['label'] = np.where(df['hr']>=120*10, 1, 0)

df = df.iloc[::-1]

old_folder = 'ecg_dataset_small'
new_folder = 'ecg_dataset_small_reshape'

for _, row in df.iterrows():
    if not os.path.isfile(f"./{new_folder}/{row['hr']}_{row['ind']}.csv"):
        df2 = pd.read_csv(f"./{old_folder}/{row['hr']}_{row['ind']}.csv", header = None)
        df3 = df2.iloc[:,0:int(df2.shape[1]/3)+1]
        df4 = pd.concat([df3, pd.DataFrame(np.asarray(df2.iloc[:,int(df2.shape[1]/3)+1:int(df2.shape[1]/3+1)*2]))], axis = 0)
        df4 = pd.concat([df4, pd.DataFrame(np.asarray(df2.iloc[:,int(df2.shape[1]/3+1)*2:int(df2.shape[1]/3+1)*3]))], 
                    axis = 0)
        df4.to_csv(f"./{new_folder}/{row['hr']}_{row['ind']}.csv", header = False, index = False)
    print(row['hr'], row['ind'])