import pandas as pd
import os

data = {'Name':['robert','michel','rené'],
'Age':[76,45,89],
'City':['Marseille','Paris','Bordeau']}

df = pd.DataFrame(data)

data_dir = 'data'
os.makedirs(data_dir,exist_ok=True)

file_path = os.path.join(data_dir,'sample_data.csv')

df.to_csv(file_path,index=False)