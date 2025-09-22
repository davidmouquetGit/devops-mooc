from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle


num_feat = ['LARG_S_E5','D_EPF_1','D_EPF_2']
cat_feat = ['METAL']
target = 'PU_MOY_F2'

num_prepro  = StandardScaler()
cat_prepro  = OneHotEncoder(handle_unknown='ignore')
prepro =  ColumnTransformer([('num',num_prepro,num_feat),('cat',cat_prepro, cat_feat)])

pipeline = Pipeline([("preprocessor",prepro),("model", LinearRegression())])
train_data = pd.read_csv("data/cleaned/train_data.csv",sep=';',decimal=",")

train_y  = train_data[target]
train_x  = train_data[num_feat+cat_feat]

pipeline.fit(train_x,train_y)

pickle.dump(pipeline,open('model.pkl','wb'))
