from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
import pickle
import pandas as pd
import json

num_feat = ['LARG_S_E5','D_EPF_1','D_EPF_2']
cat_feat = ['METAL']
target = 'PU_MOY_F2'

pipeline = pickle.load(open('model.pkl','rb'))
train_data = pd.read_csv("data/cleaned/train_data.csv",sep=';',decimal=",")
test_data = pd.read_csv("data/cleaned/test_data.csv",sep=';',decimal=",")


preds_train = pipeline.predict(train_data[num_feat+cat_feat])  
rmse_train  = root_mean_squared_error(train_data[target], preds_train)

preds_test = pipeline.predict(test_data[num_feat+cat_feat])
rmse_test = root_mean_squared_error(test_data[target], preds_test)
r2_train = r2_score(train_data[target], preds_train)
r2_test  =  r2_score(test_data[target], preds_test)

metrics_dict = {'r2_train':r2_train,
                'r2_test':r2_test,
                'rmse_train':rmse_train,
                'rmse_test':rmse_train} 

with open('metrics.json','w') as file:
    json.dump(metrics_dict,file,indent=4)