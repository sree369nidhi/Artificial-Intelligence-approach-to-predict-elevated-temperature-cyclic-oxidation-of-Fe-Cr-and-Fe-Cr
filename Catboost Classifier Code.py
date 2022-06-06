# Time = 100, seperate catboost -- 80-20 split
# import the necessary packages

from catboost import Pool, CatBoostClassifier
import pandas as pd
from catboost import CatBoostRegressor, FeaturesData, Pool
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
%matplotlib inline

# Load data

df = pd.read_csv('/content/New_Oxy.csv') 
df

# Select only the columns we need

df = df[['Fe', 'Ni', 'Cr', 'Temperature', 'Time', 'OxiChar']]
df

# Selecting all the data where time is 100 hours

df = df[df.Time == 100]
df.drop('Time', axis=1, inplace=True)
df

# Splitting the data into train and test

from sklearn.model_selection import train_test_split

X_t1 = df.iloc[:,:-1]
y_t1 = df.iloc[:,-1]

X_train_100_8020,X_test_100_8020,y_train_100_8020,y_test_100_8020 = train_test_split(X_t1,y_t1,test_size=0.20,random_state=762)

X_train_100_8020.to_excel('X_train_100_8020.xlsx')
X_test_100_8020.to_excel('X_test_100_8020.xlsx')
y_train_100_8020.to_excel('y_train_100_8020.xlsx')
y_test_100_8020.to_excel('y_test_100_8020.xlsx')


# Defining the classifier model

model_100_8020 = CatBoostClassifier(loss_function='MultiClass', cat_features=['Temperature'])

# Training the model
model_100_8020.fit(X_train_100_8020, y_train_100_8020, eval_set=(X_test_100_8020, y_test_100_8020))


model_100_8020.get_all_params()

# Predicting the train data
preds_train_raw_p_100_8020 = model_100_8020.predict(X_train_100_8020, prediction_type = 'Probability')
preds_train_raw_p_100_8020


preds_train_raw_p_df = pd.DataFrame(preds_train_raw_p_100_8020, columns=['AA', 'AAS', 'POS'])
preds_train_raw_p_df


preds_train_raw_100_8020 = model_100_8020.predict(X_train_100_8020)
preds_train_raw_100_8020

pred_train_100_8020 = pd.DataFrame({'Original': y_train_100_8020.OxiChar.to_list(), 'Predicted': list(preds_train_raw_100_8020.reshape(-1))})
pred_train_100_8020

time_100_8020split_trainpreds = pd.concat([X_train_100_8020.copy().reset_index(), preds_train_raw_p_df, pred_train_100_8020], axis=1)
time_100_8020split_trainpreds.to_excel('time_100_8020split_preds432_on_train_df.xlsx', index=False)
time_100_8020split_trainpreds


# Predicting the test data

preds_raw_p_100_8020 = model_100_8020.predict(X_test_100_8020, prediction_type = 'Probability')
preds_raw_p_100_8020


preds_raw_p_df = pd.DataFrame(preds_raw_p_100_8020, columns=['AA', 'AAS', 'POS'])
preds_raw_p_df

X_test_100_8020


preds_raw_100_8020 = model_100_8020.predict(X_test_100_8020)
preds_raw_100_8020

pred_100_8020 = pd.DataFrame({'Original': y_test_100_8020.OxiChar.to_list(), 'Predicted': list(preds_raw_100_8020.reshape(-1))})
pred_100_8020

time_100_8020split_preds = pd.concat([X_test_100_8020.copy().reset_index(), pred_100_8020, preds_raw_p_df], axis=1)
time_100_8020split_preds.to_excel('time_100_8020split_preds432.xlsx', index=False)

