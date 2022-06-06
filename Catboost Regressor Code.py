# Checking for best model among regression models

from pycaret.regression import *
numerical = setup(df, target = 'Mass Change', session_id=123, log_experiment=True, experiment_name='fecrni_numerical', numeric_features = ['Fe', 'Ni', 'Cr', 'Temperature', 'Time'], train_size = 0.8)
best_model = compare_models(fold=5)


# importing required libraries

import pandas as pd
from catboost import CatBoostRegressor, FeaturesData, Pool
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
%matplotlib inline

# loading data from csv file into a dataframe

df = pd.read_csv('/content/Dup_Dummy (1).csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#df['Time'] = df['Time'].apply(lambda x: round(x))
df.head()

# Splitting data into features and target data
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Splitting data into train and test data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
#test = df.sample(n = 21) 

# try validation also as whole set

#X_eval = df.iloc[:,:-1]
#y_eval = df.iloc[:,-1]

# CatBoost Regressor model with CatBoost parameters (found using hyper parameter tuning)

cat = CatBoostRegressor(iterations=4165 , learning_rate=0.03, loss_function='RMSE', max_depth=8)
cat.fit(X_train, y_train, eval_set=(X_test, y_test),plot=True)

# Predicting the test set results
preds = cat.predict(X_test)

# Calculating the mean absolute error

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, preds)
print("MAE: %f" % (mae))

# Calculating the mean Squared error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, preds)
print("MSE: %f" % (mse))

# calculating the root mean squared error

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# calculating the R-squared value
from sklearn.metrics import r2_score
print("R Squared : %f" % (r2_score(y_test, preds)))

# prinitng catboost regressor model feature importances

print(cat.feature_names_)
print(cat.feature_importances_)

# saving the model's prediction on test set into csv file
X_pred_total = X_test.copy()
#out = pd.concat([df1, fe_pred], axis=1)
X_pred_total['original_mass_change'] = y_test
X_pred_total['predicted_mass_change_on_Dummy_dataset_80_20'] = preds
#X_test1.to_excel("catboost_output_with_ER_dataset_90_10.xlsx", sheet_name='90_10')

X_pred_total.to_excel("catboost_output_with_Dummy_dataset_80_20.xlsx", sheet_name='80_20')

# plotting model's feature_importances_

import numpy as np 
import matplotlib.pyplot as plt 

fig = plt.figure(figsize = (10, 5)) 

# creating the bar plot 
plt.barh(cat.feature_names_, cat.feature_importances_) 

plt.xlabel("Percent") 
plt.ylabel("Features") 
plt.title("Feature Importance") 
plt.show()

# calculating model's cross validation score

from sklearn.model_selection import cross_val_score

scores = cross_val_score(cat, X_train, y_train, cv=5, scoring='r2') #neg_root_mean_squared_error accuracy

print(f"Cross Validation Accuracies: {scores}")

avg = np.mean(scores)
print("Avg Cross Validation Accuracy: %f" % (avg))


# plotting data correlation matrix

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(9,9))
sns.set_context("notebook", rc={'font.family':'sans-serif'})
sns.heatmap(df.corr(method='spearman'),
            cmap="YlOrBr",  # Choose a squential colormap
            #annot=jb_labels, # Label the maximum value
            annot_kws={'fontsize':14},  # Reduce size of label to fit
            fmt='',          # Interpret labels as strings
            square=True,     # Force square cells
            #vmax=40,         # Ensure same 
            #vmin=0,          # color scale
            linewidth=0.01,  # Add gridlines
            linecolor="#222",# Adjust gridline color
            #ax=ax[i],        # Arrange in subplot
            )
    
'''
ax[0].set_title('')
ax[1].set_title('')
ax[0].set_ylabel('Hour of Day')
ax[1].set_ylabel('Hour of Day')
ax[0].set_xlabel('')
ax[1].set_xlabel('Minute of Hour')
'''
#plt.tight_layout()
plt.savefig('spearman ari.png', dpi=120)