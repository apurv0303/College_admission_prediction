#Importing libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics
%matplotlib inline

#Loading dataset

df1 = pd.read_csv('Admission_Predict.csv')
df2 = pd.read_csv('Admission_Predict_Ver1.1.csv')
frames = [df1, df2]
df = pd.concat(frames)
df=df.drop(['Serial No.'], axis = 1)

#X for training and Y for target

X = df.drop(['Chance of Admit '], axis=1).values
y = df['Chance of Admit '].values

#Splitting train test 

model_ = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Training on different model

lr = LinearRegression()  
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print('Root Mean Squared Error for LinearRegression:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
model_.append(['LinearRegression', np.sqrt(metrics.mean_squared_error(y_test, y_pred))])

lasso = Lasso()  
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
print('Root Mean Squared Error for lasso:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
model_.append(['Lasso', np.sqrt(metrics.mean_squared_error(y_test, y_pred))])


ridge = Ridge()  
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print('Root Mean Squared Error for ridge:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
model_.append(['Ridge', np.sqrt(metrics.mean_squared_error(y_test, y_pred))])


en = ElasticNet()  
en.fit(X_train, y_train)
y_pred = en.predict(X_test)
print('Root Mean Squared Error for ElasticNet:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
model_.append(['ElasticNet', np.sqrt(metrics.mean_squared_error(y_test, y_pred))])


knn = KNeighborsRegressor()  
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Root Mean Squared Error for knn:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
model_.append(['knn', np.sqrt(metrics.mean_squared_error(y_test, y_pred))])


dt = DecisionTreeRegressor()  
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print('Root Mean Squared Error for DecisionTree:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
model_.append(['DecisionTree', np.sqrt(metrics.mean_squared_error(y_test, y_pred))])


svm = SVR()  
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print('Root Mean Squared Error for svm:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
model_.append(['svm', np.sqrt(metrics.mean_squared_error(y_test, y_pred))])


rf = RandomForestRegressor(n_estimators = 100, random_state = 0)   
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Root Mean Squared Error for RandomForest:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
model_.append(['RandomForest', np.sqrt(metrics.mean_squared_error(y_test, y_pred))])


models = pd.DataFrame(model_,columns=['Model', 'RMSE'])
models=models.sort_values(by=['RMSE'])
models=models.reset_index()

#Using Random forest regressor as it has least RMSE
my_chance1=[300, 110, 5, 4, 4, 9.5, 1]
creds=np.array(my_chance1)
my_chance=creds.reshape(-1, 7)
My_prediced_chance = rf.predict(my_chance)
My_prediced_chance
