#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install xgboost')
get_ipython().system('pip install lightgbm')
get_ipython().system('pip install shap')
get_ipython().system('pip install shapash')
get_ipython().system('pip install lime')
get_ipython().system('pip install Pillow==9.0.0')
get_ipython().system('pip install pmdarima')
get_ipython().system('pip install tkinter')


# In[ ]:


import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest,mutual_info_regression,SelectFromModel,VarianceThreshold
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap
#import shapash


# In[ ]:


Macro_MSS_UK_DERIVED=spark.read.csv('dbfs:/FileStore/tables/Macro_MSS_Master_UK.csv',header=True).toPandas()


# In[ ]:


#spark.read.table('mss.silver_insights_uk_with_covid').display()


# In[ ]:


#spark.read.table('mss.silver_insights_uk_no_covid').display()


# In[ ]:


Macro_MSS_UK_DERIVED=Macro_MSS_UK_DERIVED.sort_values(by='Date')


# In[ ]:


Master_WSE=Macro_MSS_UK_DERIVED[Macro_MSS_UK_DERIVED.Measure=='Volume Billions WSE']
Master_GBP=Macro_MSS_UK_DERIVED[Macro_MSS_UK_DERIVED.Measure=='Value Millions GBP']


# In[ ]:


Master_WSE.drop(['Geography','Measure'],axis=1,inplace=True)
import re
cols=[col for col in Master_WSE.columns if not re.compile('FCT').search(col)]
Master_WSE=Master_WSE[cols]


# In[ ]:


col_Ncov=[col for col in Master_WSE.columns if re.compile('Ncov').search(col)]
col_rem=[col for col in Master_WSE.columns if not re.compile('cov').search(col)]
col_rem.extend(col_Ncov)
Master_WSE=Master_WSE[col_rem]


# In[ ]:


Master_WSE.Date=pd.to_datetime(Master_WSE.Date)
for i in Master_WSE.columns[2:]:
    #Master_WSE[i]=pd.to_numeric(Master_WSE[i])
    Master_WSE[i]=Master_WSE[i].astype(np.float64)
    #Master_WSE[i]=Master_WSE[i].astype('float64')


# In[ ]:


Master_WSE_FCT=Master_WSE[Master_WSE.Product=='FCT'].reset_index(drop=True)
Master_WSE_FMC=Master_WSE[Master_WSE.Product=='FMC'].reset_index(drop=True)


# In[ ]:


Master_WSE_FMC.drop('Product',axis=1,inplace=True)
feature=[i for i in Master_WSE_FMC.columns if i not in (['Imperial_Share','Total_Market'])]


# In[ ]:


x_fct=Master_WSE_FCT.loc[:,feature]
y_fct=Master_WSE_FCT.loc[:,['Total_Market']]
x_fmc=Master_WSE_FMC.loc[:,feature]
y_fmc=Master_WSE_FMC.loc[:,['Total_Market']]


# In[ ]:


# correlation matrix and remove multi colinearity
'''def corelation(df,threshold):
    corelation_col=set()
    corr_matrix=df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i,j]>threshold:
                col_name=corr_matrix.columns[i]
                corelation_col.add(col_name)
    return corelation_col'''


# In[ ]:


'''corelation(x_fmc,0.80)
col_fmc=[i for i in x_fmc.columns if i in list(corelation(x_fmc,0.80))]
col_fmc'''


# In[ ]:


'''std=StandardScaler()
x_fmc_trans=std.fit_transform(x_fmc[col_fmc])
x_fmc_trans=pd.DataFrame(x_fmc_trans,columns=col_fmc)'''


# In[ ]:


min_max=MinMaxScaler()
x_fmc_trans=min_max.fit_transform(x_fmc.iloc[:,1:])
x_fmc_trans=pd.DataFrame(x_fmc_trans,columns=x_fmc.columns[1:])


# In[ ]:


model_XGB=XGBRegressor()
params_XGB = {'max_depth': [2,3,4,5,6,7,8,9,12],
              'learning_rate': [0.01, 0.05, 0.1, 0.15,0.2,0.25,0.3, 0.4],
              'subsample': np.arange(0.5, 1.5, 0.1),
              'n_estimators': [100, 200, 300, 500,600,700,800,1000]
              }
random_cv_reg = RandomizedSearchCV(estimator=model_XGB,
                             param_distributions=params_XGB,
                             scoring='r2',
                             n_jobs=-1,cv=10)
random_cv_reg.fit(x_fmc_trans, y_fmc)


# In[ ]:


print(random_cv_reg.best_score_)
print(random_cv_reg.best_estimator_)
print(random_cv_reg.best_params_)
#r2_score(model_XGB.predict(x_fmc_test),y_fmc_test)


# In[ ]:


model_XGB_final=model_XGB.set_params(**random_cv_reg.best_params_)
model_XGB_final.fit(x_fmc_trans,y_fmc)
print('training score',model_XGB_final.score(x_fmc_trans,y_fmc))
importances_XGB = pd.DataFrame(data={
    'Attribute': x_fmc_trans.columns,
    'Importance': model_XGB_final.feature_importances_
})
importances_XGB = importances_XGB.sort_values(by='Importance', ascending=False)


# In[ ]:


XGB_features=importances_XGB.nlargest(50,columns='Importance')
XGB_features.plot(x='Attribute',y='Importance',kind='barh',figsize=(10,8))


# In[ ]:


x_fmc_train,x_fmc_test,y_fmc_train,y_fmc_test=train_test_split(x_fmc_trans[XGB_features.Attribute],y_fmc,test_size=0.2,random_state=21)
model_XGB_final.fit(x_fmc_train,y_fmc_train)
model_XGB_final.feature_importances_
r2_score(model_XGB_final.predict(x_fmc_test),y_fmc_test)


# In[ ]:


model_LGB=LGBMRegressor()
param={'n_estimators':range(100,1000,50),
      'learning_rate':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.1,0.15],
      'num_leaves':range(10,200,10),
      'boosting_type':['gbdt', 'dart', 'goss'],
      'max_depth':np.random.randint(5, 60,20)}

random_cv_LGB=RandomizedSearchCV(estimator=model_LGB,param_distributions=param,scoring='r2',cv=5,n_jobs=-1)
random_cv_LGB.fit(x_fmc_trans,y_fmc)
model_LGB_final=model_LGB.set_params(**random_cv_LGB.best_params_)
model_LGB_final.fit(x_fmc_trans,y_fmc)
print('training score',model_LGB_final.score(x_fmc_trans,y_fmc))
#model_LGB_final.score(x_fmc_test,y_fmc_test)


# In[ ]:


print(random_cv_LGB.best_score_)
print(random_cv_LGB.best_estimator_)
print(random_cv_LGB.best_params_)
#r2_score(model_LGB_final.predict(x_fmc_test),y_fmc_test)


# In[ ]:


importances_LGB = pd.DataFrame(data={
    'Attribute': x_fmc_trans.columns,
    'Importance': model_LGB_final.feature_importances_
})
importances_LGB = importances_LGB.sort_values(by='Importance', ascending=False)
LGB_features=importances_LGB.nlargest(15,columns='Importance')
LGB_features.plot(x='Attribute',y='Importance',kind='barh',figsize=(10,8))
LGB_features


# In[ ]:


x_fmc_train,x_fmc_test,y_fmc_train,y_fmc_test=train_test_split(x_fmc_trans[LGB_features.Attribute],y_fmc,test_size=0.2,random_state=24)
model_LGB_final.fit(x_fmc_train,y_fmc_train)
r2_score(model_LGB_final.predict(x_fmc_test),y_fmc_test)


# In[ ]:


param_grid = {
    'max_depth': [1,2,4,5,6,7],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1,2,3,4,5,6,7,8],
    'min_samples_split': [2,4,5,7,8,10],
    'max_depth':[x for x in np.linspace(10,100,11)],
    'n_estimators': [100, 200, 300, 500,600,1000]
}
model_Random=RandomForestRegressor()
random_cv_Rndm = RandomizedSearchCV(estimator = model_Random, param_distributions = param_grid, 
                          cv = 10, n_jobs = -1, verbose = 2,scoring='r2')
random_cv_Rndm.fit(x_fmc_trans,y_fmc)
model_Random_final=model_Random.set_params(**random_cv_Rndm.best_params_)
model_Random_final.fit(x_fmc_trans,y_fmc)
print('training score',model_Random_final.score(x_fmc_trans,y_fmc))
#model_Random_final.score(x_fmc_test,y_fmc_test)


# In[ ]:


print(random_cv_Rndm.best_score_)
print(random_cv_Rndm.best_estimator_)
print(random_cv_Rndm.best_params_)
#r2_score(y_true=y_fmc_test,y_pred=model_Random_final.predict(x_fmc_test))


# In[ ]:


importances_random = pd.DataFrame(data={
    'Attribute': x_fmc_trans.columns,
    'Importance': model_Random_final.feature_importances_
})
importances_random = importances_random.sort_values(by='Importance', ascending=False)
Rndm_features=importances_random.nlargest(15,columns='Importance')
Rndm_features


# In[ ]:


import random
random.seed(21)
x_fmc_train,x_fmc_test,y_fmc_train,y_fmc_test=train_test_split(x_fmc_trans[Rndm_features.Attribute],y_fmc,test_size=0.2)
model_Random_final.fit(x_fmc_train,y_fmc_train)
r2_score(model_Random_final.predict(x_fmc_test),y_fmc_test)


# In[ ]:


explainer_shap=shap.Explainer(model_XGB.predict,x_fmc_train)
shap_values=explainer_shap(x_fmc_train)


# In[ ]:


shap_df=pd.DataFrame(shap_values.values,columns=XGB_features.Attribute)
vals = np.abs(shap_df.values).mean(0)
#vals = shap_df.values.mean(0)
shap_importance = pd.DataFrame(list(zip(XGB_features.Attribute, vals)), columns=['col_name', 'feature_importance_vals'])
shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)


# In[ ]:


shap_importance


# In[ ]:


shap.plots.waterfall(shap_values[10], max_display=15,show=False)


# In[ ]:


shap.summary_plot(shap_values,x_fmc_train[XGB_features.Attribute])


# In[ ]:


shap.summary_plot(shap_values,x_fmc_train[XGB_features.Attribute],plot_type='violin',max_display=20)
shap.plots.beeswarm(shap_values)


# In[ ]:


explainer = shap.TreeExplainer(model_XGB_final)
shap_val = explainer.shap_values(x_fmc_train)
shap.dependence_plot('Derived_Sticks_Per_Day_WSE_Ncov',shap_val,x_fmc_train,interaction_index=None)


# In[ ]:


shap.dependence_plot('Duty_Paid_Volume_WSE_SPOT_Ncov',shap_val,x_fmc_train,interaction_index=None)


# In[ ]:


fig, ax = plt.gcf(), plt.gca()
# Modifying main plot parameters
shap.dependence_plot('Producer price index',shap_val,x_fmc_train,interaction_index=None,show=False,ax=ax)
ax.set_ylabel("Market_Size", fontsize=14)
plt.show()


# In[ ]:


fig, ax = plt.gcf(), plt.gca()
plt.figure(figsize=(10, 8))
shap.dependence_plot('Price',shap_val,x_fmc_train,interaction_index=None,show=False,ax=ax)
ax.set_ylabel("Market_Size", fontsize=14)
plt.show()


# In[ ]:


x_fmc_train.display()


# In[ ]:


model_XGB=XGBRegressor()
model_XGB.fit(x_fmc_trans, y_fmc)

model_XGB.feature_importances_


# In[ ]:


Master_WSE_FMC=Master_WSE_FMC[['Date']].join(x_fmc_trans[XGB_features.Attribute])


# In[ ]:


Master_WSE_FMC['Total_Market']=Master_WSE[Master_WSE.Product=='FMC']['Total_Market'].values


# In[ ]:


Master_WSE_FMC.set_index('Date',inplace=True)


# In[ ]:


Master_WSE_FMC.fillna(Master_WSE_FMC.Total_Market.mean(),inplace=True)


# In[ ]:


Master_WSE_FMC.Total_Market.rolling(window=3).mean().plot(figsize=(10,8))


# In[ ]:


import statsmodels,pmdarima
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[ ]:


from statsmodels.tsa.stattools import adfuller
def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")
adfuller_test(Master_WSE_FMC.Total_Market)


# In[ ]:


from pmdarima.arima import ADFTest
adf_test=ADFTest(alpha=0.05)
adf_test.should_diff(Master_WSE_FCT.Total_Market)
Master_WSE_FMC['First Difference'] = Master_WSE_FMC.Total_Market-Master_WSE_FMC.Total_Market.shift(1)
adfuller_test(Master_WSE_FMC['First Difference'].dropna())
adf_test=ADFTest(alpha=0.05)
adf_test.should_diff(Master_WSE_FMC['First Difference'].dropna())


# In[ ]:


Master_WSE_FMC.drop('First Difference',axis=1,inplace=True)
Master_WSE_FMC.head()


# In[ ]:


import random
random.seed(10)
Master_WSE_FMC_Train=Master_WSE_FMC.iloc[:48,:]
Master_WSE_FMC_Test=Master_WSE_FMC.iloc[48:,:]
exog_var_FMC=Master_WSE_FMC_Train[Master_WSE_FMC_Train.columns[:-1]]
auto_arima(Master_WSE_FMC_Train.Total_Market,exogenous = exog_var_FMC,m=12, trace= False, suppress_warnings=True,start_p=0,
                            d=1,start_q=0,max_p=5,max_d=5,max_q=5,start_P=0,stationary=False,
                            D=None,start_Q=0,max_P=5,max_D=5,max_Q=5,seasonal=True,stepwise=True,random_state=20,n_fits=50)


# In[ ]:


Model_FMC_Arima =auto_arima(Master_WSE_FMC_Train.Total_Market,exogenous = exog_var_FMC,m=12, trace= False, suppress_warnings=True,start_p=0,
                            d=1,start_q=0,max_p=5,max_d=5,max_q=5,start_P=0,stationary=False,
                            D=None,start_Q=0,max_P=5,max_D=5,max_Q=5,seasonal=True,stepwise=True,random_state=20,n_fits=50)


# In[ ]:


Model_FMC_Arima.summary()


# In[ ]:


Model_FMC_Arima.predict(n_periods=12,exogenous=np.array(Master_WSE_FMC_Test[Master_WSE_FMC_Test.columns[:-1]]))
Master_WSE_FMC_Test['Predicted_Sarimax']=Model_FMC_Arima.predict(n_periods=12, exogenous =np.array(Master_WSE_FMC_Test[Master_WSE_FMC_Test.columns[:-1]]))


# In[ ]:


Master_WSE_FMC_Test.Total_Market.plot()


# In[ ]:


Master_WSE_FMC_Test[['Total_Market','Predicted_Sarimax']].plot(figsize=(10,8))


# In[ ]:


Master_WSE_FMC_Test.head()


# In[ ]:


plt.figure(figsize=(10,8))
plt.plot(Master_WSE_FMC_Train.iloc[:,-1],label='Training')
plt.plot(Master_WSE_FMC_Test.iloc[:,-2],label='Test')
plt.plot(Master_WSE_FMC_Test.iloc[:,-1],label='Predicted')
plt.show()


# In[ ]:


# plot is because of data which has more NAN values and different pattern
Master_WSE_FMC_Test.iloc[:,[-2,-1]].plot(figsize=(12,10))


# In[ ]:


Master_WSE_FMC_Test.drop('Predicted_Sarimax',axis=1,inplace=True)


# In[ ]:


get_ipython().system('pip install prophet')
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import VAR
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[ ]:


from prophet import Prophet
from sklearn.model_selection import RandomizedSearchCV,ParameterGrid


# In[ ]:


Master_WSE_FMC_Train=Master_WSE_FMC.iloc[:48,:]
Master_WSE_FMC_Test=Master_WSE_FMC.iloc[48:,:]


# In[ ]:


model_prophet=Prophet(seasonality_mode='multiplicative',weekly_seasonality=False,yearly_seasonality=False,daily_seasonality=False)
for column in Master_WSE_FMC_Train.columns[:-1]:    
    model_prophet.add_regressor(column)


# In[ ]:


model_prophet.add_seasonality(name='monthly',period=30,fourier_order=5)


# In[ ]:


Train_prophet=Master_WSE_FMC_Train.copy(deep=True)
Test_prophet=Master_WSE_FMC_Test.copy(deep=True)


# In[ ]:


Train_prophet=Train_prophet.reset_index()
Test_prophet=Test_prophet.reset_index()


# In[ ]:


Train_prophet=Train_prophet.rename(columns={'Date':'ds','Total_Market':'y'})
Test_prophet=Test_prophet.rename(columns={'Date':'ds','Total_Market':'y'})


# In[ ]:


from datetime import datetime
Train_prophet.ds=Train_prophet['ds'].dt.tz_localize(None)
Test_prophet.ds=Test_prophet['ds'].dt.tz_localize(None)
model_prophet.fit(Train_prophet)


# In[ ]:


params_grid = {'seasonality_mode':('multiplicative','additive'),
               'changepoint_prior_scale':[0.1,0.2,0.3,0.4,0.5],
              'n_changepoints' : [100,150,200]}
grid = ParameterGrid(params_grid)
cnt = 0
for p in grid:
    cnt = cnt+1

print('Total Possible Models',cnt)


# In[ ]:


import random
strt='2021-01-01'
end='2021-12-01'
model_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
for p in grid:
    test = pd.DataFrame()
    print(p)
    random.seed(0)
    train_model =Prophet(changepoint_prior_scale = p['changepoint_prior_scale'],
                         n_changepoints = p['n_changepoints'],
                         seasonality_mode = p['seasonality_mode'],
                         interval_width=0.95)
    #train_model.add_country_holidays(country_name='US')
    train_model.fit(Train_prophet)
    test_forecast = train_model.predict(Test_prophet)
    test=test_forecast[['ds','yhat']]
    Actual = Test_prophet[(Test_prophet['ds']>=strt) & (Test_prophet['ds']<=end)]
    #print(Actual['y'],test['yhat'])
    MAPE = mean_absolute_percentage_error(Actual['y'],test['yhat'])
    print('Mean Absolute Percentage Error(MAPE)------------------------------------',MAPE)
    model_parameters = model_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)


# In[ ]:


parameters = model_parameters.sort_values(by=['MAPE'])
parameters = parameters.reset_index(drop=True)
parameters.Parameters[0]


# In[ ]:


final_model = Prophet(**parameters.Parameters[0])
final_model.fit(Train_prophet)


# In[ ]:


forecast_data=final_model.predict(Test_prophet.drop(Test_prophet.columns[-1:],axis=1))
forecast_data[['ds', 'yhat', 'yhat_lower','yhat_upper']]


# In[ ]:


forecast_data[['ds', 'yhat', 'yhat_lower','yhat_upper']].plot(x='ds',y='yhat',figsize=(10,8))


# In[ ]:


ax = (Test_prophet.plot(x='ds',y='y',figsize=(10,5),title='Actual Vs Forecast'))
forecast_data.plot(x='ds',y='yhat',figsize=(10,5),title='Actual vs Forecast', ax=ax)


# In[ ]:


f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
fig = final_model.plot(forecast_data[['ds', 'yhat', 'yhat_lower','yhat_upper']],ax=ax)
plt.show()


# In[ ]:


Train_prophet.y
y_true = Test_prophet['y'].values
y_pred = forecast_data['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
print('MAE: %.3f' % mae)


# In[ ]:


plt.figure(figsize=(10,8))
plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()


# In[ ]:




