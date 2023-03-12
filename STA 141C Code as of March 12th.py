#!/usr/bin/env python
# coding: utf-8

# In[92]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy as scipy
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
import re
import Betas as b
import multiprocessing
import validators 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations


# In[2]:


UNCODES = '/Users/benjaminsionit/Downloads/UNCODES.csv'
UN = pd.read_csv(UNCODES)

DATA = '/Users/benjaminsionit/Downloads/FINALDATA.csv'
FINALDATA = pd.read_csv(DATA)


FINALDATAWITHREGIONS = pd.merge(UN,FINALDATA, right_on = list(FINALDATA.columns)[2], left_on = list(UN.columns)[2])
#print(list(FINALDATAWITHREGIONS.columns))
#FINALDATAWITHREGIONS


# In[3]:


def SortBySubRegion(subregion):
    REGION = FINALDATAWITHREGIONS[FINALDATAWITHREGIONS['sub-region'] == subregion]
    if(sum(REGION['Year'] == 'NaN') == 0):
        return(REGION)
    else:
        print("There are NAs for the Year for the inputed sub-region")


def MeanforX(DF, colname):
    ForX = pd.DataFrame()
    for i in pd.unique(DF['Year']):
            Mean = pd.DataFrame({'Year': [i], colname: [np.mean(DF[DF['Year'] == i][colname])]})
            ForX = pd.concat([ForX, Mean])
    return(ForX)


# In[4]:


COMPILEDDATA = pd.DataFrame()
for i in pd.unique(FINALDATAWITHREGIONS['sub-region']):
    SUBREGION = SortBySubRegion(i)
    SUBREGIONDATA = pd.DataFrame(np.arange(1992,2018+1))
    SUBREGIONDATA.columns = ['Year']
    for j in SUBREGION.columns[15:]:
        SUBREGIONDATA = pd.merge(SUBREGIONDATA,MeanforX(DF = SUBREGION,colname = j))
    COMPILEDDATA = pd.concat([COMPILEDDATA, SUBREGIONDATA])


# In[5]:


COMPILEDDATA.to_csv(r'/Users/benjaminsionit/Downloads/COMPILEDDATA.csv', index=False, header=True)
FINALDATAWITHREGIONS.to_csv(r'/Users/benjaminsionit/Downloads/FINALDATAWITHREGIONS.csv', index=False, header=True)


# In[6]:


print(pd.unique(FINALDATAWITHREGIONS['sub-region']))


# In[ ]:





# In[7]:


NewFinalDataUrl  = '/Users/benjaminsionit/Downloads/NewDataFinal.csv'
final = pd.read_csv(NewFinalDataUrl)
final = final.drop(columns=['Unnamed: 0'])

y = final["Surface Temperature Change"]

listvals = list(final.columns[16:])
for i in range(len(listvals)):
    final[final.columns[16+i]] = (i+1)*final[final.columns[16+i]]
    
regionfactors = np.asmatrix(final[final.columns[16:]]).sum(axis=1)
final = final.drop(columns = final.columns[16:])
final['Region Factors'] = regionfactors

final


# In[85]:


X_variables = final.iloc[:, [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16]]

vif_data = pd.DataFrame()
vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]


while max(vif_data['VIF']) > 10:
    X_variables = X_variables.drop(columns = [vif_data['feature'][vif_data.index[vif_data['VIF'].idxmax()]]])
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_variables.columns
    vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]
    


vif_data


# In[11]:


def model_fit_calculator(y, x):
    model = sm.OLS(y, x).fit()
    sse = np.sum((model.fittedvalues - y)**2)
    ssr = np.sum((model.fittedvalues - y.mean())**2)
    sst = ssr + sse
    AIC = len(y)*np.log(sse)-np.log(len(y))+2*x.shape[1]
    BIC = len(y)*np.log(sse)-np.log(len(y)) + (x.shape[1]*np.log(len(y)))
    R_squared = 1 - (ssr/sst)
    VIF = 1 - R_squared
    return(pd.Series({"Dependent": y.name, "Independent": list(x.columns)[1:], 
                      "AIC": AIC, "BIC": BIC, "R-Squared": R_squared, "VIF": VIF}))


# In[123]:


model_info = pd.DataFrame(columns = ["Dependent", "Independent", "AIC", "BIC", "R-Squared", "VIF"])

for k in range(2,7):
    comb = list(combinations([0, 1, 2, 3, 4, 5], k))
    for j in range(len(comb)):
        model = X_variables.iloc[:, list(comb[j])]
        model_info = pd.concat([model_info, model_fit_calculator(y, model).to_frame().T], ignore_index = True)



model_info.sort_values(by = ["AIC"], ascending=True)
model_info.sort_values(by = ["BIC"], ascending=True)


# In[107]:


model_info.sort_values(by = ["AIC"], ascending=False)
model_info.sort_values(by = ["BIC"], ascending=False)


# In[142]:


best_model = np.asmatrix(X_variables.iloc[:, [1,2,3,4,5]])
a = best_model.T @ best_model
print(np.linalg.matrix_rank(a) == a.shape[1])
y = np.asmatrix(y).T


# In[146]:


# SVD decomp
start = time.time()
u, s, vh = np.linalg.svd(a,full_matrices=False)
smat = np.diag(s)
# Calculating beta
smatinv = np.diag((1/(s)))
uinv = np.linalg.inv(u)
vhinv = np.linalg.inv(vh)
beta = (vhinv @ smatinv @ uinv) @ best_model.T @ y
K = y - (best_model)@beta
var = (K.T @ K)/(model_4.shape[0] - model_4.shape[1]) # Calculating variance
std_error = var * np.diag(vhinv @ smatinv @ uinv) # Calculating standard error
end = time.time()
print(end - start, "\n\n", beta, "\n\n", std_error)


# In[ ]:




