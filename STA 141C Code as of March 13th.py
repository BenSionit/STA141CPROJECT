#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[8]:


X_variables = final.iloc[:, [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16]]

vif_data = pd.DataFrame()
vif_data["feature"] = X_variables.columns
vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]


while max(vif_data['VIF']) > 5:
    X_variables = X_variables.drop(columns = [vif_data['feature'][vif_data.index[vif_data['VIF'].idxmax()]]])
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_variables.columns
    vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]
    


vif_data


# In[9]:


X_variables


# In[10]:


def model_fit_calculator(y, x):
    model = sm.OLS(y, x).fit()
    sse = np.sum((model.fittedvalues - y)**2)
    ssr = np.sum((model.fittedvalues - y.mean())**2)
    sst = ssr + sse
    AIC = len(y)*np.log(sse)-np.log(len(y))+2*x.shape[1]
    BIC = len(y)*np.log(sse)-np.log(len(y)) + (x.shape[1]*np.log(len(y)))
    R_squared = 1 - (ssr/sst)
    VIF = 1 - R_squared
    return(pd.Series({"Dependent": y.name, "Independent": list(x.columns)[0:], 
                      "AIC": AIC, "BIC": BIC, "R-Squared": R_squared, "VIF": VIF}))


# In[11]:


model_info = pd.DataFrame(columns = ["Dependent", "Independent", "AIC", "BIC", "R-Squared", "VIF"])

for k in range(2,6):
    comb = combinations([0, 1, 2, 3, 4], k)
    for j in comb:
        model = X_variables.iloc[:, list(j)]
        model_info = pd.concat([model_info, model_fit_calculator(y, model).to_frame().T], ignore_index = True)




# In[12]:


model_info.sort_values(by = ["AIC"], ascending=True)
model_info.sort_values(by = ["BIC"], ascending=True)


# In[13]:


best_model = np.asmatrix(X_variables.iloc[:, [0,1,2,3,4]])
a = best_model.T @ best_model
print(np.linalg.matrix_rank(a) == a.shape[1])
y = np.asmatrix(y).T
n = best_model.shape[0]
k = best_model.shape[1]
Xty = (best_model.T @ y)


# In[14]:


# SVD decomp
start = time.time()
u, s, vh = np.linalg.svd(a,full_matrices=False)
smat = np.diag(s)
smatinv = np.diag((1/(s)))
uinv = np.linalg.inv(u)
vhinv = np.linalg.inv(vh)
# Calculating beta
VhSMatU_Inv_Prod = (vhinv @ smatinv @ uinv)
beta1 = VhSMatU_Inv_Prod @ Xty
K1 = y - (best_model)@beta1
var1 = (K1.T @ K1)/(n - k) # Calculating variance
std_error1 = var1 * np.diag(VhSMatU_Inv_Prod) # Calculating standard error
end = time.time()
print(end - start, "\n\n", beta1, "\n\n", std_error1)


# In[ ]:





# In[15]:


# Cholesky decomposition
start = time.time()
L = np.linalg.cholesky(a)
Linv = np.linalg.inv(L)
LLt_Inv_Prod = (Linv.T @ Linv)
beta3 = LLt_Inv_Prod @ Xty
K3 = y - (best_model)@beta3
var3 = (K3.T @ K3)/(n-k)# Calculating variance
std_error3 = var3 * np.diag((LLt_Inv_Prod)) # Calculating standard error
end = time.time()
print(end - start, "\n\n", beta3, "\n\n", std_error3)


# In[16]:


# QR Decomposition
start = time.time()
Q, R = scipy.linalg.qr(a)
Qinv = np.linalg.inv(Q)
Rinv = np.linalg.inv(R)
RQ_Inv_Prod = (Rinv @ Qinv)
# Calculating beta
beta4 = RQ_Inv_Prod @ Xty
K4 = y - (best_model)@beta4
var4 = (K4.T @ K4)/(n-k)# Calculating variance
std_error4 = var4 * np.diag(RQ_Inv_Prod) # Calculating standard error
end = time.time()
print(end - start, "\n\n", beta4, "\n\n", std_error4)


# In[17]:


def SVD_Decomp(model,y):
    a = model.T @ model
    Xty = (model.T @ y)
    n = model.shape[0]
    k = model.shape[1]
    # SVD decomp
    start = time.time()
    u, s, vh = np.linalg.svd(a,full_matrices=False)
    smat = np.diag(s)
    smatinv = np.diag((1/(s)))
    uinv = np.linalg.inv(u)
    vhinv = np.linalg.inv(vh)
    # Calculating beta
    VhSMatU_Inv_Prod = (vhinv @ smatinv @ uinv)
    beta1 = VhSMatU_Inv_Prod @ Xty
    K1 = y - (model)@beta1
    var1 = (K1.T @ K1)/(n - k) # Calculating variance
    std_error1 = var1 * np.diag(VhSMatU_Inv_Prod) # Calculating standard error
    end = time.time()
    df = pd.DataFrame({'time': [end - start],"beta": [beta1], "standard error": [std_error1]})
    return(df)


# In[18]:


def LU_Decomp(model,y):
    a = model.T @ model
    Xty = (model.T @ y)
    n = model.shape[0]
    k = model.shape[1]
    # LU Decomposition
    start = time.time()
    P, L, U = scipy.linalg.lu(a)
    Pinv =  np.linalg.inv(P)
    Uinv =  np.linalg.inv(U)
    Linv =  np.linalg.inv(L)
    PLU_Inv_Prod = (Uinv @ Linv @ Pinv)
    # Calculating beta
    beta2 = PLU_Inv_Prod @ Xty
    K2 = y - (model)@beta2
    var2 = (K2.T @ K2)/(n-k)# Calculating variance
    std_error2 = var2 * np.diag(PLU_Inv_Prod) # Calculating standard error
    end = time.time()
    df = pd.DataFrame({'time': [end - start],"beta": [beta2], "standard error": [std_error2]})
    return(df)


# In[19]:


def Cholesky_Decomp(model,y):
    # Cholesky decomposition
    a = model.T @ model
    Xty = (model.T @ y)
    n = model.shape[0]
    k = model.shape[1]
    start = time.time()
    L = np.linalg.cholesky(a)
    Linv = np.linalg.inv(L)
    LLt_Inv_Prod = (Linv.T @ Linv)
    beta3 = LLt_Inv_Prod @ Xty
    K3 = y - (model)@beta3
    var3 = (K3.T @ K3)/(n-k)# Calculating variance
    std_error3 = var3 * np.diag((LLt_Inv_Prod)) # Calculating standard error
    end = time.time()
    df = pd.DataFrame({'time': [end - start],"beta": [beta3], "standard error": [std_error3]})
    return(df)


# In[20]:


def QR_Decomp(model,y):
    a = model.T @ model
    Xty = (model.T @ y)
    n = model.shape[0]
    k = model.shape[1]
    # QR Decomposition
    start = time.time()
    Q, R = scipy.linalg.qr(a)
    Qinv = np.linalg.inv(Q)
    Rinv = np.linalg.inv(R)
    RQ_Inv_Prod = (Rinv @ Qinv)
    # Calculating beta
    beta4 = RQ_Inv_Prod @ Xty
    K4 = y - (model)@beta4
    var4 = (K4.T @ K4)/(n-k)# Calculating variance
    std_error4 = var4 * np.diag(RQ_Inv_Prod) # Calculating standard error
    end = time.time()
    df = pd.DataFrame({'time': [end - start],"beta": [beta4], "standard error": [std_error4]})
    return(df)


# In[ ]:





# In[21]:


list(LU_Decomp(best_model,y)['beta'])


# In[22]:


list(Cholesky_Decomp(best_model,y)['beta'])


# In[23]:


list(QR_Decomp(best_model,y)['beta'])
vif_data


# In[24]:


count1 = 0
count2 = 0
count3 = 0
count4 = 0
for x in range(0,100):
    time1 = float(SVD_Decomp(best_model,y)['time'])
    time2 = float(LU_Decomp(best_model,y)['time'])
    time3 = float(Cholesky_Decomp(best_model,y)['time'])
    time4 = float(QR_Decomp(best_model,y)['time'])
    lists = np.asarray([time1,time2,time3,time4])
    minimum = np.ndarray.min(lists)
    if(time1 == minimum):
        count1 = count1 + 1
    elif(time2 == minimum):
        count2 = count2 + 1
    elif(time3 == minimum):
        count3 = count3 + 1
    elif(time4 == minimum):
        count4 = count4 + 1
print(count1,count2,count3,count4)


# In[25]:


X_variables


# In[26]:


BetaDf = pd.DataFrame(columns = ["Beta1", "Beta2", "Beta3", "Beta4", "Beta5"])
X_variables["Surface Temperature Change"] = y
for i in range(0,200):
    Simulation_Study = X_variables.iloc[np.random.randint(0, 47824,2391)]
    SimX = Simulation_Study[['Land Coverage','BC','CO','CO2','Region Factors']]
    SimY = Simulation_Study['Surface Temperature Change']
    BetaRowDf = pd.DataFrame(pd.Series.explode(Cholesky_Decomp(SimX,SimY)['beta'])).T
    BetaRowDf.columns = ["Beta1", "Beta2", "Beta3", "Beta4", "Beta5"]
    BetaDf = pd.concat([BetaDf,BetaRowDf])

[BetaHat1, BetaHat2, BetaHat3, BetaHat4, BetaHat5] = list(np.concatenate(Cholesky_Decomp(best_model,y)['beta']).flat)


# In[27]:


mpl.pyplot.hist(BetaDf['Beta1'])
plt.axvline(x = BetaHat1, color = 'r', label = 'axvline - full height')


# In[28]:


mpl.pyplot.hist(BetaDf['Beta2'])
plt.axvline(x = BetaHat2, color = 'r', label = 'axvline - full height')


# In[29]:


mpl.pyplot.hist(BetaDf['Beta3'])
plt.axvline(x = BetaHat3, color = 'r', label = 'axvline - full height')


# In[30]:


mpl.pyplot.hist(BetaDf['Beta4'])
plt.axvline(x = BetaHat4, color = 'r', label = 'axvline - full height')


# In[31]:


mpl.pyplot.hist(BetaDf['Beta5'])
plt.axvline(x = BetaHat5, color = 'r', label = 'axvline - full height')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




