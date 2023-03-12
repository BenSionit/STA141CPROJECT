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


# In[2]:


UNCODES = '/Users/benjaminsionit/Downloads/UNCODES.csv'
UN = pd.read_csv(UNCODES)

DATA = '/Users/benjaminsionit/Downloads/FINALDATA.csv'
FINALDATA = pd.read_csv(DATA)


FINALDATAWITHREGIONS = pd.merge(UN,FINALDATA, right_on = list(FINALDATA.columns)[2], left_on = list(UN.columns)[2])
print(list(FINALDATAWITHREGIONS.columns))
FINALDATAWITHREGIONS


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


# In[8]:


COMPILEDDATA.to_csv(r'/Users/benjaminsionit/Downloads/COMPILEDDATA.csv', index=False, header=True)
FINALDATAWITHREGIONS.to_csv(r'/Users/benjaminsionit/Downloads/FINALDATAWITHREGIONS.csv', index=False, header=True)


# In[6]:


print(pd.unique(FINALDATAWITHREGIONS['sub-region']))


# In[7]:


print(FINALDATAWITHREGIONS)
print(COMPILEDDATA)


# In[ ]:




