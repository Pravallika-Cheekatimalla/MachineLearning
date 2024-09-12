#!/usr/bin/env python
# coding: utf-8

# In[7]:


import random

random.seed(2024)
data = [(random.random(), random.random()) for i in range(100)]
X = [point[0] for point in data]
Y = [point[1] for point in data]
Xbar = sum(X)/100
Ybar = sum(Y)/100
num = 0
den = 0
for k in range(100):
    num += (X[k] - Xbar) * (Y[k] - Ybar)
    den += (X[k] - Xbar) ** 2
    
a = num/den
b = Ybar - a*Xbar

[a,b]


# In[8]:


import numpy as np

x = np.array(X).reshape(-1,1)
y = np.array(Y)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x, y)
[reg.intercept_ , reg.coef_ ]


# In[ ]:




