#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy
import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [1,3,9,15,24]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1,5,100)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()


# In[ ]:




