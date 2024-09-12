#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np

image = cv2.imread('barbara.bmp', cv2.IMREAD_GRAYSCALE)

image_resize = cv2.resize(image, (512,512), interpolation=cv2.INTER_LINEAR)


# In[7]:


import matplotlib.pyplot as plt

X = np.array(image_resize)

image_size = X.size

A, Z, Ct = np.linalg.svd(X, full_matrices=True)
Z =  np.diag(np.ravel(Z))
C = np.transpose(Ct)


list_r = [1, 5, 10, 20, 50, 100, 200, 300, 500, 512]
size = np.array([])
plt.figure(figsize = (12, 5))

for i,r in enumerate(list_r, start=1):
    Abar = A[:, :r]
    Zbar = Z[:r, :r]
    Cbar = C[:, :r]
    Xbar = Abar @ Zbar @ np.transpose(Cbar)
    ratio = (Abar.size + Zbar.size + Cbar.size)/image_size
    size = np.append(size, ratio)
    plt.subplot(2,7,i)
    plt.imshow(Xbar, cmap='gray')
    plt.title(f'r={r}')
    plt.axis('off')
plt.show()


# In[8]:


plt.plot(list_r,size,marker='s')
plt.show()


# In[11]:


#The image appears good when the value of r is 100.

r = 100
Abar = A[:, :r]
Zbar = Z [:r, :r]
Cbar = C[:, :r]

img_size = Abar.size + Zbar.size + Cbar.size
ratio = (Abar.size + Zbar.size + Cbar.size)/image_size

print('Reduced image size: ', img_size)
print('Ratio of the reduced image size over the original image size=', ratio)


# In[82]:


#The reduced image occupies 42.87% of the space utilized by the original image.

