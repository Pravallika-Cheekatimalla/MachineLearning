#!/usr/bin/env python
# coding: utf-8

# In[1]:


# the packages and minimum version requirement are listed below
'''d = {
    'numpy': '1.21.2',
    'pandas': '1.3.2',


    'sklearn': '1.0',
    'torch': '1.8',
    'torchvision': '0.9.0'
}'''


# In[2]:


import numpy as np
import torch
import torch.nn as nn


# In[3]:


get_ipython().system('pip install torchdata')


# In[4]:


import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

image_path = './'
transform = transforms.Compose([transforms.ToTensor()])

mnist_train_dataset = torchvision.datasets.MNIST(root=image_path,
                                           train=True,
                                           transform=transform,
                                           download=True)
mnist_test_dataset = torchvision.datasets.MNIST(root=image_path,
                                           train=False,
                                           transform=transform,
                                           download=False)

batch_size = 64
torch.manual_seed(1)
train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)


# In[5]:


hidden_units = [32, 16]
image_size = mnist_train_dataset[0][0].shape
##TODO
# Since images are flattened and image is 28 x 28 pixels
input_size = 28 * 28

all_layers = [nn.Flatten()]
#all_layers = []
for hidden_unit in hidden_units:
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit
##TODO add another Linear layer which maps to 10 classes
final_layer = nn.Linear(hidden_units[-1], 10)
all_layers.append(final_layer)
model = nn.Sequential(*all_layers)

model


# In[9]:


## TODO define the loss function for multiclassification task
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
torch.manual_seed(1)
num_epochs = 20
for epoch in range(num_epochs):
    accuracy_hist_train = 0
    for x_batch, y_batch in train_dl:

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        #TODO finish the 5 steps (lines) training for Deep Learning
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()

        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy_hist_train += is_correct.sum()
    accuracy_hist_train /= len(train_dl.dataset)
    print(f'Epoch {epoch}  Accuracy {accuracy_hist_train:.4f}')


# In[11]:


model=model.to('cpu')
#TODO: check the test accuracy
# Evaluating the model
test_dl = DataLoader(mnist_test_dataset, batch_size, shuffle=False)
model.eval()
accuracy_hist_test = 0
with torch.no_grad():
    for x_batch, y_batch in test_dl:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy_hist_test += is_correct.sum()

accuracy_hist_test /= len(test_dl.dataset)
print(f'Test accuracy: {accuracy_hist_test:.4f}')

