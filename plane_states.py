#Building a nn for the new plane dataset.
#I will be using min-max normalisation between [0 1]
# it will produce 18 outputs
# Import libraries
import pandas as pd
import numpy as np
import random
from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.onnx

# Import dataset from files
from numpy import isnan

data = pd.read_csv('plane_data.csv')
shape = data.shape
data = data.to_numpy()
# Data follows the format of each element = a new row
#split the data here and perform the normalisation
#use the size to split the data into N horizon in the format expected for the nn tensors, then perform the normalisation
#data is in the following form.
#[Aileron, Elevator, Rudder, Throttle, Axe, Aye, Aze, Xe, Ye, Ze, p, q, r, p_dot, q_dot, r_dot, u, v, w, theta, phi, psi]

N = 31 #receding horizon = 30, hence break into batches of 31
#there are 3998 rows of data here
sorted = []
for i in range(0,660,22):
    mini_sort = []
    for j in range(0,3999):
        d = data[j] # selecting one row.
        for t in range(i,i+22):
            mini_sort.append(d[t])
        #if condition every 31 rows to create a new instance and add the current to the array
        if (j+1)%N == 0:
            sorted.append(mini_sort)
            mini_sort = []

##tested, data is now in format: each array in a 31 length horizon of the 22 features. hence each element is 22*31
#now we need to split the data using random index
split_tr = 0.8
split_te = 0.2
index_tr = []
index_te = []
print(len(sorted))
while len(index_tr) < (split_tr * (len(sorted))):
    ran = randint(0, (len(sorted)-1))
    if ran not in index_tr:
        index_tr.append(ran)

for gg in range(0, len(sorted)):
    if gg not in index_tr:
        index_te.append(gg)

# now split the data according to the random index's you defined
train = []
test = []
for index in index_tr:
    train.append(sorted[index])
for index in index_te:
    test.append(sorted[index])

max_a = []
min_a = []
#now perform min-max normalisation on the train data set and apply the same rules to the test
#create a function to find the min and max of each value as there are 22, maybe i'd suggest setting a loop to find
#the max and min values and also adjut the datasets simultaneuesly
for index in range(0,22): #i.e. for each feature
    minimum = 10
    maximum = -10
    for dataset in train:
        for tt in range(index,len(dataset),22):
            if dataset[tt] < minimum:
                minimum = dataset[tt]
            if dataset[tt] > maximum:
                maximum = dataset[tt]
    max_a.append(maximum)
    min_a.append(minimum)
    #now apply this to both datasets
    for traindata in train:
        for jj in range(index,len(traindata),22):
            traindata[jj] = (traindata[jj] - minimum)/(maximum - minimum)
            if isnan(traindata[jj]) == True: #this is to overcome the error of too smaller calculations
                      traindata[jj] = 0
    for testdata in test:
        for ii in range(index,len(testdata),22):
            testdata[ii] = (testdata[ii] - minimum)/(maximum-minimum)
            if isnan(testdata[ii]) == True:
                      testdata[ii] = 0

#now we need to split the data into x and y values
#the y values are the last 18 values of the array and the x values are everything up to last 22
length = len(train[0])
x_train=[]
y_train =[]
y_test = []
x_test = []
for datamix in train:
    train_x = []
    train_y = []
    for i in range(0,length-18):
        train_x.append(datamix[i])
    for j in range(length-18,length):
        train_y.append(datamix[j])
    x_train.append(train_x)
    y_train.append(train_y)

for datain in test:
    test_x = []
    test_y = []
    for i in range(0,length-18):
        test_x.append(datain[i])
    for j in range(length-18,length):
        test_y.append(datain[j])
    x_test.append(test_x)
    y_test.append(test_y)

# Define a SSM class of net
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(((30*22)+4), 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 18)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        return x


x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_test = torch.Tensor(x_test)

ssm = Net()
optimizer = optim.SGD(ssm.parameters(), lr=0.001, weight_decay=0.0005)

EPOCHS = 30
print('Training network......')
for epoch in range(EPOCHS):
    combined_loss = 0
    min_error = 10
    max_error = 0
    for dd in range(len(x_train)):
        X = x_train[dd]
        y = y_train[dd]
        ssm.zero_grad()
        output = ssm(X)
        loss = (F.mse_loss(output, y))
        combined_loss += loss
        kk = loss.detach().numpy()
        if kk < min_error:
            min_error = kk
        if kk > max_error:
            max_error = kk
        loss.backward()
        optimizer.step()
    print('Epoch: ', epoch)
    print('Max error this epoch: ', max_error)
    print('Min error this epoch: ', min_error)
    print('Most recent loss', loss.detach().numpy())
    print('Total loss: ',combined_loss.detach().numpy())
    print('')

print('Training results to prevent over-fitting:')
print('')

for bb in range(3):
    print('')
    test_out = ssm(x_train[bb])
    t = test_out
    test_out = test_out.detach().numpy()
    real_vals = []
    pred_vals = []
    abs_vals = []
    per = []
    for ee in range(0,3):
        real_vals.append(((y_test[bb][ee])*(max_a[ee+4] - min_a[ee+4])) + min_a[ee+4])
        pred_vals.append((test_out[ee]*(max_a[ee+4] - min_a[ee+4])) + min_a[ee+4])
        abs_vals.append(abs(real_vals[ee]-pred_vals[ee]))
        per.append((abs_vals[ee]/abs(real_vals[ee]))*100)
    print('----------------------')
    print('Test case ', bb+1 ,': ')
    print('')
    print('  Predicted             Actual               Abs error            Percent error')
    for j in range(0,3):
        test_num = str(j+1)
        print(test_num, pred_vals[j], real_vals[j], abs_vals[j], per[j])
    print('')
    test_y = torch.Tensor(y_test)
    loss = (F.mse_loss(t, test_y[0]))
    print('Loss: ', loss.detach().numpy())
    print('')