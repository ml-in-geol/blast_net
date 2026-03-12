import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor,ToPILImage,Lambda
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from network import SpectrogramDataset_plus, cnn_v2

f_out_train = open('training_output.dat','w')

#set params
region = 'gasc'
n_epochs = 30
n_stop = 3
labels_file_train = 'labels_plus_train_{}.csv'.format(region)
labels_file_valid = 'labels_plus_valid_{}.csv'.format(region)
labels_file_test = 'labels_plus_test_{}.csv'.format(region)

#initialize loss
valid_loss_min = np.Inf

#if 'saved_models' directory doesnt exist, create it
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

#read data
batch_size=24

#training data
training_data = SpectrogramDataset_plus(annotations_file=labels_file_train,transform=None,target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

#validation data
valid_data = SpectrogramDataset_plus(annotations_file=labels_file_valid,transform=None,target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

#test data
test_data = SpectrogramDataset_plus(annotations_file=labels_file_test,transform=None,target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#train_features, train_labels = next(iter(train_dataloader))
#specgram = train_features[0]
#label = train_labels[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = cnn_v2().to(device)
total_params = sum(p.numel() for p in model.parameters())

print('##################################################')
print('the total number of model parameters is {}'.format(total_params))
print('##################################################')

#set misfit criteria and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# train model?
train = True
#train = False

total_loss = []
loss_values = []
epochs = []

if train:

    n_inc = 0

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        print('training... currently on epoch {}'.format(epoch+1))

        running_loss = 0.0
        train_loss = 0
        valid_loss = 0

        ####################
        # train the model
        ####################

        model.train()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs
            #inputs,labels,dists,evlos,evlas = data
            inputs,labels,dists,evlos,evlas,evdps,stlos,stlas = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        ####################
        # validate the model
        ####################
        model.eval()
        for i, data in enumerate(valid_dataloader, 0):

            inputs,labels,dists,evlos,evlas,evdps,stlos,stlas = data
            #inputs,labels,dists,evlos,evlas = data

            #forward pass
            outputs = model(inputs)

            #calculate the loss
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_dataloader.sampler)
        valid_loss = valid_loss / len(valid_dataloader.sampler)

        #print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        #       epoch+1, train_loss,valid_loss))
        f_out_train.write('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\n'.format(
               epoch+1, train_loss,valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,valid_loss))

            torch.save(model.state_dict(), 'preferred_model_plus_{}.pt'.format(region))
            valid_loss_min = valid_loss
            n_inc = 0
        else:
            n_inc += 1

        if n_inc >= n_stop:
            print('the validation loss has increased of {} consecutive epochs... stopping training'.format(n_inc))
            break

        torch.save(model.state_dict(), 'saved_models/model_plus_epoch{}.pt'.format(epoch))
        
load = True
if load:

    PATH='preferred_model_plus_{}.pt'.format(region)
    model = cnn_v2().to(device)
    model.load_state_dict(torch.load(PATH))

#----------------------------------------------------------------------
# ground truth
#----------------------------------------------------------------------
#dataiter = iter(test_dataloader)
#images, labels = dataiter.__next__()

#nb = 4 #used to be 8
#outputs = model(images)
#_, predicted = torch.max(outputs, 1)

test = True
if test:

    f_out = open('predict_all_test.dat'.format(n_epochs),'w')

    #check to see how network performs
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            #images, labels = data
            #inputs,labels,dists,evlos,evlas = data
            inputs,labels,dists,evlos,evlas,evdps,stlos,stlas = data

            # calculate outputs by running images through the network
            outputs = model(inputs)
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            probs = F.softmax(outputs,dim=1)

            #for i in range(0,len(outputs)):
            for i in range(0,len(probs)):
                label_0 = labels[i][0].item()
                prob_0 = probs[i][0].item()

                #f_out.write('{:5.3f} {:5.3f} {:5.3f} {:5.3f} {:8.2f} {} {}\n'.format(
                #             evlos[i],evlas[i],stlos[i],stlas[i],dists[i],labels[i],probs[i]))
                f_out.write('{:5.3f} {:5.3f} {:5.3f} {:5.3f} {:8.2f} {:1.0f} {:2.2f}\n'.format(
                             evlos[i],evlas[i],stlos[i],stlas[i],dists[i],label_0,prob_0))

            labels_arr = labels[:,1]
            total += labels.size(0)
            correct += (predicted == labels_arr).sum().item()

    acc = (correct / total)*100.
    print('The accuracy of the trained network on the test images is {:2.2f}%'.format(acc))

f_out_train.close()
#plt.plot(loss_values)
#plt.show()
