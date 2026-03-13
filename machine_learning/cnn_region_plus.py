import os
from pathlib import Path
from sys import argv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import Lambda
import torch.nn.functional as F
from network import SpectrogramDataset_plus, cnn_v2

if len(argv) != 2:
    raise SystemExit('Usage: python cnn_region_plus.py <model_dir>')

model_dir = os.path.abspath(argv[1])
region = Path(model_dir).resolve().name
n_epochs = 30
n_stop = 3
batch_size = 24

labels_file_train = os.path.join(model_dir, 'labels_train_{}.csv'.format(region))
labels_file_valid = os.path.join(model_dir, 'labels_valid_{}.csv'.format(region))
labels_file_test = os.path.join(model_dir, 'labels_test_{}.csv'.format(region))
training_output_file = os.path.join(model_dir, 'training_output.dat')
prediction_output_file = os.path.join(model_dir, 'predict_all_test.dat')
preferred_model_path = os.path.join(model_dir, 'preferred_model_plus_{}.pt'.format(region))
saved_models_dir = os.path.join(model_dir, 'saved_models')

#initialize loss
valid_loss_min = np.Inf

#if 'saved_models' directory doesnt exist, create it
if not os.path.exists(saved_models_dir):
    os.makedirs(saved_models_dir)

#read data
training_data = SpectrogramDataset_plus(annotations_file=labels_file_train,transform=None,target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

valid_data = SpectrogramDataset_plus(annotations_file=labels_file_valid,transform=None,target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

test_data = SpectrogramDataset_plus(annotations_file=labels_file_test,transform=None,target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

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

with open(training_output_file, 'w') as f_out_train:
    if train:

        n_inc = 0

        for epoch in range(n_epochs):  # loop over the dataset multiple times

            print('training... currently on epoch {}'.format(epoch+1))

            train_loss = 0.0
            valid_loss = 0.0
            train_examples = 0
            valid_examples = 0

            model.train()
            for data in train_dataloader:
                inputs,labels,dists,evlos,evlas,evdps,stlos,stlas = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_size_here = inputs.size(0)
                train_loss += loss.item() * batch_size_here
                train_examples += batch_size_here

            model.eval()
            with torch.no_grad():
                for data in valid_dataloader:
                    inputs,labels,dists,evlos,evlas,evdps,stlos,stlas = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    batch_size_here = inputs.size(0)
                    valid_loss += loss.item() * batch_size_here
                    valid_examples += batch_size_here

            train_loss = train_loss / train_examples
            valid_loss = valid_loss / valid_examples

            f_out_train.write('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\n'.format(
                   epoch+1, train_loss,valid_loss))

            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,valid_loss))

                torch.save(model.state_dict(), preferred_model_path)
                valid_loss_min = valid_loss
                n_inc = 0
            else:
                n_inc += 1

            if n_inc >= n_stop:
                print('the validation loss has increased of {} consecutive epochs... stopping training'.format(n_inc))
                break

            torch.save(model.state_dict(), os.path.join(saved_models_dir, 'model_plus_epoch{}.pt'.format(epoch)))
        
load = True
if load:
    model = cnn_v2().to(device)
    model.load_state_dict(torch.load(preferred_model_path, map_location=device))

test = True
if test:

    correct = 0
    total = 0
    with open(prediction_output_file,'w') as f_out:
        with torch.no_grad():
            for data in test_dataloader:
                inputs,labels,dists,evlos,evlas,evdps,stlos,stlas = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                probs = F.softmax(outputs,dim=1)

                for i in range(0,len(probs)):
                    label_0 = labels[i][0].item()
                    prob_0 = probs[i][0].item()

                    f_out.write('{:5.3f} {:5.3f} {:5.3f} {:5.3f} {:8.2f} {:1.0f} {:2.2f}\n'.format(
                                 evlos[i],evlas[i],stlos[i],stlas[i],dists[i],label_0,prob_0))

                labels_arr = labels[:,1]
                total += labels.size(0)
                correct += (predicted == labels_arr).sum().item()

    acc = (correct / total)*100.
    print('The accuracy of the trained network on the test images is {:2.2f}%'.format(acc))
