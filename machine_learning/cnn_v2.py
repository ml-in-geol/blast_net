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
from sys import argv

region = argv[1]
n_epochs = 20

specgram_dir = '/work2/03152/tg824509/stampede2/psratio_asdf/afrl_data/download_data/spectrograms/'

labels_file_train = 'labels_train_{}.csv'.format(region)
labels_file_valid = 'labels_valid_{}.csv'.format(region)
labels_file_test = 'labels_test_{}.csv'.format(region)

valid_loss_min = np.Inf

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

labels_map = {
    1: "Explosion",
    0: "Earthquake",
}
classes = ('Explosion','Earthquake')

class SpectrogramDataset(Dataset):

    def __init__(self, annotations_file, specgram_dir, transform=None, target_transform=None):
        self.specgram_labels = pd.read_csv(annotations_file)
        self.specgram_dir = specgram_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.specgram_labels)

    def __getitem__(self, idx):
        specgram_path = os.path.join(self.specgram_dir, self.specgram_labels.iloc[idx, 0])
        specgram = np.load(specgram_path)
        specgram = torch.from_numpy(specgram).float()

        label = self.specgram_labels.iloc[idx, 1]

        if self.target_transform:
            label = self.target_transform(label)

        return specgram, label

training_data = SpectrogramDataset(annotations_file=labels_file_train,specgram_dir=specgram_dir,transform=None,target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

valid_data = SpectrogramDataset(annotations_file=labels_file_valid,specgram_dir=specgram_dir,transform=None,target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

test_data = SpectrogramDataset(annotations_file=labels_file_test,specgram_dir=specgram_dir,transform=None,target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

#read data
batch_size=24
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
specgram = train_features[0]
label = train_labels[0]

#plt.imshow(img[0,:,:], cmap="gray",aspect='auto')
#plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#add convolutional net

class ConvNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(3,8,9)
        self.conv2 = nn.Conv2d(8,16,5)
        self.conv3 = nn.Conv2d(16,32,3)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear((47*3*32),512)
        #self.fc2 = nn.Linear(512,64)
        #self.fc3 = nn.Linear(64,2)
        self.fc2 = nn.Linear(512,2)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

#model = NeuralNetwork().to(device)
model = ConvNN().to(device)

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#PATH = './spectrogram_n{}.pth'.format(n_epochs)
train = True
#train = False

total_loss = []
loss_values = []
epochs = []

if train:

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
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

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

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            #forward pass
            outputs = model(inputs)

            #calculate the loss
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_dataloader.sampler)
        valid_loss = valid_loss / len(valid_dataloader.sampler)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
               epoch+1, train_loss,valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,valid_loss))

            torch.save(model.state_dict(), 'model_{}.pt'.format(region))
            valid_loss_min = valid_loss

        torch.save(model.state_dict(), 'model_{}_epoch{}.pt'.format(region,epoch))
        
load = True
if load:

    PATH='model_{}.pt'.format(region)
    model = ConvNN().to(device)
    model.load_state_dict(torch.load(PATH))

    #model = ConvNN().to(device)
    #model.load_state_dict('model.pt')
    #print('Reading Trained Network')
    #model = NeuralNetwork().to(device)
    #model.load_state_dict(torch.load(PATH))
    #print('Finished Reading Trained Network')

#----------------------------------------------------------------------
# ground truth
#----------------------------------------------------------------------
dataiter = iter(test_dataloader)
images, labels = dataiter.next()
# print images
#print('********************************')
#print(labels)
#print('********************************')
#imshow(torchvision.utils.make_grid(images))

#print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(8)))

nb = 4 #used to be 8
outputs = model(images)
_, predicted = torch.max(outputs, 1)
#print(predicted)
#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                              for j in range(nb)))

validate = True
if validate:

    f_out = open('validate_n{}_{}.dat'.format(n_epochs,region),'w')

    #check to see how network performs on whole dataset
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            labels_arr = labels[:,1]
            #print('*******')
            #print(predicted)
            #print(labels_arr)

            total += labels.size(0)
            correct += (predicted == labels_arr).sum().item()
            #correct += (predicted == labels).sum().item()

    acc = (correct / total)*100.
    print('The accuracy of the trained network on the test images for {} is {:2.2f}%'.format(region,acc))
    #print('Accuracy of the network on the test images: %d %%' % (
    #   100.0 * correct / total))

#plt.plot(loss_values)
#plt.show()
