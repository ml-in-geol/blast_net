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

prediction_output_file = os.path.join('predict_all_test.dat')

labels_file_test = argv[1] #path to labels file
preferred_model_path = argv[2] #path to model

batch_size = 24
#labels_file_test = os.path.join(model_dir, 'labels_test_{}.csv'.format(region))
#preferred_model_path = os.path.join(model_dir, 'preferred_model_plus_{}.pt'.format(region))


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
