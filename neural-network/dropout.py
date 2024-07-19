#!/usr/bin/env python
# coding: utf-8



from __future__ import print_function
import argparse
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import statistics as st
import numpy as np
import random
import pickle

## Network structures:

## CNN_Net -- CNN 
## FC_Net1,FC_Net2,FC_Net3 -- Fully-connected networks with different numbers of hidden nodes


class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1024, 200)
        self.fc21 = nn.Linear(200, 200)
        self.fc24 = nn.Linear(200,10)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.8)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(self.dropout2(x))
        x = self.relu(x)
        x = self.fc21(self.dropout2(x))
        x = self.relu(x)
        x = self.fc24(self.dropout2(x))
        
        return x
    
class FC_Net1(nn.Module):
    def __init__(self):
        super(FC_Net1, self).__init__()

        self.test = 0
        self.pen = 0
        self.mu0 = 0
        self.nu0 = 0
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 10)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.8)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(self.dropout2(x))
        x = self.relu(x)
        x = self.fc2(self.dropout2(x))
        x = self.relu(x)
        x = self.fc3(self.dropout2(x))
        return x

class FC_Net2(nn.Module):
    def __init__(self):
        super(FC_Net2, self).__init__()
        self.test = 0
        self.pen = 0
        self.mu0 = 0
        self.nu0 = 0
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 10)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.8)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(self.dropout2(x))
        x = self.relu(x)
        x = self.fc2(self.dropout2(x))
        x = self.relu(x)
        x = self.fc3(self.dropout2(x))

        return x
    
class FC_Net3(nn.Module):
    def __init__(self):
        super(FC_Net3, self).__init__()

        self.test = 0
        self.pen = 0
        self.mu0 = 0
        self.nu0 = 0
        self.fc1 = nn.Linear(784, 1600)
        self.fc2 = nn.Linear(1600, 1600)
        self.fc3 = nn.Linear(1600, 10)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.8)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(self.dropout2(x))
        x = self.relu(x)
        x = self.fc2(self.dropout2(x))
        x = self.relu(x)
        x = self.fc3(self.dropout2(x))
        return x
    
    

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return([test_loss,100. * correct / len(test_loader.dataset)])


# In[3]:

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run models with different settings.')
    parser.add_argument('-o', '--option', type=int, choices=[0, 1, 2, 3], help='Configuration option')

    args = parser.parse_args()

    if args.option == 0:
        string1 = "./output/dropout/mnist_FC_400_dropout_"
        Net_general = FC_Net1
    elif args.option == 1:
        string1 = "./output/dropout/mnist_FC_400_dropout_"
        Net_general = FC_Net2
    elif args.option == 2:
        string1 = "./output/dropout/mnist_FC_800_dropout_"
        Net_general = FC_Net3
    elif args.option == 3:
        string1 = "./output/dropout/mnist_cnn_dropout_"
        Net_general = CNN_Net

    else:
        raise ValueError("Invalid option selected.")
    ## Some arguments
    class myargs():
        def __init__(self):
            self.batch_size = 64
            self.test_batch_size = 512
            self.no_cuda = False
            self.dry_run = False

    args = myargs()        
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    population_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)



    model = Net_general().to(device)
    torch.save(model.state_dict(), 'null_model.pt')  ## Save initial model

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

    ## Create data_loader objects with data
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    train_full_loader = torch.utils.data.DataLoader(dataset1, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    total_params = 0
    start = 0
    total_params = 0
    this_start = -1
    for param in model.parameters():
        start += 1
        if start <= this_start:
            continue
        total_params += param.numel()



    string2 = ".pt"
    string31 = 'train_loss_hist_'
    string32 = 'train_acc_hist_'
    string33 = 'test_loss_hist_'
    string34 = 'test_acc_hist_'


    # In[17]:


    for NN in range(5):
   
        model.load_state_dict(torch.load('null_model.pt'))


        optimizer = optim.SGD(model.parameters(), ## Optimizer for model parameters
                                  lr=0.01,
                                  momentum=0.95)

        ## Note: the update process of Lambda is written from scratch in the training function.
    
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    
        train_loss_hist = []
        train_acc_hist = []
        test_loss_hist = []
        test_acc_hist = []
    
        jj = 0


        epoch = 1
    
        criterion = nn.CrossEntropyLoss()
        
        while True:
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
            
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            correct = 0

            with torch.no_grad():
                for data, target in train_full_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch,  60000, 60000,
            100. , loss.item(), correct, len(train_full_loader.dataset), 100. * correct / len(train_full_loader.dataset)))
                
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #epoch,  len(data), 120000,
            #100.  / len(train_loader), loss.item(), correct, len(train_full_loader.dataset), 100. * correct / len(train_full_loader.dataset)))

            return_list = test(model, device, test_loader)
        
            train_loss = [loss.item(), 100. * correct / len(train_full_loader.dataset)]
        
            train_loss_hist.append(train_loss[0])
            train_acc_hist.append(train_loss[1])
            test_loss_hist.append(return_list[0])
            test_acc_hist.append(return_list[1])        

            if epoch >= 50:
                scheduler.step()

                if len(set(train_acc_hist[(epoch - 10):epoch])) == 1:

                    torch.save(model.state_dict(), string1 + f'{NN:02}' + string2)
                    print("new model saved!")

                    with open(string1  + string31  + f'{NN:02}' + '.pkl', 'wb') as f:
                        pickle.dump(train_loss_hist, f)
                    with open(string1  + string32  + f'{NN:02}' + '.pkl', 'wb') as f:
                        pickle.dump(train_acc_hist, f)
                    with open(string1  + string33  + f'{NN:02}' + '.pkl', 'wb') as f:
                        pickle.dump(test_loss_hist, f)
                    with open(string1  + string34  + f'{NN:02}' + '.pkl', 'wb') as f:
                        pickle.dump(test_acc_hist, f)
                    with open(string1  + 'pen'  + f'{NN:02}' + '.pkl', 'wb') as f:
                        pickle.dump(model.pen, f)
                        
                    break
        
            epoch += 1





