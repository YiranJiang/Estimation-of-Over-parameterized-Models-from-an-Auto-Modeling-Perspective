from __future__ import print_function
import argparse
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
import pickle
import math
from scipy import stats
from scipy.stats import kstest


## Network structures:

## CNN_Net -- CNN 
## FC_Net1,FC_Net2,FC_Net3 -- Fully-connected networks with different numbers of hidden nodes


class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()

        self.pen = 0
        self.mu0 = 0
        self.nu0 = 0
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1024, 200)
        self.fc21 = nn.Linear(200, 200)
        self.fc24 = nn.Linear(200,10)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc21(x)
        x = self.relu(x)
        x = self.fc24(x)
        
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


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

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


    def forward(self, x):

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

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


    def forward(self, x):

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

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


def get_mini_batch(data_index, start_index, batch_size):
    end_index = min(start_index + batch_size, len(data_index))
    return data_index[start_index:end_index]

def train(args, model, device, batch_size, full_data, full_label, current_indices, future_indices, imp_label, train_full_loader, optimizer, epoch,
         am = True, l1_pen = True, decay = 0.99, l1_pen_val = 1e-5, l2_pen_val = 1e-4):
    model.train()
    
    ## Important initial settings

        
    alpha_a = 0.001
    beta1_a = 0.99
    beta2_a = 0.9999
    eps = 1e-8
    rho_a = (alpha_a/(1+beta1_a)) * (decay**(epoch - 1))

    mu_a = model.mu0
    nu_a = model.nu0
    lambda0 = model.pen

    num_param_list = []
    shape_list = []
    start = 0
    this_start = -1
    for param in model.parameters():
        start += 1
        if start <= this_start:
            continue
        
        num_param_list.append(param.numel())
        shape_list.append(param.shape)
    layer_len = len(shape_list)

    
    criterion = nn.CrossEntropyLoss()
    
    N = len(current_indices)
    

    num_iteration = int(np.floor(N/batch_size))
    
    # Calculate the smallest K such that len(future_indices) * K >= len(current_indices)
    K = math.ceil(len(current_indices) / len(future_indices))

    # Replicate future_indices K times
    future_indices = np.tile(future_indices, K)

    # If needed, trim the replicated array to match the length of current_indices
    future_indices = future_indices[:len(current_indices)]


    ii = 0
    for iii in range(num_iteration):
        if am:
        
            ## Load mini-batch data from population_loader_iterator for future observations

        
            if ii == 0:
                shuffle_future_indices = np.random.permutation(future_indices)
                shuffle_current_indices = np.random.permutation(current_indices)
            
            
            this_future_indices = shuffle_future_indices[(ii*batch_size):((ii+1)*batch_size)] 
            this_current_indices = shuffle_current_indices[(ii*batch_size):((ii+1)*batch_size)] 
        

            data0 = full_data[this_future_indices,:,:,:]
            target0 = imp_label[this_future_indices]
            
            ## Load mini-batch data for current observations

            data = full_data[this_current_indices,:,:,:]
            target = full_label[this_current_indices]
            

            ii += 1
            if ii == num_iteration:
                ii =0
            
            ## Calculate gradients of future observations

            data0, target0 = data0.to(device), target0.to(device)
            optimizer.zero_grad()
            output0 = model(data0)
            loss0 = criterion(output0, target0)
            loss0.backward()
            grads = []
            start = 0
            for param in model.parameters():
                start += 1
                if start <= this_start:
                    continue
                grads.append(param.grad.view(-1))


            g0 = torch.cat(grads)
            
            
            ## Calculate gradients of current observations

            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss1 = criterion(output, target)
            loss1.backward()
            grads = []
            start = 0
            for param in model.parameters():
                start += 1
                if start <= this_start:
                    continue
                grads.append(param.grad.view(-1))
            g = torch.cat(grads)

            ## Load current model parameters 
            thetas = []
            start = 0
            for param in model.parameters():
                start += 1
                if start <= this_start:
                    continue
                thetas.append(param.view(-1))
            thetas = torch.cat(thetas)
            
            ## ADAM updates of Lambdas
            
            
            with torch.no_grad():
                if l1_pen:
                    Z = -((g0 - g) - torch.sign(thetas) * lambda0) *2* torch.sign(thetas)
                else:
                    Z = -(g0 - g - 2*lambda0*thetas)*4*thetas
                mu_a = beta1_a * mu_a  + (1-beta1_a)*Z
                nu_a = beta2_a * nu_a  + (1-beta2_a)*Z*Z
                lambda1 = lambda0 - rho_a * mu_a/(torch.sqrt(nu_a/(1-beta2_a))+eps);
            lambda1[lambda1 < 0] = 0
            lambda0 = lambda1
            lambda_reshaped = []
            with torch.no_grad():
                for i in range(layer_len):
                    lambda_temp = torch.reshape(lambda1[sum(num_param_list[:i]):sum(num_param_list[:(i+1)])],shape_list[i])
                    lambda_reshaped.append(lambda_temp)
                    
            ## Get current gradients of model parameters
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            l12_norm = 0
            i = 0
            start = 0
            for p in model.parameters():
                start += 1
                if start <= this_start:
                    continue
                if l1_pen:
                    l12_norm += torch.sum(torch.abs(p) * lambda_reshaped[i])
                else:
                    l12_norm += torch.sum(p.pow(2.0) * lambda_reshaped[i])
                i += 1

            save_loss = loss
            loss = loss + l12_norm
            loss.backward()
            
            ## Updating Model Parameters 
            optimizer.step()
            

                
        else:
            
            ## Use pre-defined single lambda, no AM included
            
            if ii == 0:
                shuffle_current_indices = np.random.permutation(current_indices)
            
            
            this_current_indices = shuffle_current_indices[(ii*batch_size):((ii+1)*batch_size)] 
            
            ## Load mini-batch data for current observations

            data = full_data[this_current_indices,:,:,:]
            target = full_label[this_current_indices]

            ii += 1
            if ii == num_iteration:
                ii =0
                
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            save_loss = loss
            l12_norm = 0
            start = 0
            for p in model.parameters():
                start += 1
                if start <= this_start:
                    continue
                    
                if l1_pen:
                    l12_norm += torch.sum(torch.abs(p) * l1_pen_val)
                else:
                    l12_norm += torch.sum(p.pow(2.0) * l2_pen_val)
            loss = loss + l12_norm
            loss.backward()
            optimizer.step()
    
    correct = 0
    model.pen = lambda0
    model.mu0 = mu_a
    model.nu0 = nu_a
    
    
    ## Calculate training errors
    
    with torch.no_grad():
        for data, target in train_full_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
        
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch,  60000, 60000,
        100. , save_loss.item(), correct, len(train_full_loader.dataset), 100. * correct / len(train_full_loader.dataset)))
  
    return([save_loss.item(), 100. * correct / len(train_full_loader.dataset)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run models with different settings.')
    parser.add_argument('-o', '--option', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7], help='Configuration option')

    args = parser.parse_args()

    if args.option == 0:
        string1 = "./output/am/imputation/mnist_FC_400_am_l1_"
        Net_general = FC_Net1
        isl1_pen = True
    elif args.option == 1:
        string1 = "./output/am/imputation/mnist_FC_400_am_l2_"
        Net_general = FC_Net1
        isl1_pen = False
    elif args.option == 2:
        string1 = "./output/am/imputation/mnist_FC_800_am_l1_"
        Net_general = FC_Net2
        isl1_pen = True
    elif args.option == 3:
        string1 = "./output/am/imputation/mnist_FC_800_am_l2_"
        Net_general = FC_Net2
        isl1_pen = False
    elif args.option == 4:
        string1 = "./output/am/imputation/mnist_FC_1600_am_l1_"
        Net_general = FC_Net3
        isl1_pen = True
    elif args.option == 5:
        string1 = "./output/am/imputation/mnist_FC_1600_am_l2_"
        Net_general = FC_Net3
        isl1_pen = False
    elif args.option == 6:
        string1 = "./output/am/imputation/mnist_cnn_am_l1_"
        Net_general = CNN_Net
        isl1_pen = True
    elif args.option == 7:
        string1 = "./output/am/imputation/mnist_cnn_am_l2_"
        Net_general = CNN_Net
        isl1_pen = False

    else:
        raise ValueError("Invalid option selected.")


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
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)



    # In[53]:

    torch.manual_seed(12345)
    random.seed(12345)
    np.random.seed(12345)
    torch.cuda.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

    ## Create data_loader objects with data
    train_full_loader = torch.utils.data.DataLoader(dataset1, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)






    with torch.no_grad():
            train_data = torch.empty((0,1,28,28))
            train_label = torch.empty((0))
            full_data = torch.empty((0,1,28,28))
            full_label = torch.empty((0))

            for data, target in train_full_loader:
                full_data = torch.cat((full_data,data.cpu().detach()),0)
                full_label = torch.cat((full_label,target.cpu().detach()),0)

            train_label = train_label.type('torch.LongTensor')       
            full_label = full_label.type('torch.LongTensor')  

    with open('full_data.pkl', 'wb') as f:
        pickle.dump(full_data, f)

    with open('full_label.pkl', 'wb') as f:
        pickle.dump(full_label, f)


    torch.manual_seed(12345)
    random.seed(12345)
    np.random.seed(12345)
    torch.cuda.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    array_1 = np.array([*range(60000)])

    string2 = ".pt"
    m_ratios = np.array([0.5,0.8,1.0,1.2])
    p_values = np.empty(len(m_ratios))
    array_2 = np.random.choice(60000, size=30000, replace=False)

    for m_ratio_index in range(len(m_ratios)):

        imp_prob = torch.zeros((10,60000))

        for KK in range(2):
            if KK == 0:
                future_indices = np.setdiff1d(array_1, array_2)
            else:
                future_indices = array_2


            current_indices = (numpy.random.choice(future_indices, math.ceil(m_ratios[m_ratio_index]*30000)))


            model = Net_general().to(device)


            start = 0
            total_params = 0
            this_start = -1
            for param in model.parameters():
                start += 1
                if start <= this_start:
                    continue
                total_params += param.numel()




            model.pen = torch.zeros(total_params,device = device)
            model.mu0 = torch.zeros(total_params,device = device)
            model.nu0 = torch.zeros(total_params,device = device)

            for i in range(total_params):
                model.pen[i] = 1e-5 ## Initialization

            optimizer = optim.SGD(model.parameters(), ## Optimizer for model parameters
                                      lr=0.01,
                                      momentum=0.95)


            scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

            epoch = 1


            while True:
                train_loss = train(args, model, device, args.batch_size, full_data,full_label, current_indices, future_indices, full_label, train_full_loader,optimizer, epoch,
                          am = True, l1_pen = isl1_pen, decay = 0.98)




                scheduler.step()



                if epoch >= 50:
                    torch.save(model.state_dict(), string1 + f'{m_ratio_index:02}' +'-'+ f'{KK:01}' + string2)
                    print("new model saved!")

                    with open(string1  + 'future_indices_'  + f'{m_ratio_index:02}' +'-'+ f'{KK:01}' + '.pkl', 'wb') as f:
                        pickle.dump(future_indices, f)

                    unseen_indices = np.setdiff1d(array_1, future_indices)
                    for start in range(0, len(unseen_indices), args.test_batch_size):
                        batch = get_mini_batch(unseen_indices, start, args.test_batch_size)
                        data = full_data[batch].to(device)
                        output = model(data)
                        output = F.softmax(output, dim=1)
                        imp_prob[:,batch] = output.t().cpu().detach()
    
                    break

                epoch += 1
            
        full_cdf = np.empty(60000)

        for i in range(60000): 

            y = full_label[i]

            this_prob = np.array(imp_prob[:,i])
            this_prob = np.insert(this_prob,0,0)

            lower_bound = np.sum(this_prob[0:(y+1)])
            upper_bound = np.sum(this_prob[0:(y+2)])

            full_cdf[i] = random.uniform(lower_bound, upper_bound)

        _, p_value = kstest(full_cdf, 'uniform', args=(0, 1))

        print("P-value:", p_value)
        p_values[m_ratio_index] = p_value

    for KK in range(2):
        NN = np.argmax(p_values)
        model.load_state_dict(torch.load(string1 + f'{NN:02}' +'-'+ f'{KK:01}' + string2))
        NN = 0
        torch.save(model.state_dict(), string1 + f'{NN:02}' +'-'+ f'{KK:01}' + string2)



    array_1 = np.array([*range(60000)])
    selected_m_ratio = m_ratios[np.argmax(p_values)]

    for NN in range(1,5):
        array_2 = np.random.choice(60000, size=30000, replace=False)

    
        for KK in range(2):
            if KK == 0:
                future_indices = np.setdiff1d(array_1, array_2)
            else:
                future_indices = array_2


            current_indices = numpy.random.choice(future_indices, math.ceil(selected_m_ratio*30000))


            model = Net_general().to(device)

            start = 0
            total_params = 0
            this_start = -1
            for param in model.parameters():
                start += 1
                if start <= this_start:
                    continue
                total_params += param.numel()


            model.pen = torch.zeros(total_params,device = device)
            model.mu0 = torch.zeros(total_params,device = device)
            model.nu0 = torch.zeros(total_params,device = device)

            for i in range(total_params):
                model.pen[i] = 1e-5 ## Initialization

            optimizer = optim.SGD(model.parameters(), ## Optimizer for model parameters
                                      lr=0.01,
                                      momentum=0.95)

            ## Note: the update process of Lambda is written from scratch in the training function.

            scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

            epoch = 1


            while True:
                train_loss = train(args, model, device, args.batch_size, full_data,full_label, current_indices, future_indices, full_label, train_full_loader,optimizer, epoch,
                          am = True, l1_pen = isl1_pen, decay = 0.98)

                scheduler.step()

                if epoch >= 50:
                    torch.save(model.state_dict(), string1 + f'{NN:02}' +'-'+ f'{KK:01}' + string2)
                    print("new model saved!")

                    with open(string1  + 'future_indices_'  + f'{NN:02}' +'-'+ f'{KK:01}' + '.pkl', 'wb') as f:
                        pickle.dump(future_indices, f)

                    break

                epoch += 1

