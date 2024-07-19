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




def train_estimation(args, model, device, batch_size, full_data, full_label, current_indices, future_indices, imp_prob, all_label, train_full_loader, optimizer, epoch,
         am = True, l1_pen = True, decay = 0.99, warm_up_num_epoch = 50, l1_pen_val = 1e-5, isWeighted = True):
    model.train()
    
    ## Important initial settings

        
    beta1_a = 0.99
    beta2_a = 0.9999
    eps = 1e-8

    if epoch <= warm_up_num_epoch:
        alpha_a = 0.001
        rho_a = (alpha_a/(1+beta1_a))

    else:
        if isWeighted:
            alpha_a = 0.00001
        else:
            alpha_a = 0.0001
        rho_a = (alpha_a/(1+beta1_a)) * (decay**(epoch - warm_up_num_epoch - 1))


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
    

#     batch_size = 64
    num_iteration = int(np.floor(N/batch_size))

    ii = 0
    for iii in range(num_iteration):
        if am:
        

        
            if ii == 0:

                shuffle_future_indices = np.random.permutation(future_indices)
                shuffle_current_indices = np.random.permutation(current_indices)
            
            
            this_future_indices = shuffle_future_indices[(ii*batch_size):((ii+1)*batch_size)] 
            this_current_indices = shuffle_current_indices[(ii*batch_size):((ii+1)*batch_size)] 
        

            data0 = full_data[this_future_indices,:,:,:]
            target0 = imp_prob[this_future_indices]
            

            data = full_data[this_current_indices,:,:,:]
            target = full_label[this_current_indices]
            

            ii += 1
            if ii == num_iteration:
                ii =0
            
            ## Calculate gradients of future observations

            if epoch <= 999:
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
            if epoch <= 999:

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
                        if isWeighted:
                            Z = -((g0 - g) - torch.sign(thetas) * lambda0) *2* torch.sign(thetas)
                        else:
                            Z = 2* torch.sum(torch.sign(thetas)**2) * lambda0 - 2*torch.sum(torch.sign(thetas) * (g0-g))
                    else:
                        if isWeighted:
                            Z = -(g0 - g - 2*lambda0*thetas)*4*thetas
                        else:
                            Z = 2* torch.sum((2*thetas)**2) * lambda0 - 2*torch.sum((2*thetas) * (g0-g))
                            
                    mu_a = beta1_a * mu_a  + (1-beta1_a)*Z
                    nu_a = beta2_a * nu_a  + (1-beta2_a)*Z*Z
                    lambda1 = lambda0 - rho_a * mu_a/(torch.sqrt(nu_a/(1-beta2_a))+eps);

                if isWeighted:
                    lambda1[lambda1 < 0] = 0
                    lambda0 = lambda1
                    lambda_reshaped = []
                    with torch.no_grad():
                        for i in range(layer_len):
                            lambda_temp = torch.reshape(lambda1[sum(num_param_list[:i]):sum(num_param_list[:(i+1)])],shape_list[i])
                            lambda_reshaped.append(lambda_temp)

                else:
                    lambda1 = torch.abs(lambda1)
                    lambda0 = lambda1
                    lambda_reshaped = [lambda1]*layer_len
            else:
                lambda1 = lambda0
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
                    l12_norm += torch.sum(p.pow(2.0) * 1e-4)
            loss = loss + l12_norm
            loss.backward()
            optimizer.step()
    
    correct = 0
    model.pen = lambda0
    model.mu0 = mu_a
    model.nu0 = nu_a
    if isWeighted == False:
        print(lambda0)
    
    ## Calculate training errors
    criterion_sum = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        if len(full_label.shape) == 1:
            one_hot_encoded = torch.nn.functional.one_hot(full_label, num_classes=10).float()
        else:
            one_hot_encoded = full_label
        imp_prob = imp_prob.to(device)
        one_hot_encoded = one_hot_encoded.to(device)
        loss_train = 0        
        loss_valid = 0
        batch_size = 512
        for start in range(0, 60000, batch_size):
            batch = get_mini_batch(array_1, start, batch_size)
            data = full_data[batch].to(device)
            target = all_label[batch].to(device)
            output = model(data)
            loss_1 = criterion_sum(output, imp_prob[batch])
            loss_2 = criterion_sum(output, one_hot_encoded[batch])
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss_valid += loss_1
            loss_train += loss_2
        loss_train = loss_train/60000
        loss_valid = loss_valid/60000
        
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.0f}%), Current Loss: {:.4f}, Future Loss: {:.4f}\n'.format(
        epoch, 60000, 60000,
        100, save_loss.item(), correct, len(train_full_loader.dataset), 
        100. * correct / len(train_full_loader.dataset), loss_train, loss_valid))

    return([save_loss.item(), 100. * correct / len(train_full_loader.dataset)])



def get_mini_batch(data_index, start_index, batch_size):
    end_index = min(start_index + batch_size, len(data_index))
    return data_index[start_index:end_index]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run models with different settings.')
    parser.add_argument('-o', '--option', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7], help='Configuration option')

    args = parser.parse_args()

    if args.option == 0:
        string1 = "./output/am/imputation/mnist_FC_400_am_l1_"
        string4 = "./output/am/estimation/mnist_FC_400_am_l1_"
        Net_general = FC_Net1
        isl1_pen = True
    elif args.option == 1:
        string1 = "./output/am/imputation/mnist_FC_400_am_l2_"
        string4 = "./output/am/estimation/mnist_FC_400_am_l2_"
        Net_general = FC_Net1
        isl1_pen = False
    elif args.option == 2:
        string1 = "./output/am/imputation/mnist_FC_800_am_l1_"
        string4 = "./output/am/estimation/mnist_FC_800_am_l1_"
        Net_general = FC_Net2
        isl1_pen = True
    elif args.option == 3:
        string1 = "./output/am/imputation/mnist_FC_800_am_l2_"
        string4 = "./output/am/estimation/mnist_FC_800_am_l2_"
        Net_general = FC_Net2
        isl1_pen = False
    elif args.option == 4:
        string1 = "./output/am/imputation/mnist_FC_1600_am_l1_"
        string4 = "./output/am/estimation/mnist_FC_1600_am_l1_"
        Net_general = FC_Net3
        isl1_pen = True
    elif args.option == 5:
        string1 = "./output/am/imputation/mnist_FC_1600_am_l2_"
        string4 = "./output/am/estimation/mnist_FC_1600_am_l2_"
        Net_general = FC_Net3
        isl1_pen = False
    elif args.option == 6:
        string1 = "./output/am/imputation/mnist_cnn_am_l1_"
        string4 = "./output/am/estimation/mnist_cnn_am_l1_"
        Net_general = CNN_Net
        isl1_pen = True
    elif args.option == 7:
        string1 = "./output/am/imputation/mnist_cnn_am_l2_"
        string4 = "./output/am/estimation/mnist_cnn_am_l2_"
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
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)



    # In[53]:

    torch.manual_seed(12345)
    random.seed(12345)
    np.random.seed(12345)
    torch.cuda.manual_seed(12345)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    train_full_loader = torch.utils.data.DataLoader(dataset1, **test_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    string2 = ".pt"
    string31 = 'train_loss_hist_'
    string32 = 'train_acc_hist_'
    string33 = 'test_loss_hist_'
    string34 = 'test_acc_hist_'


    with open('full_data.pkl', 'rb') as f:
        full_data = pickle.load(f)

    with open('full_label.pkl', 'rb') as f:
        full_label = pickle.load(f)


    imp_model = Net_general().to(device)
    array_1 = np.array([*range(60000)])
    array_2 = np.array([*range(60000)])

    for jj in range(5):
        for KK in range(2):

            with open(string1  + 'future_indices_'  + f'{jj:02}' +'-'+ f'{KK:01}' + '.pkl', 'rb') as file:
                future_indices = pickle.load(file)
    
            future_indices = np.setdiff1d(array_1, future_indices)
            imp_model.load_state_dict(torch.load(string1 + f'{jj:02}' +'-'+ f'{KK:01}' +string2))

            imp_full_label = torch.empty(60000) 
            imp_full_label = imp_full_label.type('torch.LongTensor')  

            for start in range(0, 60000, args.test_batch_size):
                batch = get_mini_batch(array_1, start, args.test_batch_size)
                data = full_data[batch].to(device)
                output = imp_model(data)
                output = F.softmax(output, dim=1)
                this_pred = torch.multinomial(output, 1, replacement=True).squeeze();        
                imp_full_label[batch] = this_pred.cpu().detach()

            imp_full_label = imp_full_label.type('torch.LongTensor') 
    
        with open(string1 +'imputation_label_'  + f'{jj:02}' + '.pkl', 'wb') as f:
            pickle.dump(imp_full_label, f)



    


    for type_index in range(2):
        if type_index == 0:
            isWeighted = True
            string_type = "_weighted"
        else:
            isWeighted = False
            string_type = "_unweighted"

        for NN in range(5):    

            ## Load initial model
            model.load_state_dict(torch.load('null_model.pt'))
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
                model.pen[i] = 1e-8 ## Initialization
        

            optimizer = optim.SGD(model.parameters(), ## Optimizer for model parameters
                                      lr=0.01,
                                      momentum=0.95)

            ## Note: the update process of Lambda is written from scratch in the training function.

            scheduler1 = StepLR(optimizer, step_size=50, gamma=0.1)
            scheduler2 = StepLR(optimizer, step_size=1, gamma=0.95)

            train_loss_hist = []
            train_acc_hist = []
            test_loss_hist = []
            test_acc_hist = []

            epoch = 1
            q = 1
            jj = 0
            while True:
                if epoch == 51:   
                    if isWeighted:
                        model.pen = torch.zeros(total_params,device = device)
                        model.mu0 = torch.zeros(total_params,device = device)
                        model.nu0 = torch.zeros(total_params,device = device)
                        for i in range(total_params):
                            model.pen[i] = 1e-8 ## Initialization
                    else:
                        model.pen = torch.zeros(1,device = device)
                        model.mu0 = torch.zeros(1,device = device)
                        model.nu0 = torch.zeros(1,device = device)
                        model.pen = 1e-8

            

                if epoch > 50:            
                    if random.uniform(0,1) < q:
                        this_indices = jj % 5
                        with open(string1 +'imputation_label_'  + f'{this_indices:02}' + '.pkl', 'rb') as f:
                            imp_full_label = pickle.load(f)
                        jj = jj + 1

                    else:
                        imp_full_label = full_label
                else:
                    imp_full_label = full_label


                if epoch <= 50:   
                    train_loss = train_estimation(args, model, device, args.batch_size,full_data,full_label, array_1, array_2, imp_full_label, full_label, train_full_loader,optimizer, epoch,
                          am = True, l1_pen = True, decay = 1.0, l1_pen_val=0, isWeighted = True)
        
                else: 
                    if isWeighted == False:
                        array_2 = numpy.random.choice(np.array([*range(60000)]), 60000)
                    train_loss = train_estimation(args, model, device, args.batch_size,full_data,full_label, array_1, array_2, imp_full_label, full_label, train_full_loader,optimizer, epoch,
                          am = True, l1_pen = isl1_pen, decay = 0.95, l1_pen_val=0, isWeighted = isWeighted)

                return_list = test(model, device, test_loader)


                train_loss_hist.append(train_loss[0])
                train_acc_hist.append(train_loss[1])
                test_loss_hist.append(return_list[0])
                test_acc_hist.append(return_list[1])


                if epoch <= 50:
                    scheduler1.step()
                else:
                    scheduler2.step()
    

                if epoch >= 100:
                    torch.save(model.state_dict(), string4 + f'{NN:02}' + string_type + '_estimation' + string2) 
            
                    with open(string4  + string31  + f'{NN:02}' + string_type + '_estimation' + '.pkl', 'wb') as f:
                        pickle.dump(train_loss_hist, f)
                    with open(string4  + string32  + f'{NN:02}' + string_type + '_estimation' + '.pkl', 'wb') as f:
                        pickle.dump(train_acc_hist, f)
                    with open(string4  + string33  + f'{NN:02}' + string_type + '_estimation' + '.pkl', 'wb') as f:
                        pickle.dump(test_loss_hist, f)
                    with open(string4  + string34  + f'{NN:02}' + string_type + '_estimation' + '.pkl', 'wb') as f:
                        pickle.dump(test_acc_hist, f)
                    with open(string4  + 'pen'  + f'{NN:02}' + string_type + '_estimation' + '.pkl', 'wb') as f:
                        pickle.dump(model.pen, f)

                    break

                epoch += 1


