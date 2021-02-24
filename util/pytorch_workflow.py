import torch
import torch.nn.functional as F
import os
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils
import matplotlib.pyplot as plt

class NN_workflow:
    def __init__(self, model,criterion,optimizer,batch_size=32):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model=model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.batch_size = batch_size

    def data_preparation(self,data_dir=None,dset_name=None,transform=None):
        '''

        :param data_dir: directory where the dataset will be downloaded.
        :param dset_name: ML dataset on which model will be trained and tested.
        :param batch_size: batch size to be used by the dataloader.
        :param transforms: an array of transformations to be applied on the dataset,
                           default is conversion to tensor
        :return: a pytorch dataloader object.
        '''

        ''' STEP 1: LOADING DATASET '''
        if data_dir is None:
            data_dir=os.getcwd()
        if dset_name is None:
            print('No dataset to be processed')
            return

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])

        dataset =eval('dset.'+dset_name.upper())
        trainset = dataset('./data_dir', train = True, transform=transform, download=True)
        testset =  dataset('./data_dir', train=False, transform=transform, download=True)

        print('length of training dataset is  ', len(trainset))
        print('length of test dataset is  ', len(testset))
        print("size of data : {}, size of target : {}".format(trainset[0][0].size(),len(trainset[0][1])))

        ''' STEP 2: MAKING DATASET ITERABLE '''

        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False)
        return train_loader, test_loader

    def train(self, train_loader, epochs=2):
        '''
        :param train_loader: data loader object containing trainset
        :param epochs: epoch for training the model
        '''
        train_loss = 0
        model.train()
        if self.optimizer is None:
            self.optimizer= torch.optim.SGD(lr=0.001)
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                train_loss += loss
                loss.backward()
                self.optimizer.step()
            train_loss /= len(train_loader)
            print('\n Epoch: {}, Train Loss: {:.4f}'.format(epoch, train_loss.item()))

    def test(self, test_loader):
        test_loss, correct = 0, 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad_(False)
            target.requires_grad_(False)
            output = self.model(data)
            loss = self.criterion(output, target)
            test_loss += loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum().float()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / float(len(test_loader.dataset))
        print('\nTest set: Loss: {:.4f}'.format(test_loss.item()))
        print('\nAccuracy: ({:.2f}%)'.format(float(accuracy)))

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    model = Net()
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    neural_net = NN_workflow(model, criterion, optimizer)
    train_loader, test_loader = neural_net.data_preparation(dset_name="MNIST")
    neural_net.train(train_loader)
    neural_net.test(test_loader)