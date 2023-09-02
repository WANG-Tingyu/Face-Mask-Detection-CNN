import torch
import numpy as np
from torch import nn, optim
from networks.network import *
from losses.loss import *
import sys
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_epochs = args.epochs
        self.cuda = args.cuda

        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        self.network = Net()
        self.losses = LossFunctions(self.cuda)
        if self.cuda:
            self.network = self.network.cuda()

    def train_epoch(self, optimizer, data_loader):
        """Train the model for one epoch
        Args:
            optimizer: (Optim) optimizer to use in backpropagation
            data_loader: (DataLoader) corresponding loader containing the training data
        Returns:
            average of all loss values, accuracy, nmi
        """
        self.network.train()
        total_loss = 0.
        num_batches = 0.
        loss_function = nn.CrossEntropyLoss()
        # iterate over the dataset
        for data, labels in tqdm(data_loader):
            if self.cuda == 1:
                data = data.cuda()

            optimizer.zero_grad()

            # forward call
            out_net = self.network(data)

            # print(out_net)
            # print(torch.transpose(out_net.float()).shape)
            # sys.exit()
            loss = loss_function(out_net.float().view(1, -1), labels.float().view(1, -1))
            total_loss += loss
            # perform backpropagation
            loss.backward()
            optimizer.step()
            self.check_model_params()
            # for i in self.network.parameters():
            #     print(i)
            #     break
            # print(next(self.network.parameters()))
            num_batches += 1
            # print(total_loss)
        return total_loss

    def train(self, train_loader, val_loader):
        optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_epoch(optimizer, train_loader)
            val_loss = self.test(val_loader, False)
            # if verbose then print specific information about training
            print("\n(Epoch {} / {})".format(epoch, self.num_epochs))
            print("Train - Loss: {:.5f};".format(train_loss))
            print("Val - Loss: {:.5f};".format(val_loss))

        torch.save(self.network.state_dict(), './data/pretrain_model.pk')

    def test(self, data_loader, return_loss=False):
        """Test the model with new data
        Args:
            data_loader: (DataLoader) corresponding loader containing the test/validation data
            return_loss: (boolean) whether to return the average loss values
        Return:
            accuracy and nmi for the given test data
        """
        self.network.eval()
        total_loss = 0.
        # num_batches = 0.
        total = 0.
        correct = 0.
        loop = tqdm(data_loader, position=0, leave=True)
        with torch.no_grad():
            for data, labels in data_loader:
                output = self.network(data)
                loss_function = nn.CrossEntropyLoss()
                loss = loss_function(output.float().view(1, -1), labels.float().view(1, -1))
                total_loss += loss
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loop.set_postfix(acc=(100 * correct / total))
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
        return total_loss

    def check_model_params(self):
        for parms in self.network.parameters():
            print('-->name:', parms.name, '-->grad_requirs:', parms.requires_grad, '--weight',
                  torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
