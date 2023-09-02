from matplotlib import pyplot as plt
from tqdm import tqdm  # for making progress bar
from torch.utils.data import DataLoader
from model import MainModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

# epochs = 6  # modify the number of epochs to further train the network
# parameters
RANDOM_SEED = 1
LEARNING_RATE = 0.01
BATCH_SIZE = 60
N_EPOCHS = 10


# train the network
def train(loop, model, criterion, optimizer):
    model.train()
    train_loss = []
    for (inputs, labels) in loop:
        # get the inputs; data is a list of [inputs, labels]
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward propagation
        outputs = model(inputs)
        # loss calculation
        loss = criterion(outputs, labels)
        # backward propagation
        loss.backward()
        # weight optimization
        optimizer.step()
        train_loss.append(loss.item())
    return model, optimizer, train_loss


def training_loop(model, criterion, optimizer, trainloader, validloader, epochs):
    # train the network
    train_losses = np.zeros(epochs)
    valid_losses = np.zeros(epochs)

    for epoch in range(epochs):  # loop over the dataset multiple times
        loop = tqdm(trainloader, position=0, leave=True)
        model, optimizer, train_loss = train(loop, model, criterion, optimizer)

        model.eval()
        with torch.no_grad():
            valid_loss = []
            for data in validloader:
                # get the validset; data is a list of [inputs, labels]
                inputs, labels = data

                # forward propagation
                outputs = model(inputs)

                # loss calculation
                loss = criterion(outputs, labels)

                valid_loss.append(loss.item())

        # print statistics
        train_loss = np.mean(train_loss)
        valid_loss = np.mean(valid_loss)

        print('Epoch [%d] train loss: %.3f  valid loss: %.3f' % (epoch + 1, train_loss, valid_loss))

        # save losses
        train_losses[epoch] = train_loss
        valid_losses[epoch] = valid_loss
    print('Finished Training')
    # Plot the train loss per iteration
    plt.plot(train_losses, label='train loss')
    plt.plot(valid_losses, label='valid loss')
    plt.legend()
    plt.show()
    return model, optimizer, (train_losses, valid_losses)



train_dir = './dataset/training_set'
test_dir = './dataset/testing_set'

transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Grayscale(3),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load and transform the data
trainset = datasets.ImageFolder(train_dir, transform=transforms)
testset = datasets.ImageFolder(test_dir, transform=transforms)

num_train = len(trainset)
indices = list(range(num_train))
split = int(np.floor(0.2 * num_train))  # split around 20% from the train set to validation set
np.random.seed(0)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, sampler=train_sampler)
validloader = torch.utils.data.DataLoader(trainset, batch_size=64, sampler=valid_sampler)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)


model = MainModel.Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

model, optimizer, _ = training_loop(model, criterion, optimizer, trainloader, validloader, N_EPOCHS)
torch.save(model.state_dict(), './pretrain_model/MaskDetection_model.pk')

correct = 0
total = 0

loop = tqdm(testloader, position=0, leave=True)
model.eval()  # put model in evaluation mode
for (input, label) in loop:
    output = model.forward(input)
    _, predicted = torch.max(output.data, 1)
    total += label.size(0)
    correct += (predicted == label).sum().item()
    loop.set_postfix(acc=(100 * correct / total))
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))