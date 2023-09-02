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


test_dir = './dataset/testing_set'

transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Grayscale(3),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load and transform the data

testset = datasets.ImageFolder(test_dir, transform=transforms)


# data loader

testloader = torch.utils.data.DataLoader(testset, batch_size=64)


model = MainModel.Net()

model.load_state_dict(torch.load('./pretrain_model/MaskDetection_model.pk'), strict=True)
model.eval()
correct = 0
total = 0
TP = 0
FN = 0
FP = 0
TN = 0
loop = tqdm(testloader, position=0, leave=True)
model.eval()  # put model in evaluation mode
for (input, label) in loop:
    output = model.forward(input)
    _, predicted = torch.max(output.data, 1)
    true_cls = predicted[predicted == label]
    false_cls = predicted[predicted != label]
    TP += (true_cls == 1).sum().item()
    TN += (true_cls == 0).sum().item()
    FN += (false_cls == 0).sum().item()
    FP += (false_cls == 1).sum().item()
    total += label.size(0)
    correct += (predicted == label).sum().item()
    loop.set_postfix(acc=(100 * correct / total))
accuracy = (TP+TN)/(TP+TN+FN+FP)
recall = TP/(TP+FN)
specificity = TN/(TN+FP)
precision = TP/(TP+FP)
f1_score = 2*recall*precision/(recall+precision)
print('Accuracy of the network on the test images: %d %%' % (100*accuracy))
print('Recall of the network on the test images: %d %%' % (100*recall))
print('Specificity of the network on the test images: %d %%' % (100*specificity))
print('Precision of the network on the test images: %d %%' % (100*precision))
print('F1 score of the network on the test images: %d %%' % (100*f1_score))

