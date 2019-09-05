import torch, numpy as np, torch.nn as nn
from torch.autograd import variable
import torchvision.transforms as transforms, torchvision.datasets as dsets
import matplotlib.pyplot as plt
%matplotlib inline

train_dataset = dsets.MNIST(root = "./data", 
                            train = True,
                            transform = transforms.ToTensor(),
                            download = True)

test_dataset = dsets.MNIST(root="./data",
                           train = False, 
                           transform=transforms.ToTensor())

test_dataset[0][0].numpy().reshape(28, 28)
show_image = (test_dataset[0][0].numpy().reshape(28, 28))
show_image = train_dataset[1][0].numpy().reshape(28, 28)
plt.imshow(show_image, cmap="gray")
batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                       batch_size = batch_size,
                                       shuffle = True)

import collections
isinstance(train_loader, collections.Iterable)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = batch_size,
                                          shuffle = False)
isinstance(test_loader, collections.Iterable)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 28*28
output_dim = 10
model = LogisticRegressionModel(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
print(model.parameters())
print(len(list(model.parameters())))
