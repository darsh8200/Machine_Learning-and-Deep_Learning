import torch, numpy as np, torch.nn as nn
from torch.autograd import variable
import matplotlib.pyplot as plt

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype = np.float32)
x_train = x_train.reshape(-1, 1)
y_values = [i for i in range(11)]
y_train = np.array(y_values, dtype = np.float32)
y_train = y_train.reshape(-1, 1)
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
epochs = 100
for epoch in range(epochs):
    epoch+=1
    inputs = variable(torch.from_numpy(x_train))
    labels = variable(torch.from_numpy(y_train))
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print("epoch{}, loss {} ".format(epoch, loss.data))

predicted = model(variable(torch.from_numpy(x_train))).data.numpy()
plt.clf()
predicted
plt.plot(x_train, y_train, 'go', label="True_data", alpha = 0.5)
plt.plot(x_train, predicted, '--', label="Predictions", alpha = 0.5)
plt.legend(loc = "best")
plt.show()
