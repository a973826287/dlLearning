import torch
import torch.nn as nn
import torch.nn.functional as f


x_data = torch.randn(90, 10)
y_data = torch.randn(90, 1)

x_test_data = torch.randn(10, 10)
y_test_data = torch.randn(10, 1)

linearRegression = nn.Linear(10, 1)
print(linearRegression.weight)
print(linearRegression.bias)

optimizer = torch.optim.SGD(linearRegression.parameters(), lr = 1e-5)

loss_func = f.mse_loss
loss = loss_func(linearRegression(x_data), y_data)
print(loss)

def mse(x1, x2):
  diff = x1 - x2
  return torch.sum(diff*diff)/diff.numel()

no_of_epochs = 5000
display_interval = 200

for epoch in range(no_of_epochs):
    predictions = linearRegression(x_data)
    loss = loss_func(predictions, y_data)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % display_interval == 0:
        predictions = linearRegression(x_data)
        loss = loss_func(predictions, y_data)
        print("Epoch:", '%04d' %(epoch), "loss=", "{:.8f}".format(loss))

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
training_loss = mse(linearRegression(x_data), y_data)
print("Optimised:", "lost=", "{:.9f}".format(training_loss.data))

testing_loss = loss_func(linearRegression(x_test_data), y_test_data)
print("Testing loss=", "{:.9f}".format(testing_loss.data))
print("Absolute mean square loss difference:", "{:.9f}".format(abs(
      training_loss.data - testing_loss.data)))
