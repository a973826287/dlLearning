import torch
import numpy
import matplotlib.pyplot as plt


# training data
x_training = numpy.asarray([1,2,5,8,9,12,14,16,18,20])
y_training = numpy.asarray([1500,3500,7200,11000,12500,18500,22000,24500,28000,30500])

x_test = numpy.asarray([3,7,13,15,19])
y_test = numpy.asarray([4400,10000,19500,23500,29000])

x_data = torch.from_numpy(x_training)
y_data = torch.from_numpy(y_training)
x_test_data = torch.from_numpy(x_test)
y_test_data = torch.from_numpy(y_test)

weight = torch.tensor(numpy.random.randn(), requires_grad=True)
bias = torch.tensor(numpy.random.randn(), requires_grad=True)
print(weight)
print(bias)

def linearRegression(x):
    return x * weight + bias

predictions = linearRegression(x_data)
print(predictions)
print(y_data)

def mse(x1, x2):
    diff = x1 - x2
    return torch.sum(diff * diff)/diff.numel()

loss = mse(predictions, y_data)
print(loss)

loss.backward()

print(weight)
print(weight.grad)
print()
print(bias)
print(bias.grad)

with torch.no_grad():
    weight -= weight.grad * 1e-5
    bias -= bias *1e-5
    weight.grad.zero_()
    bias.grad.zero_()

print(weight)
print(bias)

# An epoch is one iteration over the entire input data
no_of_epochs = 10000
# How often you want to display training info.
display_interval = 200

for epoch in range(no_of_epochs):
    predictions = linearRegression(x_data)
    loss = mse(predictions, y_data)
    loss.backward()
    with torch.no_grad():
        weight -= weight.grad * 1e-5
        bias -= bias.grad * 1e-5
        weight.grad.zero_()
        bias.grad.zero_()
    if epoch % display_interval == 0:
        # calculate the cost of the current model
        predictions = linearRegression(x_data)
        loss = mse(predictions, y_data)
        print("Epoch:", '%04d' % (epoch), "loss=", "{:.8f}".format(loss), "W=", "{:.4f}".format(weight), "b=",
              "{:.4f}".format(bias))

print("=========================================================")
training_loss = mse(linearRegression(x_data), y_data)
print("Optimised:", "lost=", "{:.9f}".format(training_loss.data), \
      "W=", "{:.9f}".format(weight.data), "b=", "{:.9f}".format(bias.data))

# Plot training data on the graph
plt.plot(x_training, y_training, 'ro', label='Training data')
plt.plot(x_training, weight.data * x_training + bias.data, label='Linear')
plt.legend()
plt.show()

# Calculate testing loss
testing_loss = mse(linearRegression(x_test_data), y_test_data)
print("Testing loss=", "{:.9f}".format(testing_loss.data))
print("Absolute mean square loss difference:", "{:.9f}".format(abs(
    training_loss.data - testing_loss.data)))

# Plot testing data on the graph
plt.plot(x_test, y_test, 'bo', label='Testing data')
plt.plot(x_test, weight.data * x_test + bias.data, label='Linear')
plt.legend()
plt.show()
