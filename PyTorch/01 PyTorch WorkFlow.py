# PyTorch WorkFlow

import matplotlib.pyplot as plt
import torch
from torch import nn

# 1.data (prepare and load)
# 2.build model
# 3.fitting the model to data(training)
# 4.making predictions and evaluating a model (inference)
# 5.saving and loading a model
# 6.putting it all together

## nn contains all of PyTorch's building blocks for neural networks

## 1.Data(preparing and loading)
# Data can be almost anything...in machine learning
# Excel spreadsheet
# Images of any kind
# Videos(Youtube has lots of data...)
# Audio like songs or podcasts
# DNA
# Text

# Machine Learning is a game of two parts:
# 1.Get data into a numerical representation
# 2.Build a model to learn pattenrs in that numerical representation

# To showcase this,let's create some *known* data using the linear regression formula
# We'll use a linear regression formula to make a straight line with *known* **parameters**

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
print(X[:10])
print(y[:10])
print(len(X))
print(len(y))

### Splitting data into training and test sets(one of the most important concepts in machine learning in general)
# Let's create a training and test set with our data
# Create train/test split
train_split = int(0.8 * len(X))
print(train_split)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(len(X_train), len(y_train))
print(len(X_test), len(y_test))
print(X_train, y_train)
print()
print(X_test, y_test)


# How might we better visualize our data?
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    # Plot est data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test data")
    # Are there predictions?
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    # Plot the predictions if they exist

    # Show the legend
    plt.legend(prop={'size': 14})


plot_predictions()
plt.show()


### Build model
# Our first PyTorch model!
# Create linear regression model class

class LinearRegressionModel(nn.Module):  # <- almost everything in PyTorch inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

        # Forward method to define the computation in the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # <- "x" is the input data
        return self.weight * x + self.bias  # This is the linear regression formula


# What out model does:
# * Start with random values(weight & bias)
# * Look at training data and adjust the random values to better
# represent (or get closer to) the ideal values (the weight & bias values
# we used to create the data)

# How does it do so?
# Through two main algorithms:
# 1.Gradient descent
# 2.Backpropagation


### PyTorch model building essentials

# * torch.nn - contains all of the buildings for computational graphs (neural networks can be considered a computational graph)
# * torch.nn.Parameter - what parameters should our model try and learn, often a PyTorch layer from torch.nn will set these for us
# * torch.nn.Modules - The base class for all neural network modules,if you subclass it,you should overwrite forward()
# * torch.optim - this where the optimizers in PyTorch live,they will help with gradient descent
# * def forward() - All nn.Module classes require you to overwrite forward(),this method defines what happens in the forward computation


### Checking the contents of our PyTorch model
# Create a random seed
torch.manual_seed(42)

# Create an instance of the model(this is a subclass of nn.Module)
model_0 = LinearRegressionModel()
print(model_0.state_dict())
# Check out the parameters
print(list(model_0.parameters()))
print(model_0)

# List named parameters
print(model_0.state_dict())

### Making prediction using 'torch.inference_mode()'

# To check our model's predictive power,let's see how well it predicts 'y_test' based on 'X_test'
# When we pass data through our model,it's going to run it through the 'forward()' method.

# Make predictions with model
with torch.inference_mode():
    y_preds = model_0(X_test)
print(y_preds)
print(y_test)
# with torch.no_grad():
#     y_pred = model_0(X_test)
plot_predictions(predictions=y_preds)
plt.show()

### Train model
# The whole idea of training is for a model to move from some *unknown* parameters (these may be random) to some *known* parameters.
# Or in other words from a poor representation of the data to a better representation of the data
# One way to measure how poor or how wrong your model predictions are is to use a loss function.
# * Note: Loss function may also be called cost function or criterion in different areas.

# Things we need to train:

# ** Loss function: ** A function to measure how wrong your model's predictions are to the ideal outputs,lower is better.
# ** Optimizer: ** Takes into account the loss of a model and adjusts the model's parameters (e.g weight & bias in our case) to improve the loss function.
# And specifically for PyTorch,we need:
# * A training loop
# * A testing loop
print(list(model_0.parameters()))

# A parameter is a value that the model sets itself

print(model_0.state_dict())

# Setup a loss function
loss_fn = nn.L1Loss()

# Setup an optimizer(stochastic gradient descent)
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)  # lr = learning rate = possibly the most important learning hyperparameter you can set
# Hyperparameter is a value that us as a data scientist or a ML engineer set ourselves

## Building a training loop (and a testing loop) in PyTorch
# A couple of things we need in a training loop:
##
# 0.Loop through the data and do...
# 1.Forward pass(this involves data moving through our model's 'forward()' functions) to make predictions on data - also called forward propagation
# 2.Calculate the loss (compare forward pass predictions to ground truth labels)
# 3.Optimizer zero grad
# 4.Loss backward - move backwards through the network to calculate the gradients of each of the parameters of our model with resepct to the loss
# 5.Optimizer step - use the optimizer to adjust out model's parameters to try and improve the loss (** gradient descent **)
print(list(model_0.parameters()))
torch.manual_seed(42)
# An epoch is a loop through the data...(it is a hyperparameter because we've set it ourselves)
epochs = 5
### Training
# 0.Loop through the data
for epoch in range(epochs):
    # Set the model to training mode
    model_0.train()  # train mode in PyTorch sets all parameters that require gradients to require gradients

    # 1.Forward pass
    y_pred = model_0(X_train)

    # 2.Calculate the loss
    loss = loss_fn(y_pred, y_train)
    print(f"Loss:{loss}")
    # 3.Optimizer zero grad
    optimizer.zero_grad()

    # 4.Perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()
    # 5.Step the optimizer(perform gradient descent)
    optimizer.step()  # by default how the optimizer changes will accumulate through the loop so...we have to zero them above in step 3 for the next iteration of the loop

    ### Testing
    model_0.eval()  # turns off different settings in the model not needed for evaluation/testing(dropout/batch layers)
    with torch.inference_mode(): # turns off gradient tracking & a couple of things behind the scenes
        #1.Do the forward pass
        test_pred = model_0(X_test)

        #2.Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    print(f"Epoch:{epoch} | Loss:{loss} | Test Loss:{test_loss}")

    # Print out model.state_dict()
    print(model_0.state_dict())

