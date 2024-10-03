# 02. Neural Network classification with PyTorch
# Classification is a problem of predicting whether something is one thing or another
# (there can be multiple things as options)
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import torch
import torchmetrics

from sklearn.datasets import make_circles, make_blobs
from sklearn.model_selection import train_test_split
from torch import nn

from PyTorch.helper_functions import plot_decision_boundary

# 1. Make classification data and get it ready
# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)
print(len(X), len(y))
print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of y:\n {y[:5]}")
print(y)
# Make DataFrame of circle data
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})
print(circles.head(10))

# Visualize,visualize,visualize
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
plt.show()

# Note:The data we're working with is often referred to as a toy dataset,
# a dataset that is small enough to experiment but still sizeable enough to practice the fundamentals

# 1.1 Check the input and output shapes
print(X.shape, y.shape)

# View the first example of features and labels
X_sample = X[0]
y_sample = y[0]
print(f"Values for one sample of X:{X_sample} and the same for y:{y_sample}")
print(f"Shapes for one sample of X:{X_sample.shape} and the same for y:{y_sample.shape}")

# 1.2 Turn data into tensors and create train and test splits
# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
print(X[:5], y[:5])
print(type(X), X.dtype, y.dtype)

torch.manual_seed(42)

# Split data into training and data sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,  # 0.2 = 20% of data will be test & 80% will be train
                                                    random_state=42)
print(len(X_train), len(X_test), len(y_train), len(y_test))
print(n_samples)

# 2.Building a model
# To do so,we want to:
# 1.Setup device agnostic code so our code will run on an accelerator (GPU) if there is one
# 2.Construct a model (by subclassing nn.Module)
# 3.Define a loss function and optimizer
# 4.Creating a training and test loop

# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(X_train)
print(X_train.shape)


# 1.Construct a model that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # 2.Create 2 nn.Linear layers capable of handling the shapes of our data
        self.layer_1 = nn.Linear(in_features=2, out_features=5)  # takes in 2 features and upscales to 5 features
        self.layer_2 = nn.Linear(in_features=5,out_features=1)  # takes in 5 features from previous layer and
        # outputs a single layer(same shape as y)
        # 3.Define a forward() method that outlines the forward pass

    def forward(self, x):
        return self.layer_2(self.layer_1(x))  # x-> layer1 -> layer2 -> output


# 4. Instantiate an instance of our model class and send it to the target device
model_0 = CircleModelV0()
print(model_0)

print(device)
print(next(model_0.parameters()).device)

# nn.Sequential()
model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
)
print(model_0)

# Make predictions
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
print(f"Length of test samples :{len(X_test)}, Shape: {X_test.shape} ")
print(f"\nFirst 10 predictions:\n {torch.round(untrained_preds[:10])}")
print(f"\nFirst 10 test samples:\n {y_test[:10]}")

# 2.1 Setup loss function and optimizer
# For classification you might want binary cross entropy or categorical cross entropy(cross entropy).
# For the loss function we're going to use 'torch.nn.BCEWithLogitsLoss()'
# Setup the loss function
# loss_fn = nn.BCELoss() # BCELoss = requires inputs to have gone through the sigmoid
# activation function prior to input to BCELoss
loss_fn = torch.nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss = sigmoid activation function built-in
optimizer = torch.optim.SGD(model_0.parameters(),
                            lr=0.1)


# Calculate accuracy - out of 100 examples,what percentage does our model get right?
def accuracy_fn(y_true, y_prediction):
    correct = torch.eq(y_true, y_prediction).sum().item()
    accuracy = (correct / len(y_prediction)) * 100
    return accuracy


# 3.Train model
# 3.1.Going from raw logits->prediction probabilities->prediction labels
# Our model inputs are going to be raw **logits**
# We can convert these **logits** into **prediction probabilities** by passing
# them to some kind of activation function(e.g sigmoid for binary classification
# and softmax for multiclass classification).
print(model_0)
# Then we can convert our model's prediction probabilities to prediction labels by either
# rounding them or taking the argmax().
# View the first 5 outputs of the forward pass on the test data
y_logits = model_0(X_test)
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(X_test.to(device))[:5]
print(y_logits)
print(y[:5])

# Use the sigmoid activation function on our model logits to turn them into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)
print(torch.round(y_logits))

# For our prediction probability values,we need to perform a range-style rounding on them:
# y_pred_probs >= 0.5,y=1(class 1)
# y_pred_probs <= 0.5,y=0(class 0)

print(torch.round(y_pred_probs))

# Find the predicted labels
y_preds = torch.round(y_pred_probs)

# In full(logits->pred probs->pred labels)
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test).to(device))[:5])

# Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# Get rid of extra dimension
y_preds.squeeze()

print(y[:5])

# 3.2 Building a training and testing loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set the number of epochs
epochs = 100

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    # Training
    model_0.train()

    # 1. Forward pass
    y_logits = model_0(X_train).squeeze()
    y_predictions = torch.round(torch.sigmoid(y_logits))  # turn logits -> pred probs -> pred labels

    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # nn.BCELoss expects prediction probabilities as input
    #                y_train)
    loss = loss_fn(y_logits,  # nn.BCEWithLogitsLoss expects raw logits as input
                   y_train)
    acc = accuracy_fn(y_predictions, y_train.int())

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward (backpropagation)
    loss.backward()

    # 5. Optimizer step (gradient descent)
    optimizer.step()

    # Testing
    model_0.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze()
        test_prediction = torch.round(torch.sigmoid(test_logits))

        # 2. Calculate test loss/acc
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(test_prediction, y_test.int())

    # Print out what's happenin'
    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

# 4.Make predictions and evaluate the model
# Download helper functions from Learn PyTorch repo(if it's not already downloaded)
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists,skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)
# Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_0, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_0, X_test, y_test)
plt.show()

# 5.Improving a model(from a model perspective)
# * Add more layers - give the model more chances to learn about patterns in the data
# * Add more hidden units - go from 5 hidden units to 10 hidden units
# * Fit for longer
# * Changing the activation functions
# * Change the learning rate
# * Change the loss function
print(model_0.state_dict())


# These options are all from a model's perspective because they deal directly with the model,rather than the data
# And because these options are all values we can change,they are reffered as **hyperparameters**.
class CircleModelV1(nn.Module):
    def __init__(self):
        super(CircleModelV1, self).__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=5)
        self.layer_3 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))
        # this way of writing operations leverages speed ups where possible behind the scenes
        # z = self.layer_1(x)
        # z = self.layer_2(z)
        # z = self.layer_3(z)
        # return z


model_1 = CircleModelV1().to(device)
print(model_1)
print()
print(model_1.state_dict())

# Create a loss function
loss_fn = nn.BCEWithLogitsLoss()

# Create an optimizer
optimizer = torch.optim.SGD(model_1.parameters(),
                            lr=0.1)

# Write a training and evaluation loop for model_1
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Train for longer
epochs = 1000

# Put data on the target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    # Training
    model_1.train()
    # 1. Forward pass
    y_logits = model_1(X_train).squeeze()
    y_prediction = torch.round(torch.sigmoid(y_logits))  # logits -> pred probabilities -> prediction labels

    # 2.Calculate the loss/acc
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_prediction=y_prediction)

    # 3.Optimizer zero grad
    optimizer.zero_grad()

    # 4.Loss backward(backpropagation)
    loss.backward()

    # 5.Optimizer step
    optimizer.step()

    # Testing
    model_1.eval()
    with torch.inference_mode():
        # 1.Forward pass
        test_logits = model_1(X_test).squeeze()
        test_prediction = torch.round(torch.sigmoid(test_logits))
        # 2.Calculate the loss
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_prediction=test_prediction)
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
# Plot decision boundary of the model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_1, X_test, y_test)
plt.show()

# 5.1 Preparing data to see if our model can fit a straight line
# One way to troubleshoot to a larger problem is to test out a smaller problem
print(model_1.state_dict())

# Create some data
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.01

# Create data
X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
y_regression = weight * X_regression + bias  # Linear regression formula

# Check the data
print(len(X_regression))
print(X_regression[:5])
print(y_regression[:5])

# Create train and test splits
train_split = int(len(X_regression) * 0.8)
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]
print(len(X_train_regression), len(X_test_regression), len(y_train_regression), len(y_test_regression))


def plot_predictions(
        train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


plot_predictions(X_train_regression, y_train_regression,
                 X_test_regression, y_test_regression)
plt.show()
print(model_1)

print(X_train_regression[:10], y_train_regression[:10])

# 5.2 Adjusting 'model_1' to fit a straight line
# Same arhitecture as 'model_1'
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1),
).to(device)
print(model_2)

# Loss and optimizer
loss_fn = nn.L1Loss()  # MAE loss with regression data
optimizer = torch.optim.SGD(model_2.parameters(),
                            lr=0.01)
# Train the model
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set the number of epochs
epochs = 1000

# Put the data on the target device
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

# Training
for epoch in range(epochs):
    y_pred = model_2(X_train_regression)
    loss = loss_fn(y_pred, y_train_regression)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Testing
    model_2.eval()
    with torch.inference_mode():
        test_pred = model_2(X_test_regression)
        test_loss = loss_fn(test_pred, y_test_regression)

    # Print out what's happenin'
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")

        # Turn on evaluation mode
        model_2.eval()

        # Make predictions (inference)
        with torch.inference_mode():
            y_preds = model_2(X_test_regression)

        # Plot data and predictions
        plot_predictions(train_data=X_train_regression.cpu(),
                         train_labels=y_train_regression.cpu(),
                         test_data=X_test_regression.cpu(),
                         test_labels=y_test_regression.cpu(),
                         predictions=y_preds.cpu())
        plt.show()

# 6.The missing part piece:non-linearity
n_samples = 1000

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

# Turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
print(X[:5], y[:5])


# 6.2 Building a model with non-linearity
# Build a model with non-linear activation functions
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()  # relu is a non-linear activation function

    def forward(self, x):
        # Where should we put our non-linear activation functions?
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


model_3 = CircleModelV2()
print(model_3)
# Artificial neural networks are a large combination of linear and non-linear functions which are potentially able to find patterns in data

# Setup loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(),
                            lr=0.1)

# Random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Loop through data
epochs = 1000

for epoch in range(epochs):
    ### Training
    model_3.train()

    # 1. Forward pass
    test_logits = model_3(X_train).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))  # logits -> prediction probabilities -> prediction labels

    # 2. Calculate the loss
    loss = loss_fn(test_logits, y_train)  # BCEWithLogitsLoss (takes in logits as first input)
    acc = accuracy_fn(y_true=y_train,
                      y_prediction=test_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()

    ### Testing
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_prediction=test_pred)

    # Print out what's this happenin'
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
# print(model_3.state_dict())

# 6.4 Evaluating a model trained with non-linearity activation functions
# Makes predictions
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()
print(y_preds[:10], y_test[:10])
# In Matplotlib, a plot and a subplot refer to different concepts within the framework of
# creating visualizations. Here's a breakdown of each:
#
# Plot: This refers to the main graph or chart that is created to visualize data
# Subplot: This is a smaller plot that exists within a larger figure.
# Multiple subplots can be placed within a single figure to show different visualizations
# side by side or stacked in rows/columns.
# So subplot allows us to visualize more graphs or charts and comparing them

# Plot decision boundaries
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train)  # model_1 = no non-linearity
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)  # model_3 = has non-linearity
plt.show()

# 8.Putting it all together with a multi-class classification problem

# Toy datasets are small, simple datasets commonly used in the field of machine learning
# for training, testing, and demonstrating algorithms.

# 8.1 Creating a toy multi-class dataset
# Set the hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

# 1.Create multi-class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)
# give the clusters a little shake up

# 2.Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.long)

# 3. Split into train and test
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                        y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# 4.Plot the data
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()


# 8.2 Building a multi-class classification model
# Build a multi-class classification model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes multi-class classification model.

        Args:
          input_features (int): Number of input features to the model
          output_features (int): Number of outputs features (number of output classes)
          hidden_units (int): Number of hidden units between layers, default 8

        Returns:

        Example:
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


# Create an instance of BlobModel
model_4 = BlobModel(input_features=2,
                    output_features=4,
                    hidden_units=8).to(device)
print(model_4)
print(model_4.state_dict())

# 8.3. Create a loss function and an optimizer for a multi-class classification model
# Create a loss function for multi-class classification
loss_fn = nn.CrossEntropyLoss()

# Create an optimizer for multi-class classification
optimizer = torch.optim.SGD(params=model_4.parameters(),
                            lr=0.1)

# 8.4. Getting prediction probabilities for a multi-class model

# In order to evaluate and train and test our model, we need to convert our model's output (logits) to prediction probabilities and then to prediction labels
# Logits (raw outputs of the model) -> pred probs (use torch.softmax) -> pred labels (take the argmax of the prediction probabilities)
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test.to(device))
print(y_logits[:10])

print()

print(y_blob_test[:10])

print()

# Convert out model's logit outputs to prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_logits[:5])

print()

print(y_pred_probs[:5])

print()

print(torch.sum(y_pred_probs[0]))
print()

print(torch.max(y_pred_probs[0]))
print()

print(torch.argmax(y_pred_probs[0]))

# Convert our model's prediction probabilities to prediction labels
y_preds = torch.argmax(y_pred_probs, dim=1)
print(y_preds)
print(y_blob_test)

# 8.5 Creating a training loop and testing loop for a multi-class model
# Fit the multi-class model to the data
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
epochs = 100

# Put data to the target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

# Loop through data
for epoch in range(epochs):
    ### Training
    model_4.train()

    y_logits = model_4(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train,
                      y_prediction=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### Testing
    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_prediction=test_preds)

    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%")
# 8.6 Making and evaluating predictions with a PyTorch multi-class model
# Context managers are a way of allocating and releasing some sort of resource exactly where you need it.
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)
    # View first 10 predictions
    print(y_logits[:10])
# Logits are the raw, unnormalized predictions generated by a model before applying any activation function.
# They represent the linear output of a model before it is transformed into probabilities.

# Go from logits -> pred probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)
print(y_logits[:10])

# Go from pred probs -> pred labels
y_pred = torch.argmax(y_pred_probs, dim=1)
print(y_preds[:10])
print(y_blob_test)

print()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_blob_train, y_blob_train)
plt.show()

# 9. A few more classification metrics(to evaluate our classification model)
# Metric name                                                         Code                                 When to use
# Accuracy - out of 100 samples, how many does our model get right?   torchmetrics.Accuracy()              Default metric for classification
#                                                                           or                             problems.Not the best for
#                                                                     sklearn.metrics.accuracy_score()     imbalanced classes
#
# Precision                                                           torchmetrics.Precision()             Higher precision leads to less false positive
#                                                                          or
#                                                                     sklearn.metrics.accuracy_score()
#
# Recall                                                              torchmetrics.Recall()                Higher recalls leads to less false negatives
#                                                                          or
#                                                                     sklearn.metrics.recall_score()
# F1-score
# Confusion matrix                                                    torchmetrics.ConfusionMatrix()       When comparing predictions to truth
#                                                                                                          labels to see where model gets confused.
#
#                                                                                                      Can be hard to use with large number of classes
# Setup metric
torchmetric_accuracy = torchmetrics.Accuracy()
accuracy = torchmetric_accuracy(y_preds,y_blob_test)
print(accuracy)