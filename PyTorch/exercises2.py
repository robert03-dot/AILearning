# 1.Make a binary classification dataset with Scikit-Learn's make_moons() function.
# For consistency, the dataset should have 1000 samples and a random_state=42.
# Turn the data into PyTorch tensors. Split the data into training and test sets using train_test_split with 80%
# training and 20% testing.
# 2.Build a model by subclassing nn.Module that incorporates non-linear activation functions and is capable of
# fitting the data you created in 1.
# Feel free to use any combination of PyTorch layers (linear and non-linear) you want.
# 3.Setup a binary classification compatible loss function and optimizer to use when training the model.
# 4.Create a training and testing loop to fit the model you created in 2 to the data you created in 1.
# To measure model accuracy, you can create your own accuracy function or use the accuracy function in TorchMetrics.
# Train the model for long enough for it to reach over 96% accuracy.
# The training loop should output progress every 10 epochs of the model's training and test set loss and accuracy.
# 5.Make predictions with your trained model and plot them using the plot_decision_boundary() function created
# in this notebook.
# 6.Replicate the Tanh (hyperbolic tangent) activation function in pure PyTorch.
# Feel free to reference the ML cheatsheet website for the formula.
# 7.Create a multi-class dataset using the spirals data creation function from CS231n (see below for the code).
# Construct a model capable of fitting the data (you may need a combination of linear and non-linear layers).
# Build a loss function and optimizer capable of handling multi-class data (optional extension: use the Adam optimizer
# instead of SGD, you may have to experiment with different values of the learning rate to get it working).
# Make a training and testing loop for the multi-class data and train a model on it to reach over 95% testing accuracy
# (you can use any accuracy measuring function here that you like).
# Plot the decision boundaries on the spirals dataset from your model predictions, the plot_decision_boundary()
# function should work for this dataset too.
import torch
from matplotlib import pyplot as plt
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy

X, y = make_moons(n_samples=1000,
                  noise=0.2,
                  random_state=42)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)


class MoonModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_units=8):
        super(MoonModel, self).__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


model = MoonModel(in_features=2, out_features=1, hidden_units=8)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(),
                         lr=0.1)

accuracy_fn = Accuracy()
torch.manual_seed(42)
epochs = 1000
for epoch in range(epochs):
    model.train()
    y_logits = model(X_train).squeeze()
    y_predictions = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    accuracy = accuracy_fn(y_predictions, y_train.int())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_accuracy = accuracy_fn(test_pred, y_test.int())
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.2f} Acc: {accuracy:.2f} | Test loss: {test_loss:.2f} Test acc: {test_accuracy:.2f}")

# The tensor y_true is the true data (or target, ground truth) you pass to the fit method.
# It's a conversion of the numpy array y_train into a tensor.
#
# The tensor y_pred is the data predicted (calculated, output) by your model.
# Plot decision boundaries for training and test sets




# TK - this could go in the helper_functions.py and be explained there
def plot_decision_boundary(model, X, y):
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Source - https://madewithml.com/courses/foundations/neural-networks/
    # (with modifications)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()



# Code for creating a spiral dataset from CS231n
import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.show()

tensor_A = torch.arange(-100,100,1)
plt.plot(torch.tanh(tensor_A))
plt.show()

class SpiralModel(nn.Module):
  def __init__(self,in_features, out_features, hidden_units=8):
      super(SpiralModel, self).__init__()
      self.linear_layer_stack = nn.Sequential(
          nn.Linear(in_features=in_features, out_features=hidden_units),
          nn.ReLU(),
          nn.Linear(in_features=hidden_units, out_features=hidden_units),
          nn.ReLU(),
          nn.Linear(in_features=hidden_units, out_features=hidden_units),
          nn.ReLU(),
          nn.Linear(in_features=hidden_units, out_features=hidden_units),
          nn.ReLU(),
          nn.Linear(in_features=hidden_units, out_features=out_features)
      )

  def forward(self, x):
    return self.linear_layer_stack(x)


model_2 = SpiralModel(in_features=2, out_features=1, hidden_units=8)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_2.parameters(),
                             lr=0.02)

# Build a training loop for the model
epochs = 1000

# Loop over data
for epoch in range(epochs):
    model.train()
    y_logits = model(X_train).squeeze()
    y_predictions = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    accuracy = accuracy_fn(y_predictions, y_train.int())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_accuracy = accuracy_fn(test_pred, y_test.int())

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.2f} Acc: {accuracy:.2f} | Test loss: {test_loss:.2f} Test acc: {test_accuracy:.2f}")
# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_2, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_2, X_test, y_test)
plt.show()