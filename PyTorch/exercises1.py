# Create a straight line dataset using the linear regression formula (weight * X + bias).
# Set weight=0.3 and bias=0.9 there should be at least 100 datapoints total.
# Split the data into 80% training, 20% testing.
# Plot the training and testing data so it becomes visual.
# Build a PyTorch model by subclassing nn.Module.
# Inside should be a randomly initialized nn.Parameter() with requires_grad=True,
# one for weights and one for bias.
# Implement the forward() method to compute the linear regression function you used to
# create the dataset in 1.
# Once you've constructed the model, make an instance of it and check its state_dict().
# Note: If you'd like to use nn.Linear() instead of nn.Parameter() you can.
# Create a loss function and optimizer using nn.L1Loss() and torch.optim.SGD(params, lr) respectively.
# Set the learning rate of the optimizer to be 0.01 and the parameters to optimize should be the
# model parameters from the model you created in 2.
# Write a training loop to perform the appropriate training steps for 300 epochs.
# The training loop should test the model on the test dataset every 20 epochs.
# Make predictions with the trained model on the test data.
# Visualize these predictions against the original training and testing data (note: you may need to
# make sure the predictions are not on the GPU if you want to use non-CUDA-enabled libraries such as
# matplotlib to plot).
# Save your trained model's state_dict() to file.
# Create a new instance of your model class you made in 2. and load in the state_dict() you just saved to it.
# Perform predictions on your test data with the loaded model and confirm they match the original
# model predictions from 4.
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch import nn

weight = 0.3
bias = 0.9
X = torch.arange(0, 1, 0.01).unsqueeze(dim=1)
y = weight * X + bias
print(f"Number of X samples: {len(X)}")
print(f"Number of y samples: {len(y)}")
print(f"First 10 X & y samples:\nX: {X[:10]}\ny: {y[:10]}")

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(train_data=X_train,
                     test_data=X_test,
                     train_labels=y_train,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="r", s=4, label="Test data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="g", s=4, label="Predictions")

    plt.legend(prop={"size": 15})


plot_predictions()
plt.show()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1,
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


torch.manual_seed(42)
model_0 = LinearRegressionModel()
print(model_0.state_dict())
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)
epochs = 300
for epoch in range(epochs):
    model_0.train()

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero gradients
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()

    if epoch % 20 == 0:
        model_0.eval()
        with torch.inference_mode():
            y_preds = model_0(X_test)
            test_loss = loss_fn(y_preds, y_test)
            print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)
print(y_preds)
plot_predictions(predictions=y_preds)
plt.show()
# Save your trained model's state_dict() to file.
# Create a new instance of your model class you made in 2. and load in the state_dict() you just saved to it.
# Perform predictions on your test data with the loaded model and confirm they match the original
# model predictions from 4.
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

loaded_model_0 = LinearRegressionModel()
print(loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH)))
loaded_model_0.eval()
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test)
print(loaded_model_preds)
print(y_preds == loaded_model_preds)

