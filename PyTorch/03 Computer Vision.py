# 0.Computer vision libraries:
# torchvision - base domain library for PyTorch computer vision
# torchvision.datasets - get datasets and data loading functions for computer vision
# torchvision.models - get pretrained computer vision models
# torchvision.transforms - functions for manipulating your vision data to be suitable for use with an ML model
# torch.utils.data.Dataset - Base dataset class for PyTorch
# torch.utils.data.DataLoader - Creates a Python iterable over a dataset
from pathlib import Path

import matplotlib.pyplot as plt
import requests
import torch
from timeit import default_timer as timer
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm.auto import tqdm

from PyTorch.helper_functions import accuracy_fn

# 1.Getting a dataset
# The dataset we'll be using is FashionMNIST

# The MNIST database (Modified National Institute of Standards and Technology database[1]) is a large database of
# handwritten digits that is commonly used for training various image processing systems.

# Setup training data
train_data = datasets.FashionMNIST(
    root="data",  # where to download data to?
    train=True,  # do we want the training data?
    download=True,  # do we want to download?
    transform=torchvision.transforms.ToTensor(),  # how do we want to transform the data?
    target_transform=None  # how do we want to transform the labels/targets?
)
test_data = datasets.FashionMNIST(
    root="data",  # where to download data to?
    train=False,  # do we want the testing data?
    download=True,  # do we want to download?
    transform=torchvision.transforms.ToTensor(),  # how do we want to transform the data?
    target_transform=None  # how do we want to transform the labels/targets?
)

print(len(train_data), len(test_data))

# See the first training example
image, label = train_data[0]
print(image, label)

class_names = train_data.classes
print(class_names)

class_to_idx = train_data.class_to_idx
print(class_to_idx)

print(train_data.targets)

# Check the shape of our image
print(f"Image shape:{image.shape}->[color_channels,width,height]")
print(f"Image shape:{class_names[label]}")

# 1.2 Visualizing our data
image, label = train_data[0]
print(f"Image shape:{image.shape}")
# plt.imshow(image)
print(image)
plt.imshow(image.squeeze())
plt.title(label)
plt.show()

plt.imshow(image.squeeze(), cmap='gray')
plt.title(class_names[label])
plt.axis(False)
plt.show()

# Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    # print(i)
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    print(random_idx)
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(class_names[label])
    plt.axis(False)
    plt.show()

print(train_data)
print(test_data)

# 2.Prepare DataLoader
# DatLoader turns our data into a Python iterable
# More specifically,we want to turn our data into batches(or mini-batches)
##
# Why would we do this?
# 1.It is more computationally efficient,as in,your computing hardware may not be able to look(store in memory) at 60.000 images in one hit.
# So we break it down to 32 images at a time(batch size of 32).
# 2.It gives our neural network more chances to update its gradients per epoch.

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables(batches)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)
print(train_dataloader, test_dataloader)

print(f"DataLoader:{train_dataloader, test_dataloader}")
print(f"Length of train_dataloader:{len(train_dataloader)} batches of:{BATCH_SIZE}...")
print(f"Length of test_dataloader:{len(test_dataloader)} batches of:{BATCH_SIZE}...")

# Check out what's insider the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape)
print(train_labels_batch.shape)

# Show a sample
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap='gray')
plt.title(class_names[label])
plt.axis(False)
plt.show()
print(f"Image size:{img.shape}")
print(f"Label:{label},label size:{label.shape}")

# 3.Model 0:Build a baseline model
# Create a flatten layer
flatten_model = nn.Flatten()

# Get a single sample
x = train_features_batch[0]
print(x.shape)

# Flatten the sample
output = flatten_model(x)  # perform forward pass

# Print out what happened
print(f"Shape before flattening:{x.shape}")
print(f"Shape after flattening:{output.shape}")


class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=output_shape,
                      out_features=output_shape),
        )

    def forward(self, x):
        return self.layer_stack(x)


model_0 = FashionMNISTModelV0(
    input_shape=784,  # this is 28*28
    hidden_units=10,  # how many units in the hidden layer
    output_shape=len(class_names)  # one for every class
)
print(model_0)

dummy_x = torch.rand([1, 1, 28, 28])
print(model_0(dummy_x))
print()
print(model_0.state_dict())

# 3.1 Setup loss,optimizer and evaluation metrics
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists,skipping downloading...")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_0.parameters(),
                            lr=0.1)


# 3.2 Creating a function to time our experiments
# Two of the main things you'll often want to track are:
# 1.Model's performance(loss and accuracy values etc.)
# 2.How fast it runs
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    """Prints difference between start and end time"""
    total_time = end - start
    print(f"Train time on:{device}:{total_time:.3f} seconds")
    return total_time


start_time = timer()
end_time = timer()
print(print_train_time(start=start_time, end=end_time, device=torch.device("cpu")))

# 3.3 Creating a training loop and training a model on batches on data
# 1.Loop through epochs
# 2.Loop through training batches,perform training steps,calculate the train loss *per batch*
# 3.Loop through testing batches,perform testing steps,calculate the test loss *per batch*
# 4.Print out what's happening
# 5.Time it all
# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Set the number of epochs (we'll keep this small for faster training times)
epochs = 3

# Create training and testing loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n-------")
    ### Training
    train_loss = 0
    # Add a loop to loop through training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        # 1. Forward pass
        y_pred = model_0(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss  # accumulatively add up the loss per epoch

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Print out how many samples have been seen
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples")

    # Divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)

    ### Testing
    # Setup variables for accumulatively adding up loss and accuracy
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            # 1. Forward pass
            test_pred = model_0(X)

            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y)  # accumulatively add up the loss per epoch

            # 3. Calculate accuracy (preds need to be same as y_true)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)

    ## Print out what's happening
    print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

# Calculate training time
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                            end=train_time_end_on_cpu,
                                            device=str(next(model_0.parameters()).device))

# 4.Make predictions and get model_0 results
torch.manual_seed(42)


def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader."""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            # Make predictions with the model
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(
                                   dim=1))  # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


# Calculate model 0 results on test dataset
model_0_results = eval_model(model=model_0,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn)
print(model_0_results)


class FashionMNISTV1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
        )
    def forward(self, x:torch.Tensor):
        return self.layer_stack(x)
torch.manual_seed(42)
model_1 = FashionMNISTV1(input_shape=784,
                         hidden_units=10,
                         output_shape=len(class_names))