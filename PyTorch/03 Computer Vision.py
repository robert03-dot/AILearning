# 0.Computer vision libraries:
# torchvision - base domain library for PyTorch computer vision
# torchvision.datasets - get datasets and data loading functions for computer vision
# torchvision.models - get pretrained computer vision models
# torchvision.transforms - functions for manipulating your vision data to be suitable for use with an ML model
# torch.utils.data.Dataset - Base dataset class for PyTorch
# torch.utils.data.DataLoader - Creates a Python iterable over a dataset
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets

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

