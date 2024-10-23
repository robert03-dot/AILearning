# 0.Computer vision libraries:
# torchvision - base domain library for PyTorch computer vision
# torchvision.datasets - get datasets and data loading functions for computer vision
# torchvision.models - get pretrained computer vision models
# torchvision.transforms - functions for manipulating your vision data to be suitable for use with an ML model
# torch.utils.data.Dataset - Base dataset class for PyTorch
# torch.utils.data.DataLoader - Creates a Python iterable over a dataset
# Create a model with non-linear and linear layers
from pathlib import Path
import requests
import torchvision
from datasets import tqdm
from matplotlib import pyplot as plt
from torch import nn, torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from helper_functions import accuracy_fn
from torchvision import datasets
from timeit import default_timer as timer

# Setup training data

train_data = datasets.FashionMNIST(
    root="data",  # where to download data to?
    train=True,  # do we want the training dataset?
    download=True,  # do we want to download yes/no?
    transform=torchvision.transforms.ToTensor(),  # how do we want to transform the data?
    target_transform=None  # how do we want to transform the labels/targets?
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
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
print(f"Image shape: {image.shape} -> [color_channels, height, width]")
print(f"Image label: {class_names[label]}")

image, label = train_data[0]
print(f"Image shape: {image.shape}")
plt.imshow(image.squeeze())
plt.title(label)
plt.show()

plt.imshow(image.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
plt.show()
# Plot more images
# torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(class_names[label])
    plt.axis(False)
    plt.show()
print(train_data, test_data)

# Setup the batch size hyperparameter
BATCH_SIZE = 32

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

print(train_dataloader, test_dataloader)
# Let's check out what what we've created
print(f"DataLoaders: {train_dataloader, test_dataloader}")
print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}...")
print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}...")
# Check out what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)
# Show a sample
# torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
plt.show()
print(f"Image size: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")
# Create a flatten layer
flatten_model = nn.Flatten()

# Get a single sample
x = train_features_batch[0]

# Flatten the sample
output = flatten_model(x)  # perform forward pass

# Print out what happened
print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")


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
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


torch.manual_seed(42)

# Setup model with input parameters
model_0 = FashionMNISTModelV0(
    input_shape=28 * 28,  # this is 28*28
    hidden_units=10,  # how mnay units in the hidden layer
    output_shape=len(class_names)  # one for every class
).to("cpu")

print(model_0)
dummy_x = torch.rand([1, 1, 28, 28])
print(model_0(dummy_x))
print(model_0.state_dict())

# Download helper functions from Learn PyTorch repo
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download...")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)


def print_train_time(start: float,
                     end: float):
    """Prints difference between start and end time."""
    total_time = end - start
    return total_time


start_time = timer()
end_time = timer()
print_train_time(start=start_time,
                 end=end_time)

# Set the seed and start the timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Set the number of epochs (we'll keep this small for faster training time)
epochs = 3

# Create training and test loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n------")
    ### Training
    train_loss = 0
    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        # 1. Forward pass
        y_pred = model_0(X)

        # 2. Calculate loss (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss  # accumulate train loss

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step (update the model's parameters once *per batch*)
        optimizer.step()

        # Print out what's happening
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")

    # Divide total train loss by length of train dataloader
    train_loss /= len(train_dataloader)

    ### Testing
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            # 1. Forward pass
            test_pred = model_0(X_test)

            # 2. Calculate loss (accumulatively)
            test_loss += loss_fn(test_pred, y_test)

            # 3. Calculate accuracy
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))

        # Calculate the test loss average per batch
        test_loss /= len(test_dataloader)

        # Calculate the test acc average per batch
        test_acc /= len(test_dataloader)

    # Print out what's happening
    print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

# Calculate training time
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                            end=train_time_end_on_cpu)
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
            # Make predictions
            y_pred = model(X)

            # Accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))

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
print(torch.cuda.is_available())


class FashionMNISTModelV1(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),  # flatten inputs into a single vector
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)


# Create an instance of model_1
torch.manual_seed(42)
model_1 = FashionMNISTModelV1(input_shape=784,  # this is the output of the flatten after our 28*28 image goes in
                              hidden_units=10,
                              output_shape=len(class_names))  # send to the GPU if it's available
print(next(model_1.parameters()))

from helper_functions import accuracy_fn

loss_fn = nn.CrossEntropyLoss()  # measure how wrong our model is
optimizer = torch.optim.SGD(params=model_1.parameters(),  # tries to update our model's parameters to reduce the loss
                            lr=0.1)


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn):
    """Performs a training with model trying to learn on data_loader."""
    train_loss, train_acc = 0, 0

    # Put model into training mode
    model.train()

    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(data_loader):
        # 1. Forward pass (outputs the raw logits from the model)
        y_pred = model(X)

        # 2. Calculate loss and accuracy (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss  # accumulate train loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))  # go from logits -> prediction labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step (update the model's parameters once *per batch*)
        optimizer.step()

    # Divide total train loss and acc by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn):
    """Performs a testing loop step on model going over data_loader."""
    test_loss, test_acc = 0, 0

    # Put the model in eval mode
    model.eval()

    # Turn on inference mode context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # 1. Forward pass (outputs raw logits)
            test_pred = model(X)

            # 2. Calculuate the loss/acc
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1))  # go from logits -> prediction labels

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")


torch.manual_seed(42)

train_time_start_on_gpu = timer()

# Set epochs
epochs = 3

# Create a optimization and evaluation loop using train_step() and test_step()
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n----------")
    train_step(model=model_1,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn)
    test_step(model=model_1,
              data_loader=test_dataloader,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn)

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu)
# Train time on CPU
print(total_train_time_model_0)
# Get model_1 results dictionary
model_1_results = eval_model(model=model_1,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn)
print(model_1_results)
print()
print(model_0_results)


# Convolutional Neural Network(CNN)
# CNN's are also known as ConvNets
# CNN's are known for their capabilities to find patterns in visual data

# Convolutional layer
# The convolutional layer is the core building block of a CNN, and
# it is where the majority of computation occurs.
# It requires a few components, which are input data, a filter, and a feature map.
class FashionMNISTV2(nn.Module):
    # A
    # Conv - block
    # stacks
    # a
    # convolutional, batch
    # normalisation, activation, pooling, and a
    # dropout
    # layer in sequence.
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, )
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, )
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 0,
                      out_features=output_shape),
        )

        def forward(x):
            x = self.conv_block_1(x)
            print(x.shape)
            x = self.conv_block_2(x)
            print(x.shape)
            x = self.classifier(x)
            return x


print(image.shape)
torch.manual_seed(42)
model_2 = FashionMNISTV2(input_shape=1,
                         hidden_units=10,
                         output_shape=len(class_names))
print(model_2.state_dict())
# In the Tiny VGG architecture, convolutional layers are fully-connected,
# meaning each neuron is connected to every other neuron in the previous layer.

# ---Padding is often necessary when the kernel extends beyond the activation map.
# Padding conserves data at the borders of activation maps, which leads to better performance,
# and it can help preserve the input's spatial size, which allows an architecture designer
# to build deeper, higher performing networks. '
# 'There exist many padding techniques, but the most commonly used approach is zero-padding'
# ' because of its performance, simplicity, and computational efficiency. '
# 'The technique involves adding zeros symmetrically around the edges of an input.'
# 'This approach is adopted by many high-performing CNNs such as AlexNet.)
# ---Kernel size, often also referred to as filter size, refers to the dimensions
# of the sliding window over the input. Choosing this hyperparameter has a massive impact
# on the image classification task. For example, small kernel sizes are able to extract a
# much larger amount of information containing highly local features from the input.
# As you can see on the visualization above, a smaller kernel size also leads to a smaller
# reduction in layer dimensions, which allows for a deeper architecture.
# Conversely, a large kernel size extracts less information, which leads to a faster reduction
# in layer dimensions, often leading to worse performance. Large kernels are better suited to
# extract features that are larger. At the end of the day, choosing an appropriate kernel
# size will be dependent on your task and dataset, but generally, smaller kernel sizes lead
# to better performance for the image classification task because an architecture designer
# is able to stack more and more layers together to learn more and more complex features!
# ---Stride indicates how many pixels the kernel should be shifted over at a time.
# For example, as described in the convolutional layer example above, Tiny VGG uses a stride
# of 1 for its convolutional layers, which means that the dot product is performed on a
# 3x3 window of the input to yield an output value, then is shifted to the right by one pixel
# for every subsequent operation. The impact stride has on a CNN is similar to kernel size.
# As stride is decreased, more features are learned because more data is extracted,
# which also leads to larger output layers. On the contrary, as stride is increased,
# this leads to more limited feature extraction and smaller output layer dimensions.
# One responsibility of the architecture designer is to ensure that the kernel slides
# across the input symmetrically when implementing a CNN. Use the hyperparameter
# visualization above to alter stride on various input/kernel dimensions to understand
# this constraint!

# 7.1 Stepping through 'nn.Conv2d()'
torch.manual_seed(42)

# Create a batch of images
images = torch.randn(size=(32, 3, 64, 64))
test_image = images[0]

print(f"Image batch shape:{images.shape}")
print()
print(f"Single image batch shape:{test_image.shape}")
print()
print(f"Test image:\n{test_image}")
