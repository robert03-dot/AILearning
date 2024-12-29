import os
import pathlib
from pathlib import Path
import random
import numpy as np
import pandas as pd
import requests
import zipfile

import torchvision.io
from matplotlib import pyplot as plt

from PyTorch.helper_functions import plot_loss_curves

data_path = Path("/data")
image_path = data_path / "pizza_steak_sushi"
"""1. Get data
Our dataset is a subset of the Food101 dataset.

Food101 starts 101 different classes of food and 1000 images per class (750 training, 250 testing).

Our dataset starts with 3 classes of food and only 10% of the images (~75 training, 25 testing).

Why do this?

When starting out ML projects, it's important to try things on a small scale and then increase the scale when necessary.

The whole point is to speed up how fast you can experiment."""
if image_path.is_dir():
    print(f"{image_path} directory already exists...skipping download")
else:
    print(f"{image_path} directory does not exist, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
with open(data_path / "pizza_sushi_steak.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza,steak,sushi data...")
    f.write(request.content)
with zipfile.ZipFile(data_path / "pizza_sushi_steak.zip", "r") as zip_ref:
    print("Unzipping pizza,steak,sushi data...")
    zip_ref.extractall(image_path)

"""2. Becoming one with the data (data preparation and data exploration)"""


def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(filenames)} directories and {len(filenames)} images in '{dirpath}'.")


walk_through_dir(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"

"""2.1 Visualizing and image
Let's write some code to:

Get all of the image paths
Pick a random image path using Python's random.choice()
Get the image class name using pathlib.Path.parent.stem
Since we're working with images, let's open the image with Python's PIL
We'll then show the image and print metadata """

print(image_path)

image_path_list = list(image_path.glob("*/*/*.jpg"))
random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem

from PIL import Image

img = Image.open(random_image_path)

print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")

print(img)

img_as_array = np.asarray(img)
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels] (HWC)")
plt.axis(False)
plt.show()

print(img_as_array)

# 3.1 Transforming data with torchvision.transforms

from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == "__main__":
    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    print(data_transform(img).shape)


    def plot_transformed_image(image_paths: list, transform, n=3, seed=None):
        """
        Selects random images from a path of images and loads/transforms
        them then plots the original vs the transformed version.
        """
        if seed:
            random.seed(seed)
        random_image_paths = random.sample(image_paths, k=n)
        for image_path in random_image_paths:
            with Image.open(image_path) as f:
                fig, ax = plt.subplots(nrows=1, ncols=2)
                ax[0].imshow(f)
                ax[0].set_title(f"Original\nSize: {f.size}")
                ax[0].axis(False)

                transformed_image = transform(f).permute(1, 2,
                                                         0)  # note we will need to change shape for matplotlib (C, H, W) -> (H, W, C)
                ax[1].imshow(transformed_image)
                ax[1].set_title(f"Transformed\nSize: {transformed_image.shape}")
                ax[0].axis(False)

                fig.suptitle(f"Class:{image_path.parent.stem}", fontsize=16)


    plot_transformed_image(image_paths=image_path_list,
                           transform=data_transform,
                           n=3,
                           seed=None)
    plt.show()

    from torchvision import datasets

    # 4. Option 1: Loading image data using ImageFolder
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=data_transform,  # a transform for the data
                                      target_transform=None)  # a transform for the label/target
    test_data = datasets.ImageFolder(root=test_dir,
                                     target_transform=data_transform)
    print(train_data, test_data)
    print()
    print(train_dir, test_dir)
    print()
    class_names = train_data.classes
    print(class_names)
    class_dict = train_data.class_to_idx
    print(class_dict)
    print()
    print(len(train_data), len(test_data))
    print()
    print(train_data.samples[0])
    img, label = train_data[0]
    print(f"Image tensor:\n {img}")
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")

    img_permute = img.permute(1, 2, 0)

    # Print out different shapes
    print(f"Original shape: {img.shape} -> [color_channels, height, width]")
    print(f"Image permute: {img_permute.shape} -> [height, width, color_channels]")

    plt.figure(figsize=(10, 7))
    plt.imshow(img_permute)
    plt.axis(False)
    plt.title(class_names[label], fontsize=14)
    plt.show()

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=1,
                                  num_workers=1,
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False)
    print(train_dataloader, test_dataloader)
    print(len(train_dataloader), len(test_dataloader))
    img, label = next(iter(train_dataloader))
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")

    """5 Option 2:Loading Image Data with a Custom Dataset
    Want to be able to load images from file
    Want to be able to get class names from the Dataset
    Want to be able to get classes as dictionary from the Dataset
    
    Pros:
    Can create a Dataset out of almost anything
    Not limited to PyTorch pre-built Dataset functions
    
    Cons:
    Even though you could create Dataset out of almost anything, it doesn't mean it will work...
    Using a custom Dataset often results in us writing more code, which could be prone to errors or performance issues
    """
    print(train_data.classes, train_data.class_to_idx)

    """5.1 Creating a helper function to get class names
    We want a function to:
    
    Get the class names using os.scandir() to traverse a target directory (ideally the directory is in standard image classification format).
    Raise an error if the class names aren't found (if this happens, there might be something wrong with the directory structure).
    Turn the class names into a dict and a list and return them."""

    target_directory = train_dir
    print(f"Target directory: {target_directory}")

    class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])
    print(class_names_found)
    print(os.scandir(target_directory))

    from typing import Tuple, Dict, List


    def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folder names in a target directory."""
        # Debug: Check the directory content
        print(f"Scanning directory: {directory}")
        print(f"Contents: {[entry.name for entry in os.scandir(directory)]}")

        # Get the class names (only subdirectories)
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

        # Debug: Check the identified class names
        print(f"Found classes: {classes}")

        # Raise an error if class names could not be found
        if not classes:
            raise FileNotFoundError(f"No class folders found in {directory}. Check your file structure.")

        # Create a dictionary of index labels
        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
        print(f"Class to index mapping: {class_to_idx}")

        return classes, class_to_idx


    find_classes(str(target_directory))

    """5.2 Create a custom Dataset to replicate ImageFolder
    To create our own custom dataset, we want to:
    
    Subclass torch.utils.data.Dataset
    Init our subclass with a target directory (the directory we'd like to get data from) as well as a transform if we'd like to transform our data.
    Create several attributes:
    paths - paths of our images
    transform - the transform we'd like to use
    classes - a list of the target classes
    class_to_idx - a dict of the target classes mapped to integer labels
    Create a function to load_images(), this function will open an image
    Overwrite the __len()__ method to return the length of our dataset
    Overwrite the __getitem()__ method to return a given sample when passed an index"""
    from torch.utils.data import Dataset
    from PIL import Image
    import torch


    class ImageFolderCustom(Dataset):
        # 2. Initialize our custom dataset
        def __init__(self,
                     targ_dir: str,
                     transform=None):
            # 3. Create class attributes
            # Get all of the image paths
            self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
            self.transform = transform
            self.classes, self.class_to_idx = find_classes(targ_dir)

        def load_images(self, index: int) -> Image.Image:
            image_path = self.paths[index]
            return Image.open(image_path)

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
            # Load image
            img = self.load_images(index)
            # Get class index
            class_name = pathlib.Path(self.paths[index]).parent.name
            class_idx = self.class_to_idx[class_name]

            # Apply transform if available
            if self.transform:
                img = self.transform(img)
            else:
                raise ValueError("Transform must be provided to convert PIL Image to Tensor")

            return img, class_idx  # Return data (Tensor) and label (int)


    img, label = train_data[0]
    print(img, label)
# Create a transform
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

train_data_custom = ImageFolderCustom(targ_dir=train_dir,
                                      transform=train_transforms)

test_data_custom = ImageFolderCustom(targ_dir=test_dir,
                                     transform=test_transforms)
print(train_data_custom, test_data_custom)
print()
print(len(train_data), len(train_data_custom))
print()
print(len(test_data), len(test_data_custom))
print()
print(train_data_custom.classes)
print()
print(train_data_custom.class_to_idx)
print()
print(train_data_custom.classes == train_data.classes)
print(test_data_custom.classes == test_data.classes)

"""5.3 Create function to display random images
Take in a Dataset and a number of other parameters such as class names and how many images to visualize.
To prevent the display getting out of hand, let's cap the number of images to see at 10.
Set the random seed for reproducibility
Get a list of random sample indexes from the target dataset.
Setup a matplotlib plot.
Loop through the random sample indexes and plot them with matplotlib.
Make sure the dimensions of our images line up with matplotlib (HWC)"""


def display_random_images(dataset: torch.utils.data.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display,purposes,n shouldn't be larger than 10,setting to 10 and removing shape display.")
    if seed:
        random.seed(seed)
        random_samples_idx = random.sample(list(range(len(dataset))), k=n)
        plt.figure(figsize=(16, 8))
        for i, targ_sample in enumerate(random_samples_idx):
            targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
            targ_image_adjust = targ_image.permute(1, 2, 0)
            plt.subplot(1, n, i + 1)
            plt.imshow(targ_image_adjust)
            plt.axis('off')
            if classes:
                title = f"Class {classes[targ_label]}"
                if display_shape:
                    title = title + f"\nshape:{targ_label.shape}"
            plt.title(title)


display_random_images(train_data,
                      n=5,
                      classes=class_names,
                      seed=None)

# display_random_images(train_data_custom,
#                       n=5,
#                       classes=class_names,
#                       seed=None)
plt.show()

# 5.4 Turn custom loaded images into DataLoader's
train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                     batch_size=1,
                                     num_workers=os.cpu_count(),
                                     shuffle=True)

test_dataloader_custom = DataLoader(dataset=test_data_custom,
                                    batch_size=1,
                                    num_workers=os.cpu_count(),
                                    shuffle=False)

print(train_dataloader_custom, test_dataloader_custom)

# Get image and label from custom dataloader
img_custom, label_custom = next(iter(train_dataloader_custom))
print(img_custom, label_custom)

# 6.Oter forms of transforms(data augumentation)
"""Data augmentation is the process of artificially adding diversity to your training data
   In this case of image data,this may mean applying various image transformations to the training images.
   This practice hopefully results in a model that's more generalizable to unseen data."""

train_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])
train_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

print(image_path)

print()

image_path_list = list(image_path.glob("*/*/*.jpg"))
print(image_path_list[:10])

plot_transformed_image(
    image_paths=image_path_list,
    transform=train_transforms,
    n=3,
    seed=None
)

# 7.1 Creating transforms and loading data for Model 0
# Create simple transform
simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])
# 1.Load and transform data
train_data_simple = datasets.ImageFolder(root=train_dir,
                                         transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir,
                                        transform=simple_transform)
# 2.Turn the datasets into DataLoaders
# Setup batch size and number of works
train_dataloader_simple = DataLoader(dataset=train_data_simple,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=os.cpu_count())
test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                    batch_size=32,
                                    shuffle=False,
                                    num_workers=os.cpu_count())
from torch import nn


# 7.2 Create TinyVGG model class
class TinyVGG(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super(TinyVGG, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 13 * 13,
                      out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(class_names))
print(model_0)

# 7.3 Test the model
image_batch, label_batch = next(iter(train_dataloader_simple))
print(image_batch.shape, label_batch.shape)
print(model_0(image_batch))

# Getting an idea of the shapes going through our model
try:
    import torchinfo
except:
    'pip install torchinfo'
    import torchinfo
from torchinfo import summary

print(summary(model_0, input_size=(1, 3, 64, 64)))

"""train_step() - takes in a model and dataloader and trains the model on the dataloader.
   test_step() - takes in a model and dataloader and evaluates the model on the dataloader."""


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc


from tqdm.auto import tqdm


# 1. Create a train function that takes in various model parameters + optimizer + dataloaders + loss function
def train(model: torch.nn.Module,
          train_dataloader,
          test_dataloader,
          optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    # 3.Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)
        print(
            f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results


torch.manual_seed(42)
model_1 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data.classes))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(),
                             lr=0.01)
from timeit import default_timer as timer

start_time = timer()
model_0_results = train(model=model_1,
                        train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=5)
end_time = timer()
print(f"Total training time: {end_time - start_time:.3f} seconds")
plot_loss_curves(model_0_results)
plt.show()

# Compare model results
# After evaluating our modelling experiments on their own,it's important to compare them to each other.

print(model_0_results.keys())


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary."""
    # Get the loss values of the results dictionary(training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["test_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_acc")
    plt.plot(epochs, test_accuracy, label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


plot_loss_curves(model_0_results)
plt.show()

train_transform_trivial = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform_simple = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

# 9.2 Create train and test Dataset's and DataLoader's with data augmentation
# Turn images folders into datasets
train_data_augmented = datasets.ImageFolder(root=train_dir,
                                            transform=train_transform_trivial)
test_data_augmented = datasets.ImageFolder(root=test_dir,
                                           transform=test_transform_simple)
torch.manual_seed(42)
train_dataloader_augmented = DataLoader(dataset=train_data_augmented,
                                        batch_size=32,
                                        shuffle=True,
                                        num_workers=os.cpu_count())

test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                    batch_size=32,
                                    shuffle=False,
                                    num_workers=os.cpu_count())

model_1 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data_augmented.classes))
print(model_1)

torch.manual_seed(42)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(),
                             lr=0.001)
start_time = timer()
model_1_results = train(model=model_1,
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=5)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time for model_1: {end_time - start_time:.3f} seconds")
plot_loss_curves(model_1_results)

model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)
print(model_0_df)

plt.figure(figsize=(15, 10))
epochs = range(len(model_0_df))

# Plot train loss
plt.subplot(2, 2, 1)
plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.legend()

# Plot test loss
plt.subplot(2, 2, 2)
plt.plot(epochs, model_0_df["test_loss"], label="Model 0")
plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.legend()

# Plot train accuracy
plt.subplot(2, 2, 3)
plt.plot(epochs, model_0_df["train_acc"], label="Model 0")
plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
plt.title("Train Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()
# Plot test accuracy
plt.subplot(2, 2, 4)
plt.plot(epochs, model_0_df["test_acc"], label="Model 0")
plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
plt.title("Test Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()

"""11. Making a prediction on a custom image
Although we've trained a model on custom data... how do you make a prediction 
on a sample/image that's not in either training or testing dataset."""
custom_image_path = data_path / "04-pizza-dad.jpeg"
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        request = requests.get(
            "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists,skipping download...")

"""11.1 Loading in a custom image with PyTorch
We have to make sure our custom image is in the same format as the data our model was trained on.

In tensor form with datatype (torch.float32)
Of shape 64x64x3
On the right device"""
print(custom_image_path)

custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))
print(f"Custom image tensor:\n {custom_image_uint8}")
print(f"Custom image shape: {custom_image_uint8.shape}")
print(f"Custom image datatype: {custom_image_uint8.dtype}")

plt.imshow(custom_image_uint8.permute(1, 2, 0))


custom_image_transform = transforms.Compose([
                                             transforms.Resize(size=(64, 64))
])
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32) / 255.
# Transform target image
custom_image_transformed = custom_image_transform(custom_image)

# Print out the shapes
print(f"Original shape: {custom_image.shape}")
print(f"Transformed shape: {custom_image_transformed.shape}")
plt.imshow(custom_image_transformed.permute(1, 2, 0))

# This should this work? (added a batch size...)
model_1.eval()
with torch.inference_mode():
  custom_image_pred = model_1(custom_image_transformed.unsqueeze(0))
print(custom_image_pred)

"""Note, to make a prediction on a custom image we had to:

Load the image and turn it into a tensor
Make sure the image was the same datatype as the model (torch.float32)
Make sure the image was the same shape as the data the model was trained on (3, 64, 64) with a batch size... (1, 3, 64, 64)
Make sure the image was on the same device as our model"""
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
print(custom_image_pred_probs)

custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
print(custom_image_pred_label)

print(class_names[custom_image_pred_label])

"""11.3 Putting custom image prediction together: building a function
Ideal outcome:

A function where we pass an image path to and have our model predict on that image and plot the image + prediction."""

def pred_and_plot_image(model:torch.nn.Module,
                        image_path:str,
                        class_names:list[str] = None,
                        transform = None):
    """Makes a prediction on a target image with a trained model and plots the image and prediction."""
    # Load in the image
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    # Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.
    if transform:
        target_image = transform(target_image)
    model.eval()
    with torch.inference_mode():
        target_image = torch.unsqueeze(target_image, 0)
        target_image_pred = model(target_image)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
        plt.imshow(target_image.permute(1, 2, 0))
        if class_names:
            title = f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
        else:
            title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
        plt.title(title)
        plt.axis(False)
        plt.show()
pred_and_plot_image(model=model_1,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform)
plt.show()