## 00. PyTorch Fundamentals
import numpy as np
import torch

print(torch.__version__)

## Introductions to Tensors
### Creating tensors

# scalar
scalar = torch.tensor(7)
print(scalar)
print()
# PyTorch tensors are created using 'torch.Tensor()'
print(scalar.ndim)
# Get tensor back as Python int
print()
# Vector
vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)
print(vector.shape)
print()
# MATRIX
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX[0])
print(MATRIX[1])
print(MATRIX.shape)
print()
# TENSOR
TENSORS = torch.tensor([[[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]]])
print(TENSORS)
print(TENSORS.ndim)
print(TENSORS.shape)
print()
### Random tensors
# Why random tensors?
# Random tensors are important because the way many neural netowrks learn is that they start tensors with full numbers
# and then adjust those random numbers to better represent the data
# Start with random numbers -> look at data -> update random numbers -> look at data -> update random numbers

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim)
print()
# Create a random tensor with similarm shape to an image tensor
random_image_size_tensor = torch.rand(size=(224, 224, 3))  # height, width, colour channels (R, G, B)
print(random_image_size_tensor.ndim)
print(random_image_size_tensor.shape)
print()
# Zeros and ones
# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print(zeros)
print()
# Create a tensor of all ones
ones = torch.ones(size=(3, 4))
print(ones)
print(ones.dtype)
print(random_tensor.dtype)
print()
### Create a range of tensors and tensors-like
# Use torch.range()
one_to_ten = torch.arange(start=0, end=1000, step=77)
print(one_to_ten)
print()
# Creating tensors-like
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)
print()
### Tensor datatypes
# **Note:** Tensor datatypes is one of the 3 big errors you'll run into with PyTorch & deep learning:
# 1.Tensors not right datatypes
# 2.Tensors not right shape
# 3.Tensors not on the right device
# Float 32 tensor
float_32_tensor = torch.tensor([1.0, 2.0, 3.0],
                               dtype=None,  # What datatype is the tensor(e.g float32 or float16)
                               device=None,  # What device is your tensor on
                               requires_grad=False)  # Whether or not to track gradients with this tensors operations
print(float_32_tensor)
print(float_32_tensor.dtype)
print()
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)
print()
int_32_tensor = torch.tensor([3, 6, 9], dtype=torch.int32)
print(int_32_tensor)
print(int_32_tensor * float_32_tensor)
print()
### Getting information from tensors (tensor attributes)
# 1.Tensors not right datatypes - to get datatype from a tensor,can use tensor.dtype
# 2.Tensors not right shape - to get shape from a tensor,can use tensor.shape
# 3.Tensors not on the right device - to get device from a tensor,can use tensor.device

# 1.Create a tensor
some_tensor = torch.rand(3, 4)
print(some_tensor)

# Find out some details about tensor
print(some_tensor)
print(f"Datatype of tensor:{some_tensor.dtype}")
print(f"Shape of tensor:{some_tensor.shape}")
print(f"Device tensor is on:{some_tensor.device}")
print()
### Manipulating tensors(tensor operations)

## Tensor operations include:
# Addition
# Subtraction
# Multiplication(element-wise)
# Division
# Matrix multiplication

# Create a tensor and add 10 to it
tensor1 = torch.tensor([1, 2, 3])
tensor1 = tensor1 + 10
print(tensor1)
print()
# Multiply tensor by 10
tensor2 = torch.tensor([1, 2, 3])
tensor2 = tensor2 * 10
print(tensor2)
print()
# Substract 10
tensor3 = torch.tensor([1, 2, 3])
print(tensor3 - 10)
# Try-out PyTorch in-built functions
tensor4 = torch.tensor([1, 2, 3])
print(torch.mul(tensor4, 10))
print()
tensor5 = torch.tensor([1, 2, 3])
print(torch.add(tensor5, 10))
print()
tensor6 = torch.tensor([1, 2, 3])
print(torch.sub(tensor6, 10))

### Matrix multiplication

# Two main ways of performing multiplication in neural networks and deep learning:
# 1.Element-wise multiplication
# 2.Matrix multiplication(dot product)

# Element-wise multiplication
print(tensor1, "*", tensor1)
print(f"Equals:{tensor1 * tensor1}")
# Matrix multiplication
print(torch.matmul(tensor1, tensor1))
print()
value = 0
for i in range(len(tensor2)):
    value += tensor2[i] * tensor2[i]
print(value)
print()
# There are two rules that performing matrix multiplication needs to satisfy:
# 1.The **inners dimensions** must match:
# (3,2) @ (3,2) won't work
# (2,3) @ (3,2) will work
# (3,2) @ (2,3) will work

# print(torch.matmul(torch.rand(3, 2), torch.rand(3, 2)))
print(torch.matmul(torch.rand(3, 10), torch.rand(10, 3)).shape)
print()
# 2.The resulting matrix has the shape of the **outer dimensions**:
# (2,3) @ (3,2) -> (2,2)
# (3,2) @ (2,3) -> (3,3)


# One of the must common errors in deep learning:shape errors
# Shapes for matrix multiplication
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])
tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])

print()

# print(torch.mm(tensor_A, tensor_B))  # torch.mm is the same as torch.matmul(it's an alias for writing less code)

print(tensor_A.shape)
print(tensor_B.shape)
print()
# To fix our tensor shape issues,we can manipulate the shape of one of our tensors using a **transpose**.
# A **transpose** switches the axes or dimensions of a given tensor
print(tensor_A.shape)
print(tensor_B.shape)

print()

print(tensor_A.T)
print(tensor_B.T)

print()

print(tensor_A.T.shape)
print(tensor_B.T.shape)

print()

print(torch.matmul(tensor_A, tensor_B.T).shape)
print()
# The matrix multiplication operation works when tensor_B is transposed
print(f"Original shape of tensor_A:{tensor_A.shape}, tensor_B:{tensor_B.shape}")
print(f"The transposed shape of tensor_A:{tensor_A.T.shape}, tensor_B:{tensor_B.T.shape}")
print(f"Multiplying:{tensor_A.shape} @ {tensor_B.T.shape} <- inner dimensions must match")
print("Output:\n")
print()
output = torch.matmul(tensor_A, tensor_B.T)
print(output)
print(f"\nOutput shape:{output.shape}")
print()
### Finding the min,max,mean,sum,etc.(tensor aggregation)
# Create a tensor
x = torch.arange(1, 100, 10)
print(x)
print(x.dtype)
print()
# Find the min
print(x.min)
print(torch.min(x))
print()
# Find the max
print(x.max)
print(torch.max(x))
print()
# Find the mean - note: the torch.mean() function requires a tensor of float32 datatype to work
print(torch.mean(x.type(torch.float32)))
# Find the sum
print()
print(x.sum())
print(torch.sum(x))

print()

# Find the positions in tensor that has the minimum value with argmin() -> return index position of target tensor where the minimum value occurs
print(x.argmin())
print(x[x.argmin()])
print()
# Find the positions in tensor that has the maximum value with argmax()
print(x.argmax())
print(x[x.argmax()])
print()
## Reshaping, stacking, squeezing and unsqueezing tensors
# * Reshaping - reshapes an input tensor to a defined shaped
# * View - Return a view of an input tensor of certain shape but keep the same memory as the original tensor
# * Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
# * Squeeze - removes all '1' dimensions from a tensor
# * Unsqueeze - add a '1' dimension to a target tensor
# * Permute - return a view of the input with dimensions permuted (swapped) in a certain way


# Let's create a tensor
x = torch.arange(1., 10.)
print(x)
print(x.shape)
print()
# Add an extra dimension
x_reshaped = x.reshape(1, 9)
print(x_reshaped)
print(x_reshaped.shape)
print()
# Change the view
z = x.view(1, 9)
print(z)
print(z.shape)

# Changing z changes x (because a view of a tensor shares the same memory as the original input)
z[:, 0] = 5
print(z)
print(x)

# Stack tensors on top of each other
x_stacked = torch.stack((x, x, x, x), dim=1)
print(x_stacked)

# torch.squeeze() - removes all single dimensions from a target tensor
print(f"Previous tensor:{x_reshaped}")
print(f"Previous shape:{x_reshaped.shape}")
print()
# Remove extra dimensions from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor:{x_squeezed}")
print(f"New shape:{x_squeezed.shape}")
print()
print(x_reshaped)
print(x_reshaped.shape)
print(x_reshaped.squeeze())
print(x_reshaped.squeeze().shape)

# torch.unsqueeze() - adds a single dimension to a target tensor at a specific dim(dimension)
print(f"\nPrevious tensor:{x_squeezed}")
print(f"Previous shape:{x_squeezed.shape}")

# Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=1)
print(f"\nNew tensor:{x_unsqueezed}")
print(f"New shape:{x_unsqueezed.shape}")

print(x_reshaped.squeeze().shape)

# torch.permute - rearranges the dimensions of a target tensor in a specified order
x_original = torch.rand(size=(224, 224, 3))  # [height,width,colour_channels]
# Permute the original tensor to rearrange the axis(or dim) order
x_permuted = x_original.permute(2, 0, 1)  # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape:{x_original.shape}")
print(f"New shape:{x_permuted.shape}")  # [colour_channels,height,width]
print(x_original[0, 0, 0])

### Indexing (select data from tensors)
# Indexing with PyTorch is similar tp indexing in NumPy
# Create a tensor
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x)
print(x.shape)
# Let's index on our new tensor
print(x[0])
# Let's index on the middle bracket(dim=1)
print(x[0][0])
# Let's index on the most inner bracket(last dimension)
# print(x[1][1][1])
print(x[0][0][0])
print(x[0][2][2])
# You can also use ":" to select "all" of a target dimension
print(x[:, 0])
# Get all values of 0th and 1st dimensions but only index 1 of 2nd dimension
print(x[:, :, 1])
# Get all values of the 0 dimension but only the 1 index value of 1st and 2nd dimension
print(x[:, 1, 1])
# Get index 0 of 0th and 1st dimension and all values of 2nd dimension
print(x[0, 0, :])

# Index on x to return 9
print(x[:, 2, 2])
# Index on x to return 3,6,9
print(x[:, :, 2])
# [1, 2, 3],
# [4, 5, 6],
# [7, 8, 9]

### PyTorchTensors & NumPy
# NumPy is a popular scientific Python numerical computing library
# And because of this, PyTorch has functionality to interact with it
# Data in NumPy,want in PyTorch tensor -> torch.from_numpy(ndarray)
# PyTorch tensor -> NumPy -> torch.Tensor.numpy()

# NumPy array to tensor
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(
    array)  # float64 is numpy's default datatype.warning:when converting from numpy to pytorch,pytorch reflects numpy's default datatype of float64 unless specified otherwise
print(array.dtype)
print(tensor.dtype)
print(torch.arange(1.0, 9.0).dtype)  # torch.float32

# Change the value od array,what will this do to 'tensor'?
array = array + 1
print(array)
print(tensor)

# Tensor to NumPy array
tensor = torch.ones(7)
numpy_tensor = tensor.numpy()
print(tensor)
print(numpy_tensor)
print(numpy_tensor.dtype)
# if i am going between pytorch and numpy,default datatype of numpy is float64

# Change the tensor, what happens tp 'numpy_tensor'?
tensor = tensor + 1
print(tensor)
print(numpy_tensor)

##Reproducibility(trying to take random out of random)
# In short,how a neural network learns:
# start with random numbers -> tensor operations -> update random numbers to try and make them better representations of the data -> again -> again -> again...
# to reduce the randomness in neural networks and pytorch comes the concept of **random F**
# essentially what the random seed does is "flavour" the randomness

# Create two random tensors
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)
print(random_tensor_A)
print(random_tensor_B)
print(random_tensor_A == random_tensor_B)

# Let's make some random but reproducible tensors

# Set the random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)
torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)
print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)

## Running tensors and PyTorch objects on the GPUs(and making faster computations)
# GPUs = faster computations on numbers,thanks to CUDA + NVIDIA hardware + PyTorch working behind the scenes to make everything hunky dory(good).


### Getting a GPU
# 1.Easiest - Use Google Colab for a free GPU
# 2.Use your own GPU - takes a little bit of setup and requires the investment of purchasing a GPU,there's lots of options...
# 3.Use cloud computing - GCP,AWS,Azure,these services allow you to rent computers on the cloud and access them


### Check GPU access with PyTorch
print(torch.cuda.is_available())
# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Count number of devices
print(torch.cuda.device_count())
## Putting tensors (and models) on the GPU
# The reason we want our tensors/models on the GPU is because using a GPU results in faster computations
# Create a tensor(default on the CPU)
tensor = torch.tensor([1, 2, 3])
# Tensor not on GPU
print(tensor, tensor.device)
# Move tensor to GPU(if available)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

### Moving tensors back to the CPU
# If tensor is on GPU, can't transform it to NumPy
print(tensor_on_gpu.numpy())
# To fix the GPU tensor with NumPy issue,we can first set it to the CPU
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)
