# Create a random tensor with shape (7, 7).
# Perform a matrix multiplication on the tensor from 2 with another random tensor
# with shape (1, 7) (hint: you may have to transpose the second tensor).
# Set the random seed to 0 and do exercises 2 & 3 over again.
# Speaking of random seeds, we saw how to set it with torch.manual_seed() but is there a GPU equivalent?
# (hint: you'll need to look into the documentation for torch.cuda for this one).
# If there is, set the GPU random seed to 1234.
# Create two random tensors of shape (2, 3) and send them both to the GPU (you'll need access to a GPU for this).
# Set torch.manual_seed(1234) when creating the tensors (this doesn't have to be the GPU random seed).
# Perform a matrix multiplication on the tensors you created in 6 (again, you may have to adjust the shapes of
# one of the tensors).
# Find the maximum and minimum values of the output of 7.
# Find the maximum and minimum index values of the output of 7.
# Make a random tensor with shape (1, 1, 1, 10) and then create a new tensor with all the 1 dimensions removed
# to be left with a tensor of shape (10). Set the seed to 7 when you create it and print out the first tensor
# and it's shape as well as the second tensor and it's shape.
import torch

firstTensor = torch.rand(7, 7)
print(firstTensor)
secondTensor = torch.rand(1, 7)
print(torch.matmul(firstTensor, secondTensor.T))

torch.manual_seed(1234)
firstTensor = torch.rand(7, 7)
print(firstTensor)
torch.manual_seed(1234)
secondTensor = torch.rand(1, 7)
print(torch.matmul(firstTensor, secondTensor.T))

seed = 32
torch.cuda.manual_seed(seed)
firstTensor = torch.rand(7, 7)
print(firstTensor)
torch.cuda.manual_seed(seed)
secondTensor = torch.rand(1, 7)
print(torch.matmul(firstTensor, secondTensor.T))

firstRandomTensor = torch.rand(2, 3)
secondRandomTensor = torch.rand(2, 3)
device = "cpu"
firstRandomTensor = firstRandomTensor.to(device)
secondRandomTensor = secondRandomTensor.to(device)
print(torch.matmul(firstRandomTensor, secondRandomTensor.T))
print(torch.max(firstRandomTensor))
print(torch.max(secondRandomTensor))
print(torch.min(firstRandomTensor))
print(torch.min(secondRandomTensor))

print(torch.argmax(firstRandomTensor))
print(torch.argmin(secondRandomTensor))
print(torch.argmax(firstRandomTensor))
print(torch.argmin(secondRandomTensor))

tensor = torch.rand(1, 1, 1, 10)
torch.manual_seed(7)
print(tensor.shape)
new_tensor = tensor.view(-1)
print(new_tensor.shape)