import torch
from torch import nn
import numpy as np
import torch.onnx
import sys
import os
import pandas as pd
from tqdm import tqdm
# terminal_command = 'wget --no-cache --backups=1 https://raw.githubusercontent.com/DDiekmann/Applied-Verification-Lab-Neural-Networks/main/lib/mnist_trainer.py'

# os.system(terminal_command)

import matplotlib.pyplot as plt
sys.path.insert(1, '/Users/moonkyung/Desktop/Marabou')
from maraboupy import Marabou
from maraboupy import MarabouCore
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
options = Marabou.createOptions(verbosity = 0)
transform = transforms.Compose([
        transforms.ToTensor(),
    ])
training_data = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_dataloader=DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
device = "cuda" if torch.cuda.is_available() else "cpu"
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, number_of_neurons):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, number_of_neurons),
            nn.GELU(),
            nn.Linear(number_of_neurons, number_of_neurons*2),
            nn.GELU(),
            nn.Linear(number_of_neurons*2, number_of_neurons*4),
            nn.GELU(),
            nn.Linear(number_of_neurons*4, number_of_neurons*8),
            nn.GELU(),
            nn.Linear(number_of_neurons*8, number_of_neurons*8),
            nn.GELU(),
            nn.Linear(number_of_neurons*8, number_of_neurons*8),
            nn.GELU(),
            nn.Linear(number_of_neurons*8, number_of_neurons*4),
            nn.GELU(),
            nn.Linear(number_of_neurons*4, number_of_neurons*2),
            nn.GELU(),
            nn.Linear(number_of_neurons*2, number_of_neurons),
            nn.GELU(),
            nn.Linear(number_of_neurons, output_dim),
        )

    def forward(self, x):
        logits = self.model(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct, test_loss


def train_model(model, epochs, train_dataloader, test_dataloader):
    torch.manual_seed(42)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        correct, test_loss = test(test_dataloader, model, loss_fn)
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    print("Done!")
    return model

model = train_model(
    NeuralNetwork(input_dim=28*28, output_dim=10, number_of_neurons=20), 
    epochs=20, 
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    )

model_filename = "/Users/moonkyung/Desktop/Marabou/project/fmnist_GeLU_model_del_DIV.onnx"

# set model to eval mode
model.eval()

# create a dummy input in the shape of the input values
dummy_input = torch.randn(1, 28, 28)

dummy_input = dummy_input.to(device)

# torch.onnx.export(model,
#                   dummy_input,
#                   model_filename,
#                   export_params=True,
#                   verbose=False,
#                   )

torch.onnx.export(
    model,
    dummy_input,
    model_filename,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}}
)

