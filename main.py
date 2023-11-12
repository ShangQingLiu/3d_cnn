import sys

import pytorchvideo.models.resnet
import torch
import pandas as pd
from torch import nn
import os

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import json
import numpy as np

import wandb
import torch
from rich.progress import Progress

from util import load_dataset, CustomHandObjectDataset


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def make_kinetics_resnet():
    n = pytorchvideo.models.resnet.create_resnet(
        input_channel=2,  # RGB input from Kinetics
        model_depth=50,  # For the tutorial let's just use a 50 layer network
        model_num_class=8,  # Kinetics has 400 classes so we need out final head to align
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )
    csn = pytorchvideo.models.csn.create_csn(
        input_channel=2,  # RGB input from Kinetics
        model_depth=50,  # For the tutorial let's just use a 50 layer network
        model_num_class=8,  # Kinetics has 400 classes so we need out final head to align
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )
    # x = pytorchvideo.models.x3d.create_x3d_stem(
    #     input_channel=2,  # RGB input from Kinetics
    #     model_depth=50,  # For the tutorial let's just use a 50 layer network
    #     model_num_class=28,  # Kinetics has 400 classes so we need out final head to align
    #     norm=nn.BatchNorm3d,
    #     activation=nn.ReLU,
    # )
    return n


# %%
#
net = make_kinetics_resnet()
# %%


base = "/dataHDD/1sliu/EhoA/hand_object/frames_number_16_split_by_number/"
train_data, test_data = load_dataset(base)

# Define your dataset and dataloaders (replace with your dataset and data loaders)
train_dataset = CustomHandObjectDataset(train_data, base)
val_dataset = CustomHandObjectDataset(test_data, base)
batch_size = 2
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
# Define the VideoClassifier model as shown in the previous code snippet
# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# Training loop
num_epochs = 1000
# %%

# Initialize Weights and Biases
wandb.init(project="your_project_name", name="training_name")
# Initialize a Progress instance for tracking progress
progress = Progress()
# Create a task for training progress
training_task = progress.add_task("[cyan]Training...", total=len(train_loader))
# Initialize variables to keep track of the best validation accuracy and checkpoint
best_val_accuracy = 0.0
best_checkpoint = None
# Move the model to GPU
net.to('cuda:1')
for epoch in range(num_epochs):
    net.train()  # Set the model to training mode
    total_loss = 0.0
    correct = 0
    total = 0
    with progress:
        for inputs, labels in train_loader:
            inputs = inputs.to("cuda:1")
            inputs = inputs.permute(0, 2, 1, 3, 4)
            labels = labels.to("cuda:1")
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # Update the model parameters
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Update the training progress
            progress.update(training_task, advance=1)
        train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        # Log training progress to Weights and Biases
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy
        })
    if epoch % 5 == 0:
        # Validation loop
        net.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient computation during validation
            for inputs, labels in val_loader:
                inputs = inputs.to("cuda:1")
                inputs = inputs.permute(0, 2, 1, 3, 4)
                labels = labels.to("cuda:1")
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss = total_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        # Log validation progress to Weights and Biases
        wandb.log({
            "Epoch": epoch + 1,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy
        })
        # Check if the current validation accuracy is higher than the best
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_checkpoint = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            print('get points!!')
            torch.save(best_checkpoint, './best_tmp_checkpoint.pth')
# Save the best checkpoint
if best_checkpoint is not None:
    print('get points!!')
    torch.save(best_checkpoint, './best_checkpoint.pth')
# Mark the progress as completed
progress.stop()
wandb.finish()  # Finish Weights and Biases run
