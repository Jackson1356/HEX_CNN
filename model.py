import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.optim as optim
import numpy as np
import pandas as pd
import scanpy as sc
import hexagdly
import os
import math
import data_processing
import argparse
import matplotlib.pyplot as plt

def to_dataloader(features, labels, batch_size=2):
    data = torch.tensor(features, dtype=torch.float32)
    labels = np.asarray(labels)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    train_data, val_data = data[:7], data[7:]
    train_labels, val_labels = labels[:7], labels[7:]
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

class hex_model(nn.Module):
    def __init__(self, nin):          
        super(hex_model, self).__init__()
        self.name = 'hex_model'
        self.hexconv1 = hexagdly.Conv2d(in_channels = nin, out_channels = 128, \
                                         kernel_size = 2, stride = 1, bias=True)
        self.hexpool1 = hexagdly.MaxPool2d(kernel_size = 2, stride = 2)
        self.hexconv2 = hexagdly.Conv2d(128, 64, 2, 1, bias=True)
        self.hexpool2 = hexagdly.MaxPool2d(kernel_size = 2, stride = 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * 32 * 39, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.hexconv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.hexpool1(x)

        x = self.hexconv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.hexpool2(x)

        x = self.dropout(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


#LOOCV training
def main(args):
    data_path = args.input_path
    file_nums = [5,7,12,19,208,209,1294]
    input_channel = args.nint
    features, labels = data_processing(data_path, file_nums, input_channel)

    data = torch.tensor(features, dtype=torch.float32)
    labels = np.asarray(labels)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(data, labels)

    epoch_acc = []
    epoch_train_losses = []
    epoch_val_losses = []

    for i in range(len(dataset)):
        train_indices = list(range(len(dataset)))
        train_indices.pop(i)
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, [i])

        train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = hex_model(input_channel).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005)

        fold_train_losses = []
        fold_val_losses = []
        fold_acc = []
        
        for epoch in range(args.epochs):
            # training period
            model.train()
            training_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            fold_train_losses.append(training_loss / len(train_loader))
        
            # evaluation period
            model.eval()
            total_val_loss = 0
            total_abs_error = 0
            count_samples = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    total_val_loss += loss.item()
                    total_abs_error += torch.abs(outputs - targets).sum().item()
                    count_samples += targets.size(0)
            fold_val_losses.append(total_val_loss / len(test_loader))
            fold_acc.append(total_abs_error / count_samples)
            print(f'Fold {i+1}, Epoch {epoch+1}, Train Loss: {fold_train_losses[-1]}, Validation Loss: {fold_val_losses[-1]}, MAE: {fold_epoch_acc[-1]}')

        epoch_train_losses.append(fold_train_losses)
        epoch_val_losses.append(fold_val_losses)
        epoch_acc.append(fold_acc)

    mean_epoch_train_losses = np.mean(epoch_train_losses, axis=0)
    mean_epoch_val_losses = np.mean(epoch_val_losses, axis=0)
    mean_epoch_acc = np.mean(epoch_acc, axis=0)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    plt.savefig(f'{args.output_dir}/losses.png')


# def main(args):
#     data_path = args.input_path
#     file_nums = [5,7,12,19,208,209,1294]
#     input_channel = args.nint
#     features, labels = data_processing(data_path, file_nums, input_channel)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = hex_model(input_channel).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.005)
#     train_losses, val_losses = [], []
#     for epoch in range(args.epochs):
#         # training period
#         model.train()
#         running_loss = 0.0
#         for inputs, targets in train_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         train_losses.append(running_loss/len(train_loader))
    
#         # evaluation period
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for inputs, targets in val_loader:
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 val_loss += loss.item()
#         val_losses.append(val_loss/len(val_loader))
        
#         print(f'Epoch {epoch+1}, Train Loss: {training_losses[-1]}, Validation Loss: {val_losses[-1]}')

#     plt.plot(training_losses, label='Training loss')
#     plt.plot(val_losses, label='Validation loss')
#     plt.legend()
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#     plt.savefig(f'{args.output_dir}/losses.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hexagonal CNN predicting response based on spatial transcriptomics')
    parser.add_argument('-i', '--nint', default=1000, type=int,
                        help='input channel of feature (default: 1000)')
    parser.add_argument('-e', '--epochs', default=25, type=int,
                        help='input channel of feature (default: 25)')
    parser.add_argument('-i', '--input_path', default='Visium', type=str,
                        help='output directory')
    parser.add_argument('-o', '--output_path', default='experiments', type=str,
                        help='output directory')
    args = parser.parse_args()
    main(args)