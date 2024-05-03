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
    file_nums = [5,7,12,19,21,27,208,209,1294]
    input_channel = args.nint
    features = data_processing.data_processing(data_path, file_nums, input_channel)
    labels = [1, 0, 0, 0, 0, 1, 0, 1, 1]

    data = torch.tensor(features, dtype=torch.float32)
    labels = np.asarray(labels)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(data, labels)

    epoch_acc = []
    epoch_train_losses = []
    epoch_test_losses = []

    for i in range(len(dataset)-1):
        train_indices = list(range(len(dataset)))
        train_indices.pop(i)
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, [i])

        train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = hex_model(input_channel).to(device)
        criterion = nn.MSELoss()
        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr)

        fold_train_losses = []
        fold_test_losses = []
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
            test_loss = 0
            abs_error = 0
            count_samples = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    abs_error += torch.abs(outputs - targets).sum().item()
                    count_samples += targets.size(0)
            fold_test_losses.append(test_loss / len(test_loader))
            fold_acc.append(abs_error / count_samples)
            print(f'Fold {i+1}, Epoch {epoch+1}, Train Loss: {fold_train_losses[-1]}, Test Loss: {fold_test_losses[-1]}, MAE: {fold_acc[-1]}')

        epoch_train_losses.append(fold_train_losses)
        epoch_test_losses.append(fold_test_losses)
        epoch_acc.append(fold_acc)

    mean_epoch_train_losses = np.mean(epoch_train_losses, axis=0)
    mean_epoch_test_losses = np.mean(epoch_test_losses, axis=0)
    mean_epoch_acc = np.mean(epoch_acc, axis=0)

    fig, ax = plt.subplots(2, figsize=(14, 9))
    ax[0].plot(mean_epoch_train_losses)
    ax[0].set_title('Training loss')
    ax[1].plot(mean_epoch_test_losses)
    ax[1].set_title('Test loss')
    ax[1].set(xlabel='epoch')
    ax[1].text(0.02, 0.02, f"input_channel={input_channel}, lr={lr}, loss function: MSE Loss", fontsize=13, transform=plt.gcf().transFigure)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    plt.savefig(f'{args.output_path}/{args.output_filename}.png')


# def main(args):
#     data_path = args.input_path
#     file_nums = [5,7,12,19,21,27,208,209,1294]
#     input_channel = args.nint
#     features = data_processing.data_processing(data_path, file_nums, input_channel)
#     labels = [1, 0, 0, 0, 0, 1, 0, 1, 1]
#     train_loader, val_loader = to_dataloader(features, labels, batch_size=2)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = hex_model(input_channel).to(device)
#     criterion = nn.MSELoss()
#     lr = args.lr
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     epochs = 25
#     train_losses, val_losses = [], []
#     for epoch in range(epochs):
#         # training period
#         model.train()
#         running_loss = 0
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
        
#         print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')

#     fig, ax = plt.subplots(2, figsize=(14, 9))
#     ax[0].plot(train_losses)
#     ax[0].set_title('Training loss')
#     ax[0].set(xlabel='epoch')
#     ax[1].plot(val_losses)
#     ax[0].set_title('Validation loss')
#     ax[1].text(0.02, 0.02, f"input_channel={input_channel}, lr={lr}, loss function: MSE Loss", fontsize=13, transform=plt.gcf().transFigure)

#     if not os.path.exists(args.output_path):
#         os.makedirs(args.output_path)
#     plt.savefig(f'{args.output_path}/{args.output_filename}.png')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hexagonal CNN predicting response based on spatial transcriptomics')
    parser.add_argument('-n', '--nint', default=1000, type=int,
                        help='input channel of feature (default: 1000)')
    parser.add_argument('-e', '--epochs', default=25, type=int,
                        help='input channel of feature (default: 25)')
    parser.add_argument('-i', '--input_path', default='Visium', type=str,
                        help='output directory')
    parser.add_argument('-o', '--output_path', default='experiments', type=str,
                        help='output directory')
    parser.add_argument('-f', '--output_filename', default='losses', type=str,
                        help='output directory')
    parser.add_argument('-l', '--lr', default=0.005, type=float,
                    help='learning rate')
    args = parser.parse_args()
    main(args)