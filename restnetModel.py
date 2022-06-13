import glob
import os
import sys

import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

data_path = sys.argv[1]
results_path = sys.argv[2]

class HandwrittingDataset(Dataset):

    def __init__(self, data_path):
        self.imgs_path = data_path
        file_list = glob.glob(self.imgs_path + "*")
        print(file_list)
        self.data = []

        for class_path in file_list:
            #print("class_path:", class_path)
            class_name = class_path.split("/")[-1]
            #print("class_name", class_name)
            for img_path in glob.glob(class_path + "/*.png"):
                #print("img_path: {}".format(img_path))
                self.data.append([img_path, class_name])

        #print(self.data)
        self.class_map = {"Pol": 0, "Brayan": 1, "Cinta": 2}
        self.img_dim = (700, 700)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id


# use torch to make standard resNet
class ResSim(nn.Module):
    def __init__(self, num_classes=3):
        super(ResSim, self).__init__()

        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv21 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(175 * 175 * 64, num_classes)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.float()

        out11 = self.relu(self.conv11(x))
        out12 = self.relu(self.conv12(out11)) + out11  # Residual connection 1

        out = self.maxpool(out12)

        out21 = self.relu(self.conv21(out))
        out = self.relu(self.conv22(out21)) + out21  # Residual connection 2
        out = self.maxpool(out)

        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


############################################################################################################
#                                        Initialize variables                                              #
############################################################################################################

CNN = ResSim()
# CNN = CNN.cuda()

# Cross entropy loss for classification problems
criterion = nn.CrossEntropyLoss()

# Initialize optimizer
learning_rate = .001
optimizer = torch.optim.Adam(CNN.parameters(), lr=learning_rate)

# Device configuration (choose GPU if it is available )
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 5


############################################################################################################

# Train the model
def train_model():
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    train_dataset = HandwrittingDataset(data_path + '/' + 'train/')
    train_data_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)

    CNN.train()  # Set the model in train mode
    total_step = len(train_data_loader)
    # Iterate over epochs
    for epoch in range(num_epochs):
        # Iterate the dataset
        for i, (images, labels) in enumerate(train_data_loader):
            # Get batch of samples and labels
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = CNN(images)  # algo
            loss = criterion(outputs, torch.max(labels, 1)[1])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Save the model checkpoint
    torch.save(CNN.state_dict(), results_path + 'model.ckpt')
    # to load : model.load_state_dict(torch.load(save_name_ori))


# Load the model
def load_and_test():
    CNN.load_state_dict(torch.load(results_path + 'model.ckpt'))

    # Test the model

    # Load test dataset
    test_dataset = HandwrittingDataset(data_path + '/' + 'test')
    test_data_loader = DataLoader(test_dataset, batch_size=25, shuffle=True)
    CNN.eval()  # Set the model in evaluation mode

    # Compute testing accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            # get network predictions
            outputs = CNN(images)

            # get predicted class
            _, predicted = torch.max(outputs.data, 1)
            # compare with the ground-truth
            total += labels.size(0)
            print('labels size = {}'.format(labels.size(0)))
            print('total = {}'.format(total))
            correct += (predicted == labels).sum().item()
            print('predicted = {}'.format(predicted))
            print('labels = {}'.format(labels))
            print('correct = {}'.format(correct))
            acc = 100 * correct / total

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(acc))
        return acc


train_model()
load_and_test()