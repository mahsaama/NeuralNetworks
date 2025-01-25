from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import GTSRB
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision import models


# You may add aditional augmentations, but don't change the output size
_resize_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((32, 32))
])


def get_data_split(transform: Optional[Callable] = _resize_transform):
    """
    Downloads and returns the test and train set of the German Traffic Sign Recognition Benchmark (GTSRB)
    dataset.

    :param transform: An optional transform applied to the images
    :returns: Train and test Dataset instance
    """
    train_set = GTSRB(root="./data", split="train", download=True, transform=transform)
    test_set = GTSRB(root="./data", split="test", download=True, transform=transform)
    return train_set, test_set


# Implement your CNN and finetune ResNet18
# Don't forget to submit your loss and accuracy results in terms of the log file.


class GTSRBNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(50176, 256), 
            nn.ReLU(), 
            nn.Linear(256, 43)
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.net(x)



train_set, test_set = get_data_split()
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

model = GTSRBNetwork(num_classes=43)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def train_and_evaluate():
    for epoch in range(20):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100.0 * correct / total
        train_loss /= total

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_accuracy = 100.0 * correct / total
        test_loss /= total

        print(f"Epoch [{epoch+1}/{20}]: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% Test Loss: {test_loss:.4f}, Test Accuracy:{test_accuracy:.2f}%")
train_and_evaluate()




train_set, test_set = get_data_split()
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

model = models.resnet18(pretrained=True)
num_classes = 43
model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def train_and_evaluate():
    for epoch in range(20):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100.0 * correct / total
        train_loss /= total

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_accuracy = 100.0 * correct / total
        test_loss /= total

        print(f"Epoch [{epoch+1}/{20}]: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% Test Loss: {test_loss:.4f}, Test Accuracy:{test_accuracy:.2f}%")

train_and_evaluate()
