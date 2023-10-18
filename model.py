import torch
import torch.nn as nn
import torch.nn.functional as F


# Redefine the Simple1DCNN model
class firstCNN(nn.Module):
    def __init__(self, num_classes=24):
        super(firstCNN, self).__init__()

        # 1D convolutional layers
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 128, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First convolutional layer
        x = self.pool(F.relu(self.conv1(x)))

        # Second convolutional layer
        x = self.pool(F.relu(self.conv2(x)))

        # Third convolutional layer
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)

        # First fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Second fully connected layer
        x = self.fc2(x)

        return x


# Create an instance of the model
model = firstCNN()

# Redefine the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

