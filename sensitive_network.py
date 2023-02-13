import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd 
import torch.nn.functional as F

df = pd.read_csv("Data/sensitive_adult.csv")
train, test = train_test_split(df, test_size=0.2)

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(97, 20)
        #self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(20, 1)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x_race = torch.sigmoid(self.fc3(x))
        x_sex = torch.sigmoid(self.fc4(x))
        return x_race, x_sex

# Load the data
X_train = train.iloc[:,:97].values
y_race_train = train['race'].values
y_sex_train = train['gender'].values
X_test = test.iloc[:,:97].values
y_race_test = test['race'].values
y_sex_test = test['race'].values


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_race_train = torch.tensor(y_race_train, dtype=torch.float32)
y_sex_train = torch.tensor(y_sex_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_race_test = torch.tensor(y_race_test, dtype=torch.float32)
y_sex_test = torch.tensor(y_sex_test, dtype=torch.float32)

y_race_train = y_race_train.view(y_race_train.shape[0], 1)
y_sex_train = y_sex_train.view(y_sex_train.shape[0], 1)
y_race_test = y_race_test.view(y_race_test.shape[0], 1)
y_sex_test = y_sex_test.view(y_sex_test.shape[0], 1)


# Initialize the model, loss function and optimizer
model = Net()
criterion_race = nn.BCELoss()
criterion_sex = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(500):
    # Forward pass
    y_race_pred, y_sex_pred = model(X_train)
    loss_race = criterion_race(y_race_pred, y_race_train)
    loss_sex = criterion_sex(y_sex_pred, y_sex_train)
    loss = loss_race + loss_sex

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

# Make predictions
y_race_test_pred, y_sex_test_pred = model(X_test)

# Calculate accuracy
acc_race = np.mean(np.round(y_race_test_pred.detach().numpy()) == y_race_test.numpy())
acc_sex = np.mean(np.round(y_sex_test_pred.detach().numpy()) == y_sex_test.numpy())
print((acc_race+acc_sex)/2)