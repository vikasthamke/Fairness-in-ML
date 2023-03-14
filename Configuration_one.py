import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Encoded_Adult.csv")

train, test = train_test_split(df, shuffle =True, test_size = 0.2)

y_race_train = train['race'].values
y_sex_train = train['gender'].values
y_train = train['income'].values

y_race_test = test['race'].values
y_sex_test = test['gender'].values
y_test = test['income'].values

train.drop(["income","race","gender"], axis=1, inplace=True)
X_train = train.values

test.drop(["income","race","gender"], axis=1, inplace=True)
X_test = test.values

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_race_train = torch.tensor(y_race_train, dtype=torch.float32)
y_sex_train = torch.tensor(y_sex_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

y_race_train = y_race_train.view(y_race_train.shape[0], 1)
y_sex_train = y_sex_train.view(y_sex_train.shape[0], 1)
y_train = y_train.view(y_train.shape[0], 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_race_test = torch.tensor(y_race_test, dtype=torch.float32)
y_sex_test = torch.tensor(y_sex_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

y_race_test = y_race_test.view(y_race_test.shape[0], 1)
y_sex_test = y_sex_test.view(y_sex_test.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


class FirstNetwork(nn.Module):
    def __init__(self):
        super(FirstNetwork, self).__init__()
        self.fc1 = nn.Linear(X_train.size()[1], X_train.size()[1], bias = False)
        self.fc1.weight.data.fill_(0.0)
        for i in range(X_train.size()[1]):
            self.fc1.weight.data[i][i] = 1.0
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return x

class BaseClassifier(nn.Module):
    def __init__(self):
        super(BaseClassifier, self).__init__()
        self.fc1 = nn.Linear(X_train.size()[1], 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,128)
        self.fc4 = nn.Linear(128,2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        out = self.softmax(self.fc4(x))
        return out

class SensitiveClassifier(nn.Module):
    def __init__(self):
        super(SensitiveClassifier, self).__init__()
        self.fc1 = nn.Linear(X_train.size()[1], 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,128)
        self.fc4 = nn.Linear(128, 2)
        self.fc5 = nn.Linear(128, 2)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        out1 = self.softmax1(self.fc4(x))
        out2 = self.softmax2(self.fc5(x))
        return out1,out2
    
 
model1 = FirstNetwork()
model2 = BaseClassifier()
model3 = SensitiveClassifier()


criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()


optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(list(model2.parameters()) + list(model3.parameters()), lr=0.0001)

num_epochs=2000
for epoch in range(num_epochs):
   
    for param in model1.parameters():
            param.requires_grad = False
    for param in model2.parameters():
            param.requires_grad = True
    for param in model3.parameters():
            param.requires_grad = True
    optimizer2.zero_grad()
    
    output = model1(X_train)
    output1 = model2(output)
    out1, out2 = model3(output)
    
    loss1 = criterion1(output1, y_train[:,0].long())
    loss2_race = criterion2(out1, y_race_train[:,0].long())  # extract first target variable
    loss2_sex = criterion2(out2, y_sex_train[:,0].long())  # extract second target variable
    loss3 = loss2_race + loss2_sex
    loss_ratio = loss1 + loss3
    loss_ratio.backward(retain_graph=True)
    
    optimizer2.step()
    if (epoch+1) % 500 == 0:
        print(f'Epoch {epoch+1}, Loss {loss_ratio.item():.4f}')


reg_lambda = 0.01
num_epochs = 2000

for epoch in range(num_epochs):
   
    for param in model1.parameters():
        if param.dim() > 1:
            param.requires_grad = True
            param.data = torch.diag(torch.diag(param.data))
        else:
            param.requires_grad = False
    for param in model2.parameters():
            param.requires_grad = False
    for param in model3.parameters():
            param.requires_grad = False
    
    optimizer1.zero_grad()
    
    output = model1(X_train)
    output1 = model2(output)
    out1, out2 = model3(output)

    loss1 = criterion1(output1, y_train[:,0].long())
    loss2_race = criterion2(out1, y_race_train[:,0].long())
    loss2_sex = criterion2(out2, y_sex_train[:,0].long())
    loss3 = loss2_race + loss2_sex
    loss_ratio = loss1 / loss3
    reg_loss_ratio = loss_ratio + reg_lambda * (loss_ratio * (1 - loss_ratio))**2
    reg_loss_ratio.backward()
    
    optimizer1.step()
    
    if (epoch+1) % 500 == 0:
        print(f'Epoch {epoch+1}, Loss {reg_loss_ratio.item():.4f}')

for name, param in model1.state_dict().items():
    if 'fc1.weight' in name:
        lst = []
        for i in range(param.size()[0]):
            lst.append(param[i][i].item())
        features = list(train.columns)
        wt_fea = list(zip(features, lst))
        print(sorted(wt_fea, key=lambda x: x[1], reverse=True))