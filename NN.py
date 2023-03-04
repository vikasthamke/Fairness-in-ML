import pandas as pd 
import torch
import torch.nn as nn
from aif360.datasets import StandardDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv('Data/adult.csv')
df.head()

delRows = df[(df["workclass"] == "?") | (df["occupation"] == "?") | (df["native-country"] == "?")].index
df.drop(delRows, inplace=True)
df.reset_index(drop=True, inplace=True)
df.loc[df["race"] != "White", "race"] = "Non-White"

# Define the first network
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(X_train.size()[1], X_train.size()[1], bias=False)

        self.fc1.weight.data.fill_(0.0)
        for i in range(X_train.size()[1]):
            self.fc1.weight.data[i][i] = 1.0

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return x

    # Define the second network
class BaseClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 20, bias=False)
        self.fc2 = nn.Linear(20, output_size, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Define the third network
class SensitiveClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(SensitiveClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 20, bias=False)
        self.fc_race = nn.Linear(20, output_size, bias=False)
        self.fc_sex = nn.Linear(20, output_size, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        race = torch.sigmoid(self.fc_race(x))
        sex = torch.sigmoid(self.fc_sex(x))
        return race, sex


categorical_features = ['education','marital-status','workclass','relationship','occupation','native-country']
features_drop = []
accuracies=[]
losses=[]
feature_to_remove=""
while len(features_drop) != 12:
    if feature_to_remove in categorical_features: categorical_features.remove(feature_to_remove)
    features_drop.insert(0, feature_to_remove)
    dataset = StandardDataset(df, 
                              label_name='income', 
                              favorable_classes=['>50K'], 
                              protected_attribute_names=['gender','race'],
                              categorical_features=categorical_features,
                              features_to_drop=features_drop,
                              privileged_classes=[['Male'],['White']])

    df_data = dataset.convert_to_dataframe()[0]

    y_race_train = df_data['race'].values
    y_sex_train = df_data['gender'].values
    y_train = df_data['income'].values

    df_data.drop(["income","race","gender"], axis=1, inplace=True)
    X_train = df_data.values

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_race_train = torch.tensor(y_race_train, dtype=torch.float32)
    y_sex_train = torch.tensor(y_sex_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    y_race_train = y_race_train.view(y_race_train.shape[0], 1)
    y_sex_train = y_sex_train.view(y_sex_train.shape[0], 1)
    y_train = y_train.view(y_train.shape[0], 1)

    # Create the models
    model1 = MyModel()
    model2 = BaseClassifier(input_size=X_train.size()[1], output_size=1)
    model3 = SensitiveClassifier(input_size=X_train.size()[1], output_size=1)


    # Define the loss function and optimizer
    criterion1 = nn.BCELoss()
    criterion2 = nn.BCELoss()
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()) + list(model3.parameters()), lr=0.0001)

    weight_norms = []
    # Define the training loop
    num_epochs = 3000
    for epoch in range(num_epochs):
        # Set the requires_grad flag for each network
        if epoch < num_epochs/2:
            for param in model1.parameters():
                param.requires_grad = False
            for param in model2.parameters():
                param.requires_grad = True
            for param in model3.parameters():
                param.requires_grad = True
        else:
            for param in model1.parameters():
                param.requires_grad = True
            for param in model2.parameters():
                param.requires_grad = False
            for param in model3.parameters():
                param.requires_grad = False

        # Forward pass and backpropagation for each batch

        outputs1 = model1(X_train)
        outputs2 = model2(outputs1)
        outputs_race, outputs_sex = model3(outputs1)
        
        loss1 = criterion1(outputs2, y_train)
        loss2_race = criterion2(outputs_race, y_race_train)
        loss2_sex = criterion2(outputs_sex, y_sex_train)
        loss3 = loss2_race + loss2_sex
        loss_ratio = loss1 / loss3
        loss_ratio.backward()
        optimizer.step()
        
        #weight_norms.append([torch.norm(layer.weight).item() for layer in model1.children() if hasattr(layer, 'weight')])
        optimizer.zero_grad()
        # Print progress
        if (epoch+1) % 3000 == 0:
            losses.append(loss_ratio.item())
            #print(f"Epoch {epoch+1}, Loss: {loss_ratio.item():.4f}")
        
    features = list(df_data.columns)
    wt = 10
    index = 0
    for i in range(X_train.size()[1]):
        if(model1.state_dict()['fc1.weight'][i][i] < wt):
            wt = model1.state_dict()['fc1.weight'][i][i]
            index = i
    feature_to_remove = features[index].split('=')[0]
    print(features_drop)