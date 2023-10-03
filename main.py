import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import torch.optim as optim
import torch.nn as nn
import torch
import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(33, 256)
        self.hidden2 = nn.Linear(256, 128)  # Add a new hidden layer
        self.hidden3 = nn.Linear(128, 64)   # Add another hidden layer
        self.output = nn.Linear(64, 5)
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))  # Apply ReLU activation to the first hidden layer
        x = F.relu(self.hidden2(x))  # Apply ReLU activation to the second hidden layer
        x = F.relu(self.hidden3(x))  # Apply ReLU activation to the third hidden layer
        x = self.output(x)
        return x

model = Multiclass().to(DEVICE)

data = pd.read_csv("Balanced PFCP APP Layer/15-sec-CSV/Training/Training_15s.csv")
data = data.drop(columns='flow ID')
data = data.drop(columns=' destination IP')
data = data.drop(columns=' source IP')

X = data.iloc[:, 0:-1]
y = data.iloc[:, -1:]
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)

y = ohe.transform(y)

X = torch.tensor(X.values, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).to(DEVICE)
print("X = ", X)
print("y = ", y)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0000001)

n_epochs = 1000
batch_size = 5
batches_per_epoch = len(X)

for epoch in range(n_epochs):
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            X_batch = X[start:start+batch_size]
            y_batch = y[start:start+batch_size]

            # Convert one-hot encoded y_batch to class labels
            y_batch_labels = torch.argmax(y_batch, dim=1)

            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch_labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()
    y_pred = model(X_test)
    ce = loss_fn(y_pred, y_test)
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
    print(f"Epoch {epoch} validation: Cross-entropy={float(ce)}, Accuracy={float(acc)}")
