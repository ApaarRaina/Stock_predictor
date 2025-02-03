import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.model_selection import train_test_split


df=pd.read_csv('Mock_2_data.csv')
df.drop(columns=['increase','percentage_increase','Ticker','Date'],inplace=True)

X=df.iloc[:,:-1]
X=np.array(X)
X=X.astype(np.float32)
y=df.iloc[:,-1]
y=np.array(y)
y=y.astype(np.long)

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


train=pd.read_csv('Train_data.csv')
test=pd.read_csv('Test_data.csv')

X_train=train.iloc[:,:-1]
y_train=train.iloc[:,-1]

X_test=test.iloc[:,:-1]
y_test=test.iloc[:,-1]

X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)

y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)


class Model(nn.Module):

    def __init__(self,in_features=18,hl1=20,hl2=15,hl3=10,out_features=2):
        super().__init__()      # instantiate the model class
        self.fc1=nn.Linear(in_features,hl1)
        self.fc2=nn.Linear(hl1,hl2)
        self.fc3=nn.Linear(hl2,hl3)
        self.fc4=nn.Linear(hl3,out_features)



    def forward(self,x):

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        out=self.fc4(x)

        return out



torch.manual_seed(41)

model=Model()

criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=0.01)


epochs=300

for i in range(epochs):

    y_pred=model.forward(X_train)
    loss=criterion(y_pred,y_train)

    if (i + 1) % 10 == 0:  # Print loss every 10 epochs
        print(f'Epoch [{i + 1}/{epochs}], Loss: {loss.item():.4f}')

    #back propogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()   #updates the parameters




model.eval()
with torch.no_grad():
     y_pred=model(X_test)
     y_pred = torch.argmax(y_pred, axis=1)
     f_score = fbeta_score(y_test, y_pred, beta=0.5)
     acc = accuracy_score(y_test, y_pred)
     print(f_score)
     print(acc)
     print(y_pred)