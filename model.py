import numpy as np
import torch
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AndModel(nn.Module):
    def __init__(self):
        super(AndModel, self).__init__()
        self.layer1 = nn.Linear(2, 10, bias=True)
        self.layer2 = nn.Linear(10, 10, bias=True)
        self.layer3 = nn.Linear(10, 1, bias=True)
        self.relu = nn.ReLU()
        self.to(device)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x    

    def train(self):
        learning_rate = 0.1
        epochs = 100
        inputs = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
        outputs = torch.FloatTensor([[0], [0], [0], [1]]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(inputs)
            loss = criterion(output, outputs)
            loss.backward()
            optimizer.step()
            print(f'epoch: {epoch}, loss: {loss.item()}')
  
    def predict(self, input_data):     
        input_data = torch.FloatTensor(input_data).to(device)
        output = torch.sigmoid(self(input_data)).item()
        return int(output > 0.5)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

class XorModel(nn.Module):
    def __init__(self):
        super(XorModel, self).__init__()
        self.layer1 = nn.Linear(2, 10, bias=True)
        self.layer2 = nn.Linear(10, 10, bias=True)
        self.layer3 = nn.Linear(10, 10, bias=True)
        self.layer4 = nn.Linear(10, 1, bias=True)
        self.relu = nn.ReLU()
        self.to(device)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))   
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x
        
    def train(self, mode=True):
        learning_rate = 0.1
        epochs = 100
        inputs = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
        outputs = torch.FloatTensor([[0], [1], [1], [0]]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(inputs)
            loss = criterion(output, outputs)
            loss.backward()
            optimizer.step()
            print(f'epoch: {epoch}, loss: {loss.item()}')

    def predict(self, input_data) ->int:
        input_data = torch.FloatTensor(input_data).to(device)
        output = torch.sigmoid(self(input_data)).item()
        return int(output > 0.5)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

class OrModel(nn.Module):
    def __init__(self):
        # 파라메터
        super(OrModel, self).__init__()
        self.layer1 = nn.Linear(2, 10, bias=True)
        self.layer2 = nn.Linear(10, 10, bias=True)
        self.layer3 = nn.Linear(10, 1, bias=True)
        self.relu = nn.ReLU()
        self.to(device)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))   
        x = self.layer3(x)
        return x
    
    def train(self, mode = True):
        learning_rate = 0.1
        epochs = 100
        inputs = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
        outputs = torch.FloatTensor([[0], [1], [1], [1]]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(inputs)
            loss = criterion(output, outputs)
            loss.backward()
            optimizer.step()
            print(f'epoch: {epoch}, loss: {loss.item()}')
            
    def predict(self, input_data) ->int: 
        input_data = torch.FloatTensor(input_data).to(device)
        output = torch.sigmoid(self(input_data)).item()
        return int(output > 0.5)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
class NotModel(nn.Module):
    
    def __init__(self):
        super(NotModel, self).__init__()
        self.layer1 = nn.Linear(1, 10, bias=True)
        self.layer2 = nn.Linear(10, 10, bias=True)
        self.layer3 = nn.Linear(10, 1, bias=True)
        self.relu = nn.ReLU()
        self.to(device)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))   
        x = self.layer3(x)
        return x

    def train(self, mode = True):
        inputs = torch.FloatTensor([[0], [1]]).to(device)
        outputs = torch.FloatTensor([[1], [0]]).to(device)
        learning_rate = 0.1
        epochs = 100
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(inputs)
            loss = criterion(output, outputs)
            loss.backward()
            optimizer.step()
            print(f'epoch: {epoch}, loss: {loss.item()}')
        
    def predict(self, input_data) ->int:
        input_data = torch.FloatTensor(input_data).to(device)
        output = torch.sigmoid(self(input_data)).item()
        return int(output > 0.5)
    
    def save(self, path):
        torch.save(self.state_dict(), path) 
        
    def load(self, path):
        self.load_state_dict(torch.load(path))

