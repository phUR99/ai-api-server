import numpy as np
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AndModel:
    def __init__(self):
        # 파라메터
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)

    def train(self):
        learning_rate = 0.1
        epochs = 20
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        outputs = np.array([0, 0, 0, 1])        
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # 총 입력 계산
                total_input = np.dot(inputs[i], self.weights) + self.bias
                # 예측 출력 계산
                prediction = self.step_function(total_input)
                # 오차 계산
                error = outputs[i] - prediction
                print(f'inputs[i] : {inputs[i]}')
                print(f'weights : {self.weights}')
                print(f'bias before update: {self.bias}')
                print(f'prediction: {prediction}')
                print(f'error: {error}')
                # 가중치와 편향 업데이트
                self.weights += learning_rate * error * inputs[i]
                self.bias += learning_rate * error
                print('====')        

    def step_function(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, input_data):
        total_input = np.dot(input_data, self.weights) + self.bias
        return self.step_function(total_input)    

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
        
    def train(self):
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
    
