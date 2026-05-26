import yfinance as yf
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

ticker = "AAPL"
file_path = f"{ticker}_data.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
else:
    data = yf.download(ticker, period="5y", interval="1d", progress=False)
    df = data[['Close']].copy()
    df = df.dropna()
    df.to_csv(file_path)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values.astype(float))

seq_length = 90

X = []
y = []

for i in range(len(scaled_data) - seq_length):
    X.append(scaled_data[i:i + seq_length])
    y.append(scaled_data[i + seq_length])

X = np.array(X)
y = np.array(y)

train_size = int(len(X) * 0.8)

X_train = torch.FloatTensor(X[:train_size])
y_train = torch.FloatTensor(y[:train_size])
X_test = torch.FloatTensor(X[train_size:])
y_test = torch.FloatTensor(y[train_size:])

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=32, shuffle=False)

class LSTM_Predictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM_Predictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.6f}")

torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")

model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        pred = model(batch_X)
        predictions.extend(pred.numpy())
        actuals.extend(batch_y.numpy())

predictions = scaler.inverse_transform(predictions)
actuals = scaler.inverse_transform(actuals)

plt.figure(figsize=(12, 6))
plt.plot(actuals, label='Actual Price', color='blue')
plt.plot(predictions, label='Predicted Price', color='red')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.legend()
plt.savefig('Figure_1.png')
plt.show()