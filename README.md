# DL- Developing a Recurrent Neural Network Model for Stock Prediction


## DESIGN STEPS
### STEP 1: 

Load and normalize data, create sequences.



### STEP 2: 

Convert data to tensors and set up DataLoader.



### STEP 3: 

Define the RNN model architecture.



### STEP 4: 

Summarize, compile with loss and optimizer.



### STEP 5: 

Train the model with loss tracking.



### STEP 6: 


Predict on test data, plot actual vs. predicted prices.


## PROGRAM

### Name: SANJAYKUMAR B

### Register Number: 212224230242

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# If torchinfo is not installed, run:
# !{sys.executable} -m pip install torchinfo
from torchinfo import summary


# ==========================================
# STEP 1: LOAD DATA
# ==========================================

df_train = pd.read_csv(r"C:\Users\admin\Downloads\trainset.csv")
df_test = pd.read_csv(r"C:\Users\admin\Downloads\testset.csv")

print("Training data:")
print(df_train.head())

print("\nTesting data:")
print(df_test.head())


# ==========================================
# STEP 2: GET CLOSING PRICES
# ==========================================

train_prices = df_train["Close"].values.reshape(-1, 1)
test_prices = df_test["Close"].values.reshape(-1, 1)


# ==========================================
# STEP 3: NORMALIZE DATA
# ==========================================

scaler = MinMaxScaler()

scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)


# ==========================================
# STEP 4: CREATE SEQUENCES
# ==========================================

def create_sequences(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(x), np.array(y)


seq_length = 60

x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

print("\nShapes:")
print("X Train:", x_train.shape)
print("Y Train:", y_train.shape)
print("X Test :", x_test.shape)
print("Y Test :", y_test.shape)


# ==========================================
# STEP 5: CONVERT TO PYTORCH TENSORS
# ==========================================

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# ==========================================
# STEP 6: CREATE DATASET AND DATALOADER
# ==========================================

train_dataset = TensorDataset(
    x_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# ==========================================
# STEP 7: DEFINE RNN MODEL
# ==========================================

class RNNModel(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=2,
        output_size=1
    ):
        super(RNNModel, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        out, _ = self.rnn(x)

        out = self.fc(
            out[:, -1, :]
        )

        return out


# ==========================================
# STEP 8: CREATE MODEL
# ==========================================

model = RNNModel()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

print("\nDevice:", device)


# ==========================================
# STEP 9: MODEL SUMMARY
# ==========================================

summary(
    model,
    input_size=(64, 60, 1)
)


# ==========================================
# STEP 10: LOSS AND OPTIMIZER
# ==========================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# STEP 11: TRAIN MODEL
# ==========================================

epochs = 20

model.train()

train_losses = []

for epoch in range(epochs):

    epoch_loss = 0

    for x_batch, y_batch in train_loader:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(x_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)

    train_losses.append(average_loss)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
    )


# ==========================================
# STEP 12: PLOT TRAINING LOSS
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    marker="o"
)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.show()


# ==========================================
# STEP 13: TEST THE MODEL
# ==========================================

model.eval()

with torch.no_grad():

    x_test_tensor = x_test_tensor.to(device)

    predictions = model(
        x_test_tensor
    )

    predictions = predictions.cpu().numpy()


# ==========================================
# STEP 14: CONVERT BACK TO ORIGINAL PRICE
# ==========================================

predicted_prices = scaler.inverse_transform(
    predictions
)

actual_prices = scaler.inverse_transform(
    y_test
)


# ==========================================
# STEP 15: PLOT ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    actual_prices,
    label="Actual Price"
)

plt.plot(
    predicted_prices,
    label="Predicted Price"
)

plt.title("Actual vs Predicted Stock Price")

plt.xlabel("Time")

plt.ylabel("Closing Price")

plt.legend()

plt.show()


# ==========================================
# STEP 16: CALCULATE TEST LOSS
# ==========================================

model.eval()

with torch.no_grad():

    test_input = x_test_tensor.to(device)
    test_target = y_test_tensor.to(device)

    test_output = model(test_input)

    test_loss = criterion(
        test_output,
        test_target
    )

print("\nTest Loss:", test_loss.item())
```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="451" height="491" alt="image" src="https://github.com/user-attachments/assets/d72c16c2-43c8-4423-863d-2be541a67ba1" />


## True Stock Price, Predicted Stock Price vs time

<img width="721" height="569" alt="image" src="https://github.com/user-attachments/assets/0f6c0fdc-8a85-4cb3-acd1-3e8c6b48298b" />

### Predictions
<img width="985" height="618" alt="image" src="https://github.com/user-attachments/assets/502b3baa-e62a-4119-b14f-feffc8186307" />

<img width="326" height="93" alt="image" src="https://github.com/user-attachments/assets/b836e1c4-7d9b-4121-9515-45df118f6337" />


## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
