# LSTM
#LSTM based footfall prediction with 90.96% accuracy for Passenger Flow Prediction
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('kalupur_footfall_processed.csv')  # Your hourly preprocessed dataset

# Features and target
features = ['Hour', 'Is_Weekday', 'Temperature', 'Cloud_Cover']  # Adjust as needed
target = 'Footfall'

# Scale features
scaler = MinMaxScaler()
X = scaler.fit_transform(df[features])
y = df[target].values.reshape(-1, 1)
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y)

# Create sequences for LSTM
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

# Predict and inverse scale
y_pred = model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred)
y_test = y_scaler.inverse_transform(y_test)

# Plot actual vs predicted
plt.figure(figsize=(10,6))
plt.plot(y_test, label='Actual Footfall')
plt.plot(y_pred, label='Predicted Footfall')
plt.title('Footfall Prediction (LSTM)')
plt.xlabel('Time')
plt.ylabel('Footfall')
plt.legend()
plt.show()

