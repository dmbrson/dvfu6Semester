import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df_train = pd.read_csv("DailyDelhiClimateTrain.csv")
df_test = pd.read_csv("DailyDelhiClimateTest.csv")

df = pd.concat([df_train, df_test], ignore_index=True)
df['date'] = pd.to_datetime(df['date'])

df = df.sort_values('date')
print(df.head())

plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['meantemp'], label='Mean Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (C)')
plt.legend()
plt.show()

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df.drop(columns=['date'], inplace=True)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

N = 60
x_train, x_test, y_train, y_test = [], [], [], []

max_year = df['year'].max()

for i in range(len(df_scaled) - N):
    seq = df_scaled.iloc[i:i + N].values
    target = df_scaled['meantemp'].iloc[i + N]

    if df['year'].iloc[i + N] == max_year:
        x_test.append(seq)
        y_test.append(target)
    else:
        x_train.append(seq)
        y_train.append(target)

x_train, x_test = np.array(x_train), np.array(x_test)
y_train, y_test = np.array(y_train), np.array(y_test)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64),
    BatchNormalization(),
    Dropout(0.3),

    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_data=(x_test, y_test))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

y_pred = model.predict(x_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")

plt.figure(figsize=(14, 6))
plt.plot(y_test, label='Real')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("RNN Temperature Prediction")
plt.show()
