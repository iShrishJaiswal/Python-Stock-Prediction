#Import libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Read the data
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2022-10-15')

#plot the data
plt.figure(figsize=(16,8))
plt.title('Close price history')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price in USD($)', fontsize=18)
plt.show()

#Take only closing price
data = df.filter(['Close'])

# convert to numpy array
dataset = data.values 

# number of rows to train the model (80% of the original data)
training_data_length = math.ceil(len(dataset) * 0.8)

# Scale the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# create training data set
train_data = scaled_data[:training_data_length, :]
x_train = [] #independent training variabls
y_train = [] #dependent training variables

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])

#x_train contains 60 values which are used 
#  to train the 61th value which is stored in y_train variable

#convert x_train and y_train to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

#Reshape the data, as LSTM expects input to be 3D but our current dataset is 2D
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Now we will build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values
test_data = scaled_data[training_data_length - 60:, :]

x_test = []
y_test = dataset[training_data_length: , :]

for i in range(60, len(test_data)):
  x_test.append(test_data[i - 60: i, 0])

#Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape again to make it 3D
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) #unscaling values

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)

# Plot the data
train = data[:training_data_length]
actual_val = data[training_data_length:]
actual_val['Predictions'] = predictions - rmse

#Plot the graph with predicted and actual values
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(actual_val[['Close', 'Predictions']])
plt.legend(['Training Data', 'Actual Value', 'Predicted Value'], loc = 'lower right')

# Show the actual and predicted prices
actual_val

# Get the price of next day
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2022-10-15')
new_df = apple_quote.filter(['Close'])

last_60_days = new_df[-60:].values #convert to np array
last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
X_test.append(last_60_days_scaled)

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_scale_price = model.predict(X_test)

pred_price = scaler.inverse_transform(pred_scale_price)

print(pred_price)

today_price = web.DataReader('AAPL', data_source='yahoo', start='2022-10-17', end='2022-10-17')
print(today_price['Close'])