import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential,load_model
from keras.layers import LSTM, SimpleRNN, GRU, Dense 


def linear_regression():
    file_path = os.path.join(os.path.dirname(__file__), 'new_df.csv')
    new_df = pd.read_csv('/Users/moulalishariff/Electric/new_df.csv')
    X_linear = new_df[['temperature', 'var1', 'pressure', 'windspeed','year','month','day','day_of_week','hour']]
    y_linear = new_df['electricity_consumption']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)
    
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Evaluate the model
    mea = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        'Mean Absolute Error': mea,
        'Mean Square Error': mse,
        'Root Mean Square Error': rmse,
        'R2 Score': r2
    }
    # Evaluate the model
    accuracy = r2_score(y_test, y_pred)
    
    return accuracy, metrics


def rnn():
    file_path = os.path.join(os.path.dirname(__file__), 'new_df.csv')
    new_df = pd.read_csv(file_path)
    Xrnn = new_df[['temperature', 'var1', 'pressure', 'windspeed','year','month','day','day_of_week','hour']].values
    yrnn = new_df['electricity_consumption'].values

    sequence_length = 1  # Each data point is treated independently

    # Reshape the input data into sequences
    X_seq = []
    y_seq = []

    for i in range(len(Xrnn) - sequence_length):
        X_seq.append(Xrnn[i:i + sequence_length])
        y_seq.append(yrnn[i + sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    X_trainrnn, X_testrnn, y_trainrnn, y_testrnn = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Define the RNN model
    RNNmodel = Sequential()
    RNNmodel.add(SimpleRNN(50, activation='relu', input_shape=(X_trainrnn.shape[1], X_trainrnn.shape[2])))
    RNNmodel.add(Dense(1))  # Output layer with 1 neuron for regression
    RNNmodel.compile(optimizer='adam', loss='mean_squared_error')
    
    RNNmodel.fit(X_trainrnn, y_trainrnn, epochs=10, batch_size=32, verbose=1)

    y_predRNN = RNNmodel.predict(X_testrnn)
    
    # Evaluate the model
    mea = mean_absolute_error(y_testrnn, y_predRNN)
    mse = mean_squared_error(y_testrnn, y_predRNN)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_testrnn, y_predRNN)
    metrics = {
        'Mean Absolute Error': mea,
        'Mean Square Error': mse,
        'Root Mean Square Error': rmse,
        'R2 Score': r2
    }
    # Evaluate the model
    accuracy = r2_score(y_testrnn, y_predRNN)
    
    return accuracy, metrics


def lstm():
    file_path = os.path.join(os.path.dirname(__file__), 'new_df.csv')
    new_df = pd.read_csv('/Users/moulalishariff/Electric/new_df.csv')
    Xlstm = new_df[['temperature', 'var1', 'pressure', 'windspeed', 'year', 'month', 'day', 'day_of_week', 'hour']].values
    ylstm = new_df['electricity_consumption'].values
    
    sequence_length = 1  # Each data point is treated independently

    # Reshape the input data into sequences
    X_seq = []
    y_seq = []

    for i in range(len(Xlstm) - sequence_length):
        X_seq.append(Xlstm[i:i + sequence_length])
        y_seq.append(ylstm[i + sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Split the data into training and testing sets
    X_trainlstm, X_testlstm, y_trainlstm, y_testlstm = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_trainlstm.shape[1], X_trainlstm.shape[2])))
    model.add(Dense(1))  # Output layer with 1 neuron for regression
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the LSTM model
    model.fit(X_trainlstm, y_trainlstm, epochs=10, batch_size=32, verbose=1)

    # Evaluate the model on the test set
    y_predLSTM = model.predict(X_testlstm)
    # Evaluate the model
    mea = mean_absolute_error(y_testlstm, y_predLSTM)
    mse = mean_squared_error(y_testlstm, y_predLSTM)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_testlstm, y_predLSTM)
    r2 = abs(r2)
    error_rate = 1-r2
    metrics = {
        'Mean Absolute Error': mea,
        'Mean Square Error': mse,
        'Root Mean Square Error': rmse,
        'R2 Score': r2,
        'Error Rate': abs(error_rate)
    }
    # Evaluate the model
    accuracy = r2_score(y_testlstm, y_predLSTM)
    
    return accuracy, metrics

def stacked():
    file_path = os.path.join(os.path.dirname(__file__), 'new_df.csv')
    new_df = pd.read_csv('/Users/moulalishariff/Electric/new_df.csv')
    XSlstm = new_df[['temperature', 'var1', 'pressure', 'windspeed', 'year', 'month', 'day', 'day_of_week', 'hour']].values
    ySlstm = new_df['electricity_consumption'].values

    # Define sequence length (number of previous time steps to consider)
    sequence_length = 1  # Each data point is treated independently

    # Reshape the input data into sequences
    X_seq = []
    y_seq = []

    for i in range(len(XSlstm) - sequence_length):
        X_seq.append(XSlstm[i:i + sequence_length])
        y_seq.append(ySlstm[i + sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Split the data into training and testing sets
    X_trainSlstm, X_testSlstm, y_trainSlstm, y_testSlstm = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    

    # Define the stacked LSTM model
    SLSTMmodel = Sequential()

    # First LSTM layer
    SLSTMmodel.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_trainSlstm.shape[1], X_trainSlstm.shape[2])))

    # Second LSTM layer (stacked)
    SLSTMmodel.add(LSTM(50, activation='relu', return_sequences=True))

    # Third LSTM layer (stacked)
    SLSTMmodel.add(LSTM(50, activation='relu'))

    # Output layer with 1 neuron for regression
    SLSTMmodel.add(Dense(1))

    SLSTMmodel.compile(optimizer='adam', loss='mean_squared_error')

    # Train the LSTM model
    SLSTMmodel.fit(X_trainSlstm, y_trainSlstm, epochs=10, batch_size=32, verbose=1)

    # Evaluate the model on the test set
    y_predSLSTM = SLSTMmodel.predict(X_testSlstm)

    mea = mean_absolute_error(y_testSlstm, y_predSLSTM)
    mse = mean_squared_error(y_testSlstm, y_predSLSTM)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_testSlstm, y_predSLSTM)
    metrics = {
        'Mean Absolute Error': mea,
        'Mean Square Error': mse,
        'Root Mean Square Error': rmse,
        'R2 Score': r2
    }
    # Evaluate the model
    accuracy = r2_score(y_testSlstm, y_predSLSTM)
    
    return accuracy, metrics


def gru():
    file_path = os.path.join(os.path.dirname(__file__), 'new_df.csv')
    new_df = pd.read_csv('/Users/moulalishariff/Electric/new_df.csv')
    Xgru = new_df[['temperature', 'var1', 'pressure', 'windspeed','year','month','day','day_of_week','hour']].values
    ygru = new_df['electricity_consumption'].values

    # Define sequence length (number of previous time steps to consider)
    sequence_length = 1  # Each data point is treated independently

    # Reshape the input data into sequences
    X_seq = []
    y_seq = []

    for i in range(len(Xgru) - sequence_length):
        X_seq.append(Xgru[i:i + sequence_length])
        y_seq.append(ygru[i + sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    X_traingru, X_testgru, y_traingru, y_testgru = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Define the single-layer GRU model
    GRUmodel = Sequential()

    # Single GRU layer
    GRUmodel.add(GRU(50, activation='relu', input_shape=(X_traingru.shape[1], X_traingru.shape[2])))

    # Output layer with 1 neuron for regression
    GRUmodel.add(Dense(1))

    # Compile the model
    GRUmodel.compile(optimizer='adam', loss='mean_squared_error')

   # Train the LSTM model
    GRUmodel.fit(X_traingru, y_traingru, epochs=10, batch_size=32, verbose=1)

    # Evaluate the model on the test set
    y_predGRU = GRUmodel.predict(X_testgru)
    
    # Make predictions on the test set
    y_pred = GRUmodel.predict(X_testgru)
    # Evaluate the model
    mea = mean_absolute_error(y_testgru, y_pred)
    mse = mean_squared_error(y_testgru, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_testgru, y_pred)
    metrics = {
        'Mean Absolute Error': mea,
        'Mean Square Error': mse,
        'Root Mean Square Error': rmse,
        'R2 Score': r2
    }
    # Evaluate the model
    accuracy = r2_score(y_testgru, y_pred)
    
    return accuracy, metrics