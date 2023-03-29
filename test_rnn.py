import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('expert_data.csv')

# Split data into training and testing sets
X = data.drop(['expert_id', 'relevance_score'], axis=1)
y = data['relevance_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for use in recurrent neural network
X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])

# Define model architecture
input_layer = Input(shape=X_train.shape[1:])
lstm1 = LSTM(64, activation='relu')(input_layer)
dropout1 = Dropout(0.2)(lstm1)
output_layer = Dense(1, activation='linear')(dropout1)
model = Model(inputs=input_layer, outputs=output_layer)
optimizer = Adam(learning_rate=0.01)
model.compile(loss='mse', optimizer=optimizer)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Evaluate model on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error on test set:', mse)

# Use model to rank experts for a new query
new_query = np.array([[2, 7, 12, 1]])
rankings = model.predict(new_query)
print('Ranking of experts for the new query:',rankings)

new_query = new_query.reshape(1, 1, new_query.shape[1])
rankings = np.squeeze(rankings)
data['predicted_score'] = rankings
ranked_experts = data.sort_values('predicted_score', ascending=False)[['expert_id', 'predicted_score']].reset_index(drop=True)

print(ranked_experts)
