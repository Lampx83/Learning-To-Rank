import lightgbm as lgb
import numpy as np
import pandas as pd

# Load data
data = pd.DataFrame({
    'expert_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'education_level': [2, 1, 2, 1, 3, 1, 2, 1, 3, 2],
    'years_of_experience': [5, 8, 3, 12, 2, 15, 7, 10, 1, 4],
    'projects': [10, 20, 5, 50, 2, 30, 15, 25, 0, 8],
    'awards': [0, 2, 1, 5, 0, 3, 1, 2, 0, 0],
    'relevance_score': [0.7, 0.9, 0.6, 1.0, 0.5, 0.95, 0.8, 0.85, 0.4, 0.65],
    'query': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], # Query information
})

# Split data into training and validation sets
train_size = int(0.8 * len(data))
train_data = lgb.Dataset(data=data[:train_size], label=np.zeros(train_size), group=data[:train_size]['query'])
valid_data = lgb.Dataset(data=data[train_size:], label=np.zeros(len(data) - train_size), group=data[train_size:]['query'])

# Set ranking task parameters
params = {
    'objective': 'rank:ndcg',
    'learning_rate': 0.1,
    'max_depth': 3,
    'num_leaves': 31,
    'verbose': 1,
}

# Train the model
model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[valid_data])
