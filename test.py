import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1 Load data
data = pd.read_csv('expert_data.csv')

# 2 Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 3 Prepare data for training
train_features = train_data.drop(['expert_id', 'relevance_score'], axis=1)
train_target = train_data['relevance_score']

# 4 Train model
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [3],  # optimize for NDCG@3
    'learning_rate': 0.1,
    'max_depth': 3,
    'num_leaves': 31,
    'random_state': 42,
    'verbose': -1,
}

train_dataset = lgb.Dataset(train_features, label=train_target)
model = lgb.train(params, train_dataset, num_boost_round=100)

# 5 Evaluate model on test set
test_features = test_data.drop(['expert_id', 'relevance_score'], axis=1)
test_target = test_data['relevance_score']
test_predictions = model.predict(test_features)
mse = mean_squared_error(test_target, test_predictions)
print('Mean Squared Error:', mse)

# 6 Ranking of new expert
new_query = pd.DataFrame({
    'education_level': [3],
    'years_of_experience': [6],
    'projects': [15],
    'awards': [2],
})
expert_scores = model.predict(new_query)
print("Ranking of new expert:", expert_scores)

# 7 Predict relevance scores for each expert
experts = data.drop(['expert_id', 'relevance_score'], axis=1)
rankings = model.predict(experts)
# Add predicted relevance scores to original data frame
data['predicted_score'] = rankings
ranked_experts = data.sort_values('predicted_score', ascending=False)
# Sort experts by predicted relevance score
query_ranking = ranked_experts[['expert_id', 'predicted_score']].reset_index(drop=True)
print('Sort experts by predicted relevance score:')
print(query_ranking)
