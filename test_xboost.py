import pandas as pd
from sklearn.model_selection import train_test_split

# Read data from CSV file
data = pd.read_csv('expert_data.csv')

# Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare data for training
train_features = train_data.drop(['expert_id', 'relevance_score'], axis=1)
train_target = train_data['relevance_score']

test_features = test_data.drop(['expert_id', 'relevance_score'], axis=1)
test_target = test_data['relevance_score']

import xgboost as xgb

# Create XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)

# Train the model
model.fit(train_features, train_target)

# New query features
new_query = pd.DataFrame({
    'education_level': [3],
    'years_of_experience': [6],
    'projects': [15],
    'awards': [2]
})
expert_scores = model.predict(pd.DataFrame(new_query))
print("Ranking of experts for the new query", expert_scores)

# Predict relevance scores for each expert
experts = data.drop(['expert_id', 'relevance_score'], axis=1)
rankings = model.predict(experts)

# Sort experts by predicted relevance score
ranked_experts = data.copy()
ranked_experts['rank'] = rankings
ranked_experts = ranked_experts.sort_values('rank', ascending=False)

# Print ranked list of experts
print(ranked_experts[['expert_id', 'rank']])


# import pandas as pd
#
# # Read data from CSV file
# train_data = pd.read_csv('expert_data_train.csv')
#
# # Prepare data for training
# train_features = train_data.drop(['expert_id', 'relevance_score'], axis=1)
# train_target = train_data['relevance_score']
#
# import xgboost as xgb
#
# # Create XGBoost model
# model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
#
# # Train the model
# model.fit(train_features, train_target)
#
# # New query features
# new_query = pd.DataFrame({
#     'education_level': [3],
#     'years_of_experience': [6],
#     'projects': [15],
#     'awards': [2]
# })
# expert_scores = model.predict(pd.DataFrame(new_query))
# print("Ranking of experts for the new query", expert_scores)
#
# # Read data from CSV file
# test_data = pd.read_csv('expert_data_test.csv')
#
# # Prepare data for training
# train_features = train_data.drop(['expert_id', 'relevance_score'], axis=1)
# train_target = train_data['relevance_score']
#
# # Predict relevance scores for each expert
# experts = test_data.drop(['expert_id', 'relevance_score'], axis=1)
# rankings = model.predict(experts)
#
# # Sort experts by predicted relevance score
# ranked_experts = test_data.copy()
# ranked_experts['rank'] = rankings
# ranked_experts = ranked_experts.sort_values('rank', ascending=False)
#
# # Print ranked list of experts
# print(ranked_experts[['expert_id', 'rank']])



