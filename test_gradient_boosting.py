import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('expert_data.csv')

# Split data into training and testing sets
X = data.drop(['expert_id', 'relevance_score'], axis=1)
y = data['relevance_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate model on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error on test set:', mse)
# New query features
new_query = pd.DataFrame({
    'education_level': [2],
    'years_of_experience': [7],
    'projects': [12],
    'awards': [1]
})
expert_scores = model.predict(pd.DataFrame(new_query))
print("Ranking of experts for the new query",expert_scores)

# Predict relevance scores for each expert
experts = data.drop(['expert_id', 'relevance_score'], axis=1)
rankings = model.predict(experts)

# Add predicted relevance scores to original data frame
data['predicted_score'] = rankings

# Sort experts by predicted relevance score
ranked_experts = data.sort_values('predicted_score', ascending=False)

# Print ranked list of experts for new query
query_ranking = ranked_experts[['expert_id', 'predicted_score']].reset_index(drop=True)
print('Ranking of experts for the new query:')
print(query_ranking)
