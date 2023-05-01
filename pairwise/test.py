import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from itertools import combinations

# Load data
data = pd.read_csv('../expert_data.csv')

# Create all pairwise combinations
combs = list(combinations(data['expert_id'], 2))

# Compute relevance scores for pairwise combinations
pairs = pd.DataFrame(combs, columns=['expert_1', 'expert_2'])
pairs['relevance_score'] = np.zeros(pairs.shape[0])
for i in range(pairs.shape[0]):
    e1 = data[data['expert_id'] == pairs.loc[i, 'expert_1']].iloc[:, 1:-1].values
    e2 = data[data['expert_id'] == pairs.loc[i, 'expert_2']].iloc[:, 1:-1].values
    pairs.loc[i, 'relevance_score'] = int(model.predict(e1 - e2) > 0.5)

# Fit logistic regression model to pairwise data
X = pairs.drop(['expert_1', 'expert_2', 'relevance_score'], axis=1)
y = pairs['relevance_score']
model = LogisticRegression()
model.fit(X, y)

# Predict relevance scores for each expert
experts = data.drop(['expert_id', 'relevance_score'], axis=1)
rankings = []
for i in range(experts.shape[0]):
    e1 = experts.iloc[i, :].values
    e2 = experts.iloc[:, :].values
    relevance_scores = model.predict_proba(e2 - e1)[:, 1]
    ranking = np.sum(relevance_scores > relevance_scores[i]) + 1
    rankings.append(ranking)

# Add predicted relevance scores to original data frame
data['predicted_score'] = rankings

# Sort experts by predicted relevance score
ranked_experts = data.sort_values('predicted_score', ascending=True)

# Print ranked list of experts
expert_ranking = ranked_experts[['expert_id', 'predicted_score']].reset_index(drop=True)
print('Ranking of experts:')
print(expert_ranking)
