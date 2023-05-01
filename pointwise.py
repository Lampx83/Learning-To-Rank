import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from catboost import CatBoostRegressor

# 1 Load data
data = pd.read_csv('expert_data.csv')
# 2 Split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

X = data.drop(['expert_id', 'relevance_score'], axis=1)
y = data['relevance_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# 3 Prepare data for training
train_features = train_data.drop(['expert_id', 'relevance_score'], axis=1)
train_target = train_data['relevance_score']

# 4 Evaluate models
models = []
models.append((1940, 'LinearRegression', LinearRegression()))
models.append((1970, 'KNeighborsRegressor', KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30)))
models.append((1980, 'DecisionTreeRegressor', DecisionTreeRegressor(max_depth=3, random_state=42)))
models.append((1995, 'Support Vector Regression,', SVR(kernel='linear', C=1.0, epsilon=0.1)))
models.append((1995, 'AdaBoostRegressor', AdaBoostRegressor()))
models.append((1995, 'RandomForestRegressor', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)))
models.append((1999, 'GradientBoostingRegressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)))
models.append((2000, 'BayesianRidge', BayesianRidge()))
models.append((2006, 'MLPRegressor', MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.0001, max_iter=1000, random_state=42)))
models.append((2014, 'XGBRegressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000))),
models.append((2017, 'CatBoostRegressor', CatBoostRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_seed=42)))
models.append((2021, 'TensorFlowRegressor', tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(1)])))
models.append((2022, 'LightGBM', lgb.LGBMRegressor()))


results = []
names = []

for year, name, model in models:
    if name == 'TensorFlowRegressor':
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_features, train_target)
    test_features = test_data.drop(['expert_id', 'relevance_score'], axis=1)
    test_target = test_data['relevance_score']
    test_predictions = model.predict(test_features)
    mse = mean_squared_error(test_target, test_predictions)
    mae = mean_absolute_error(test_target, test_predictions)
    r2 = r2_score(test_target, test_predictions)
    results.append((year, mse, mae, r2))
    names.append(name)

df_results = pd.DataFrame(results, columns=['year', 'MSE', 'MAE', 'R2 Score'], index=names)
# df_results = df_results.sort_values(by=['MSE'])
print("##############################################################")
print(df_results)
print("MSE: Mean Squared Error, MAE: Mean Absolute Error")
print("##############################################################")
# 6 Evaluate a model
alg = 0
# alg = int(input("Model:"))
model = models[alg][2]
test_features = test_data.drop(['expert_id', 'relevance_score'], axis=1)
test_target = test_data['relevance_score']
test_predictions = model.predict(test_features)
print('Mean Squared Error   :', mean_squared_error(test_target, test_predictions))
print('Mean Absolute Error  :', mean_absolute_error(test_target, test_predictions))
print('R2 Score             :', r2_score(test_target, test_predictions))

# 7 Ranking of new expert
new_query = pd.DataFrame({
    'education_level': [3],
    'years_of_experience': [6],
    'projects': [15],
    'awards': [2],
})
expert_scores = model.predict(pd.DataFrame(new_query))
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
print("Done")
# Compare models
# Mean Squared Error (MSE): Đây là một chỉ số đánh giá khoảng cách giữa các giá trị dự đoán và giá trị thực tế. Giá trị MSE càng thấp thì mô hình dự đoán càng tốt.
# Mean Absolute Error (MAE): Đây cũng là một chỉ số đánh giá khoảng cách giữa các giá trị dự đoán và giá trị thực tế. Tuy nhiên, MAE đo khoảng cách tuyệt đối thay vì khoảng cách bình phương. Giá trị MAE càng thấp thì mô hình dự đoán càng tốt.
# R2 Score: Đây là một chỉ số đánh giá khác đo lường khả năng giải thích của mô hình. Giá trị R2 càng gần 1 thì mô hình càng tốt.
