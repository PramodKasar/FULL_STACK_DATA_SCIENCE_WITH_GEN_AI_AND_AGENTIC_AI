import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,SGDRegressor, HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline   #data Leakages
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
#import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

#Load Dataset
data = pd.read_csv(r"C:\Users\Lenovo\Desktop\FULL STACK DATA SCIENCE BY PRAKASH SENAPATI SIR\Machine Learning\USA Housing Full Project\USA_Housing.csv")
# Preprocessing
X = data.drop(['Price', 'Address'], axis=1)
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#define Models
models = {
    'LinearRegression' : LinearRegression(),
    'RobustRegression' : HuberRegressor(),
    'RidgeRegression' : Ridge(),
    'LassoRegression' : Lasso(),
    'ElasticNetRegression' : ElasticNet(),
    'PolynomialRegression' : Pipeline([
        ('poly', PolynomialFeatures(degree=4)),
        ('linear', LinearRegression())
    ]),
    'SGDRegressor' : SGDRegressor(),
    'ANNRegressor' : MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
    'RandomForestRegressor' : RandomForestRegressor(),
    'SVM' : SVR(),
    #'LGBM': lgb.LGBMRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'KNN': KNeighborsRegressor()
}

# Train and evaluate models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    })
    with open(f"{name}.pkl", 'wb') as f:
        pickle.dump(model, f)

#concat results into a DataFrame and save to csv
results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)
print("Models have been trained and saved as pickle files. Evaluation results have been saved to model_evaluation_results.csv.")
