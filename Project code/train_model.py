import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Function to calculate Supply Readiness based on the Inventory to Demand Ratio
def calculate_supply_readiness(inventory_to_demand_ratio):
    # Assuming Supply Readiness is ready if ratio > 0.8
    return (inventory_to_demand_ratio > 1.0).astype(int)

# Load dataset
dataset = pd.read_csv('new_dataset.csv')

# Feature Engineering: Only Demand_Forecast and Current_Inventory are input by the user
dataset['Inventory_to_Demand_Ratio'] = dataset['Current_Inventory'] / dataset['Demand_Forecast']
dataset['Inventory_Demand_Difference'] = dataset['Current_Inventory'] - dataset['Demand_Forecast']

# Automatically calculate Supply Readiness based on Inventory to Demand Ratio
dataset['Supply_Ready'] = calculate_supply_readiness(dataset['Inventory_to_Demand_Ratio'])

# Define features and target
features = ['Demand_Forecast', 'Current_Inventory', 'Inventory_to_Demand_Ratio', 'Inventory_Demand_Difference', 'Supply_Ready']
target = 'Revenue_Status'

X = dataset[features]
y = dataset[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameter Tuning with Grid Search
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best estimator from GridSearchCV
best_rf = grid_search.best_estimator_

# Evaluate the model using cross-validation
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores) * 2:.2f})")

# Fit the final model on the entire training set
best_rf.fit(X_train, y_train)

# Save the trained model
joblib.dump(best_rf, 'revenue_status_model_selected_features_v2.pkl')

# Test the model on the test set
y_pred = best_rf.predict(X_test)
print("Test Set Accuracy:", np.mean(y_pred == y_test))

# Display feature importances
importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))
