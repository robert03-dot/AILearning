import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb

# Classification problem
X_class, y_class = make_classification(n_samples=1000, n_features=20)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2)

# XGBoost Classifier
clf = xgb.XGBClassifier()
clf.fit(X_train_class, y_train_class)

# Predicting and evaluating classification
y_pred_class = clf.predict(X_test_class)
accuracy_class = accuracy_score(y_test_class, y_pred_class)
print("Initial Classification Accuracy:", accuracy_class)

# Regression problem
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2)

# Linear Regression
regressor = LinearRegression()
regressor.fit(X_train_reg, y_train_reg)

# Predicting and evaluating regression
y_pred_reg = regressor.predict(X_test_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
print("Regression RMSE:", rmse)

# Hyperparameter tuning with GridSearchCV for XGBoost classifier
params_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 4, 5],
    "n_estimators": [50, 100, 200]
}

clf = xgb.XGBClassifier()
grad_clf = GridSearchCV(clf, params_grid, cv=3, scoring="accuracy")
best_clf = grad_clf.fit(X_train_class, y_train_class)

# Printing the best parameters and model
print("Best Parameters from GridSearchCV:", best_clf.best_params_)

# Predicting and evaluating after GridSearchCV
y_pred_best = grad_clf.predict(X_test_class)
accuracy_best = accuracy_score(y_test_class, y_pred_best)
print("Tuned Classification Accuracy:", accuracy_best)

params_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 4, 5],
    "n_estimators": [50, 100, 200]
}
regressor = xgb.XGBRegressor()
grid_regressor = GridSearchCV(regressor, params_grid, cv=3, scoring="neg_mean_squared_error")
print("Best Parameters from GridSearchCV:", grid_regressor)
grid_regressor.fit(X_train_reg, y_train_reg)
best_regressor = grid_regressor.best_estimator_
y_pred_reg = grid_regressor.predict(X_test_reg)
print(y_pred_reg)

rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
print("Tuned RMSE:", rmse)