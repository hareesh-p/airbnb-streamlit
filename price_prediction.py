import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

def train_price_model(listings, top_n=10):
    """ Train multiple models with feature selection, hyperparameter tuning, and outlier removal """

    # Define numerical and categorical features
    numerical_features = ["accommodates", "bedrooms", "bathrooms", "number_of_reviews", "review_scores_rating", "review_scores_value"]
    categorical_features = ["room_type", "neighbourhood_cleansed", "property_type", "instant_bookable", "host_is_superhost"]

    # Extract `bathrooms` from `bathrooms_text` if missing
    if "bathrooms" in listings.columns and listings["bathrooms"].isna().sum() > 0:
        if "bathrooms_text" in listings.columns:
            listings["bathrooms"] = listings["bathrooms"].fillna(
                listings["bathrooms_text"].str.extract(r"(\d+(\.\d+)?)")[0].astype(float)
            )

    # Remove outliers using Interquartile Range (IQR)
    Q1 = listings["price"].quantile(0.25)
    Q3 = listings["price"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    listings = listings[(listings["price"] >= lower_bound) & (listings["price"] <= upper_bound)]

    # Keep only rows with complete data
    listings = listings.dropna(subset=numerical_features + categorical_features + ["price"])

    # Keep only features that exist in the dataset
    available_numerical = [col for col in numerical_features if col in listings.columns]
    available_categorical = [col for col in categorical_features if col in listings.columns]

    # Apply one-hot encoding to categorical features
    if available_categorical:
        listings = pd.get_dummies(listings, columns=available_categorical, drop_first=True)

    # Automatically detect encoded categorical columns
    encoded_features = [col for col in listings.columns if col.startswith(tuple(categorical_features))]

    # Final feature set
    all_features = available_numerical + encoded_features

    X = listings[all_features]
    y = listings["price"]

    # Normalize numerical features
    scaler = StandardScaler()
    X[available_numerical] = scaler.fit_transform(X[available_numerical])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest for feature importance selection
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42)
    rf_model.fit(X_train, y_train)

    # Get feature importance
    feature_importance = pd.DataFrame({"Feature": all_features, "Importance": rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    # **Ensure essential features are always included**
    essential_features = ["bathrooms"]
    selected_features = list(set(feature_importance["Feature"].head(top_n).tolist() + essential_features))

    # Identify categorical variables among selected features
    selected_categorical_features = [col for col in categorical_features if any(f.startswith(col + "_") for f in selected_features)]

    # Ensure all categories of selected categorical variables are included
    for cat in selected_categorical_features:
        selected_features.extend([col for col in encoded_features if col.startswith(cat)])

    # Save the exact feature order
    selected_features = sorted(set(selected_features))  # Ensure order consistency

    # Retrain models using only selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # 📌 Hyperparameter Tuning for Random Forest
    rf_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, n_iter=10, cv=3, scoring="r2", n_jobs=-1, random_state=42)
    rf_search.fit(X_train_selected, y_train)
    tuned_rf = rf_search.best_estimator_

    # 📌 Hyperparameter Tuning for XGBoost
    xgb_param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9]
    }
    xgb_search = RandomizedSearchCV(XGBRegressor(random_state=42), xgb_param_grid, n_iter=10, cv=3, scoring="r2", n_jobs=-1, random_state=42)
    xgb_search.fit(X_train_selected, y_train)
    tuned_xgb = xgb_search.best_estimator_

    # 📌 Train and Compare Tuned Models
    models = {
        "Random Forest (Tuned)": tuned_rf,
        "XGBoost (Tuned)": tuned_xgb
    }

    model_performance = {}
    best_model = None
    best_r2 = float('-inf')

    for model_name, model in models.items():
        # Train model
        model.fit(X_train_selected, y_train)

        # Predict
        y_pred = model.predict(X_test_selected)

        # Calculate performance metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Store performance
        model_performance[model_name] = {"R² Score": r2, "RMSE": rmse}

        # Track the best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    # Convert performance dictionary to DataFrame
    performance_df = pd.DataFrame(model_performance).T

    return best_model, performance_df, selected_features, feature_importance, scaler, available_numerical
