import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def train_price_model(listings):
    """ Train a regression model to predict Airbnb prices """
    
    # Selecting important features for detailed dataset
    features = ["accommodates", "bedrooms", "bathrooms", "number_of_reviews"]

    # Add extra features if they exist in the detailed dataset
    #additional_features = ["square_feet", "host_is_superhost", "availability_365", "review_scores_rating"]
    #for feature in additional_features:
    #    if feature in listings.columns:
    #        features.append(feature)

    # Ensure necessary features are available
    listings = listings.dropna(subset=features + ["price"])

    X = listings[features]
    y = listings["price"]

    # Encode categorical variables if present
    if "host_is_superhost" in X.columns:
        X["host_is_superhost"] = X["host_is_superhost"].map({"t": 1, "f": 0})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return model, r2
