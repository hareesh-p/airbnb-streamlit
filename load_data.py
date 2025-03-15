import pandas as pd
import os

def load_airbnb_data(city):
    """ Load Airbnb data for the selected city (Only detailed dataset is used). """
    path = f"data/{city}/"

    # Load datasets
    try:
        listings = pd.read_csv(os.path.join(path, "listings.csv"))
        reviews = pd.read_csv(os.path.join(path, "reviews.csv"))
        neighbourhoods = pd.read_csv(os.path.join(path, "neighbourhoods.csv"))

        # Ensure price column is numeric
        if "price" in listings.columns:
            listings["price"] = listings["price"].replace('[\$,]', '', regex=True).astype(float)

        return listings, reviews, neighbourhoods
    except Exception as e:
        raise ValueError(f"Error loading data for {city}: {e}")
