import pandas as pd
import os

def load_airbnb_data(city, data_type="Detailed"):
    """ Load Airbnb data for the selected city. 
        data_type: "Detailed" (default) loads full dataset, "Summary" loads smaller dataset.
    """
    path = f"data/{city}/"

    # Select file names based on user choice
    if data_type == "Detailed":
        listings_file = "listings.csv"
        reviews_file = "reviews.csv"
    else:  # "Summary"
        listings_file = "listings-summary.csv"
        reviews_file = "reviews-summary.csv"

    # Load datasets
    try:
        listings = pd.read_csv(os.path.join(path, listings_file))
        reviews = pd.read_csv(os.path.join(path, reviews_file))
        neighbourhoods = pd.read_csv(os.path.join(path, "neighbourhoods.csv"))

        # Ensure price column is numeric (important for ML)
        if "price" in listings.columns:
            listings["price"] = listings["price"].replace('[\$,]', '', regex=True).astype(float)

        return listings, reviews, neighbourhoods
    except Exception as e:
        raise ValueError(f"Error loading data for {city} - {data_type}: {e}")
