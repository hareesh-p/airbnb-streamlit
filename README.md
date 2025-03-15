# Airbnb Market Analysis - Streamlit App

This Streamlit application provides an interactive dashboard for analyzing Airbnb listings across multiple cities. Users can:
- Explore Airbnb listings data through visualizations.
- Analyze price distribution and trends based on different filters.
- Predict Airbnb rental prices using advanced Machine Learning models.
- Perform sentiment analysis on guest reviews.
- Utilize geo-visualizations with property coordinates.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Data Sources & Structure](#data-sources--structure)  
4. [Data Dictionary](#data-dictionary)  
5. [Installation & Setup](#installation--setup)  
6. [App Usage](#app-usage)  
7. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
8. [Price Analysis](#price-analysis)  
9. [Price Prediction](#price-prediction)  
10. [Sentiment Analysis](#sentiment-analysis)  
11. [Geo-Visualizations](#geo-visualizations)  
12. [Model Training & Feature Selection](#model-training--feature-selection)  
13. [License & Acknowledgments](#license--acknowledgments)  

---

## Project Overview
This Streamlit-based dashboard helps users explore and analyze Airbnb listings.  
It supports five cities: Bristol, Geneva, Hague, Hong Kong, and Singapore.  

The app enables:  
- Data exploration with summary statistics and missing values handling.  
- Airbnb price trend analysis through visualizations.  
- Machine Learning-powered price prediction using handpicked features based on feature importance.  
- Sentiment analysis on guest reviews.  
- Geo-visualizations for property mapping.  

---

## Features
- Multi-city support: Select Airbnb data from 5 cities.
- Interactive price analysis: Analyze pricing trends across neighborhoods.
- AI-powered price prediction: Uses Random Forest & XGBoost models.
- Sentiment analysis: Understand customer satisfaction from guest reviews.
- Geo-visualizations: Scatter Mapbox & Choropleth Maps provide location-based insights.

---

## Data Sources & Structure
The app uses detailed Airbnb datasets from [Inside Airbnb](https://insideairbnb.com/).  
Each city's data is stored as:

```
/data/{city}/listings.csv
/data/{city}/reviews.csv
/data/{city}/neighbourhoods.geojson
```

- **listings.csv** – Contains detailed Airbnb listing information.
- **reviews.csv** – Stores guest reviews for sentiment analysis.
- **neighbourhoods.geojson** – Geo-data for mapping Airbnb listings.

---

## Data Dictionary
[Official Airbnb Data Dictionary](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit?usp=sharing)  

Key attributes:
- `price`: Nightly rental price of an Airbnb listing.
- `room_type`: Type of Airbnb accommodation (Entire home, Private room, etc.).
- `neighbourhood_cleansed`: Cleaned version of the neighborhood name.
- `latitude` & `longitude`: Geo-coordinates for mapping.
- `number_of_reviews`: Count of guest reviews.
- `review_scores_rating`: Overall rating of the listing.
- **Predictors used for price prediction**:
  - `accommodates`: Number of people the listing accommodates.
  - `bedrooms`: Number of bedrooms in the listing.
  - `bathrooms`: Number of bathrooms in the listing.
  - `number_of_reviews`: Count of guest reviews.
  - `review_scores_rating`: Overall rating of the listing.
  - `review_scores_value`: Value rating of the listing.
  - `room_type`: Type of Airbnb accommodation (Entire home, Private room, etc.).
  - `neighbourhood_cleansed`: Cleaned version of the neighborhood name.
  - `property_type`: Type of property (Apartment, House, etc.).
  - `instant_bookable`: Whether the listing is instantly bookable.
  - `host_is_superhost`: Whether the host is a superhost.

---

## Installation & Setup
To run this project on your local machine:

### 1. Clone the Repository
```sh
git clone https://github.com/hareesh-p/airbnb-streamlit.git
cd airbnb-streamlit
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```sh
streamlit run app.py
```

---

## App Usage
The app has four main sections:

### Exploratory Data Analysis (EDA)
- Provides an overview of the dataset.
- Displays missing values and summary statistics.
- Visualizes price distribution and room type proportions.

### Price Analysis
- **Overall Price Distribution**: Analyzes the price distribution across all listings in the selected city.
- **Neighborhood Price Distribution**: Analyzes price trends within specific neighborhoods.
- Maps & Visualizations:
  - Scatter Map: Displays individual property prices.
  - Choropleth Map: Shows average price per neighborhood.

### Price Prediction
- Uses Random Forest & XGBoost models to predict Airbnb prices.
- Handpicked the best features based on feature importance for prediction.
- Compares models using R² Score and RMSE.
- Users input listing details to get a predicted price.

### Sentiment Analysis
- Generates a Word Cloud from guest reviews.
- Uses VADER Sentiment Analysis to classify reviews.
- Displays sentiment distribution across listings.

---

## Geo-Visualizations
- Scatter Mapbox – Maps individual Airbnb properties.
- Choropleth Map – Displays average price per neighborhood.
- Uses GeoJSON files for accurate location-based insights.

---

## Model Training & Feature Selection
1. Handles missing values, outliers, and normalizes numerical data.
2. Uses Random Forest & XGBoost for predictions.
3. Feature Selection:
   - Handpicks top important features based on feature importance.
   - Ensures categorical variables (e.g., `property_type`) are included.
   - Removes outliers using Interquartile Range (IQR).
4. Hyperparameter Tuning:
   - Optimizes `n_estimators`, `max_depth`, and `learning_rate` for XGBoost.
   - Optimizes `n_estimators`, `max_depth`, and `min_samples_split` for Random Forest using Randomized Search.

---

## License & Acknowledgments
This project is for educational and analytical purposes only.  
Data Source: [Inside Airbnb](https://insideairbnb.com/)  
Contributors: Sumit Kumar(EMBADTA24013), Hareesh P(EMBADTA24015), Abhishek Mishra(EMBADTA24016)