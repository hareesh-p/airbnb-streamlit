# 🏡 Airbnb Market Analysis - Streamlit App

This **Streamlit application** provides an **interactive dashboard** for analyzing Airbnb listings across multiple cities. Users can:
- Explore **Airbnb listings data** through visualizations.
- Analyze **price distribution and trends** based on different filters.
- Predict Airbnb **rental prices** using advanced Machine Learning models.
- Perform **sentiment analysis** on guest reviews.
- Utilize **geo-visualizations** with property coordinates.

---

## **📖 Table of Contents**
1. [📌 Project Overview](#project-overview)  
2. [🔧 Features](#features)  
3. [📊 Data Sources & Structure](#data-sources--structure)  
4. [📌 Data Dictionary](#data-dictionary)  
5. [🚀 Installation & Setup](#installation--setup)  
6. [🏡 App Usage](#app-usage)  
7. [🔬 Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
8. [📊 Price Analysis](#price-analysis)  
9. [💰 Price Prediction](#price-prediction)  
10. [💬 Sentiment Analysis](#sentiment-analysis)  
11. [🗺️ Geo-Visualizations](#geo-visualizations)  
12. [🔧 Model Training & Feature Selection](#model-training--feature-selection)  
13. [📜 License & Acknowledgments](#license--acknowledgments)  

---

## **📌 Project Overview**
This **Streamlit-based dashboard** helps users explore and analyze Airbnb listings.  
It supports **five cities**: **Bristol, Geneva, Hague, Hong Kong, and Singapore**.  

The app enables:  
✔️ **Data exploration** with summary statistics and missing values handling.  
✔️ **Airbnb price trend analysis** through visualizations.  
✔️ **Machine Learning-powered price prediction** using dynamic feature selection.  
✔️ **Sentiment analysis** on guest reviews.  
✔️ **Geo-visualizations** for property mapping.  

---

## **🔧 Features**
- **Multi-city support**: Select Airbnb data from **5 cities**.
- **Interactive price analysis**: Analyze pricing trends across neighborhoods.
- **AI-powered price prediction**: Uses **Random Forest & XGBoost models**.
- **Sentiment analysis**: Understand customer satisfaction from guest reviews.
- **Geo-visualizations**: **Scatter Mapbox** & **Choropleth Maps** provide location-based insights.

---

## **📊 Data Sources & Structure**
The app uses **detailed Airbnb datasets** from [Inside Airbnb](https://insideairbnb.com/).  
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

## **📌 Data Dictionary**
🔗 **[Official Airbnb Data Dictionary](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit?usp=sharing)**  

Key attributes:
- `price`: Nightly rental price of an Airbnb listing.
- `room_type`: Type of Airbnb accommodation (Entire home, Private room, etc.).
- `neighbourhood_cleansed`: Cleaned version of the neighborhood name.
- `latitude` & `longitude`: Geo-coordinates for mapping.
- `number_of_reviews`: Count of guest reviews.
- `review_scores_rating`: Overall rating of the listing.

---

## **🚀 Installation & Setup**
To run this project on your local machine:

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/hareesh-p/airbnb-streamlit.git
cd airbnb-streamlit
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Run the Streamlit App**
```sh
streamlit run app.py
```

---

## **🏡 App Usage**
The app has **four main sections**:

### **1️⃣ Exploratory Data Analysis (EDA)**
- Provides an **overview of the dataset**.
- Displays **missing values** and **summary statistics**.
- Visualizes **price distribution** and **room type proportions**.

### **2️⃣ Price Analysis**
- **Price trends** across different neighborhoods.
- **Neighborhood filtering** to analyze pricing in specific areas.
- **Maps & Visualizations**:
  - **Scatter Map**: Displays **individual property prices**.
  - **Choropleth Map**: Shows **average price per neighborhood**.

### **3️⃣ Price Prediction**
- Uses **Random Forest & XGBoost models** to predict Airbnb prices.
- **Automatically selects the best features** for prediction.
- Compares different models using **R² scores**.
- Users **input listing details** to get a **predicted price**.

### **4️⃣ Sentiment Analysis**
- Generates a **Word Cloud** from guest reviews.
- Uses **VADER Sentiment Analysis** to classify reviews.
- Displays **sentiment distribution** across listings.

---

## **🗺️ Geo-Visualizations**
- **Scatter Mapbox** – Maps **individual Airbnb properties**.
- **Choropleth Map** – Displays **average price per neighborhood**.
- **Uses GeoJSON files** for accurate location-based insights.

---

## **🔧 Model Training & Feature Selection**
1. **Handles missing values**, outliers, and normalizes numerical data.
2. Uses **Random Forest & XGBoost** for predictions.
3. **Feature Selection**:
   - Selects **top 10 important features** dynamically per city.
   - Ensures **categorical variables** (e.g., `property_type`) are included.
   - **Removes outliers** using **Interquartile Range (IQR)**.
4. **Hyperparameter Tuning**:
   - Optimizes `n_estimators`, `max_depth`, and `learning_rate` for XGBoost.
   - Uses **Grid Search** for optimal Random Forest settings.

---

## **📜 License & Acknowledgments**
This project is for **educational and analytical purposes** only.  
**Data Source**: [Inside Airbnb](https://insideairbnb.com/)  
📌 **Contributors**: Sumit Kumar, Hareesh P & Abhishek Mishra 

---

🚀 **Now Ready to Explore the Airbnb Market!** 🚀  
🎯 **Run the app & analyze Airbnb trends like never before!** 🎯  
