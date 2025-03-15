import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import json
from load_data import load_airbnb_data
from price_prediction import train_price_model
from sentiment_analysis import generate_wordcloud, analyze_sentiment

# 📍 Available cities
cities = ["bristol", "geneva", "hague", "hongkong", "singapore"]

# Sidebar: Select a city
st.sidebar.title("City Selection")
selected_city = st.sidebar.selectbox("Choose a City", cities)

# Load data (Only detailed dataset is used)
listings, reviews, neighbourhoods = load_airbnb_data(selected_city)

# Load GeoJSON file for the selected city
geojson_path = f"data/{selected_city}/neighbourhoods.geojson"
with open(geojson_path, "r") as f:
    geojson_data = json.load(f)

# App Title
st.title(f"🏡 Airbnb Market Analysis for {selected_city.capitalize()}")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Exploratory Data Analysis", "Price Analysis", "Price Prediction", "Sentiment Analysis"])

# 📌 1️⃣ Exploratory Data Analysis (EDA)
if page == "Exploratory Data Analysis":
    st.header(f"📊 Exploratory Data Analysis (EDA) - {selected_city.capitalize()}")

    # 📌 Dataset Overview
    st.subheader("🔍 Dataset Overview")
    st.write(f"Total Listings: {listings.shape[0]}")
    st.write(f"Total Columns: {listings.shape[1]}")
    st.write("Preview of the Dataset:")
    st.dataframe(listings.head())

    # 📌 Missing Values Summary
    st.subheader("🛑 Missing Values in Dataset")
    missing_values = listings.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    if not missing_values.empty:
        st.dataframe(missing_values.to_frame().rename(columns={0: "Missing Count"}))
    else:
        st.write("✅ No missing values in the dataset.")

    # 📌 Summary Statistics (Excluding Non-Informative Fields)
    st.subheader("📈 Summary Statistics")
    excluded_columns = ["id", "listing_url", "scrape_id", "host_id", "host_url", "picture_url"]
    numeric_cols = listings.select_dtypes(include=["number", "float", "int"]).columns
    numeric_cols = [col for col in numeric_cols if col not in excluded_columns]

    if numeric_cols:
        st.write(listings[numeric_cols].describe())
    else:
        st.write("⚠️ No relevant numeric features for summary statistics.")

    # 📌 Price Distribution
    st.subheader("💰 Price Distribution")
    fig = px.histogram(listings, x="price", nbins=50, title="Price Distribution Across Listings")
    st.plotly_chart(fig)

# 📌 2️⃣ Price Analysis Page
elif page == "Price Analysis":
    st.header(f"📊 Airbnb Price Analysis in {selected_city.capitalize()}")

    # 📌 Properties Count by Neighborhood
    st.subheader("🏘️ Properties Count by Neighborhood")
    neighborhood_counts = listings["neighbourhood_cleansed"].value_counts().reset_index()
    neighborhood_counts.columns = ["Neighborhood", "Property Count"]
    st.dataframe(neighborhood_counts)

    # 📌 Option to View All Properties
    show_all = st.checkbox("Show all properties")
    if show_all:
        st.write("Displaying all properties in the dataset:")
        st.dataframe(listings)
    else:
        # Select Neighborhood
        neighborhood = st.selectbox("Select a Neighborhood", listings["neighbourhood_cleansed"].unique())

        # Display Average Price in Selected Neighborhood
        avg_price = listings[listings["neighbourhood_cleansed"] == neighborhood]["price"].mean()
        st.write(f"**Average Price in {neighborhood}:** ${avg_price:.2f}")

        # Show Price Distribution
        fig = px.histogram(listings[listings["neighbourhood_cleansed"] == neighborhood], x="price", nbins=50, title="Price Distribution")
        st.plotly_chart(fig)

    # 📌 Price Distribution Visualization
    st.subheader("📊 Price Distribution Across Listings")
    fig = px.histogram(listings, x="price", nbins=50, title="Distribution of Airbnb Prices")
    st.plotly_chart(fig)

    # 📌 Average Price by Room Type
    st.subheader("🏠 Average Price by Room Type")
    avg_price_room_type = listings.groupby("room_type")["price"].mean().reset_index()
    fig = px.bar(avg_price_room_type, x="room_type", y="price", title="Average Price by Room Type")
    st.plotly_chart(fig)

    # 📌 Map Visualization Toggle
    st.subheader("🗺️ Map Visualization")
    map_option = st.radio("Choose Map Type", ["Scatter Map (Price Distribution)", "Choropleth (Average Price)"])

    if map_option == "Scatter Map (Price Distribution)":
        st.subheader("📍 Price Distribution on Map")

        # Drop rows with NaN values in 'price'
        listings_clean = listings.dropna(subset=["price"])

        fig = px.scatter_mapbox(
            listings_clean, lat="latitude", lon="longitude", color="price",
            size="price", zoom=10, mapbox_style="open-street-map",
            hover_name="name", hover_data=["neighbourhood_cleansed", "room_type"]
        )
        st.plotly_chart(fig)

    elif map_option == "Choropleth (Average Price)":
        st.subheader("🗺️ Average Price by Neighbourhood")
        neighbourhood_avg_price = listings.groupby("neighbourhood_cleansed")["price"].mean().reset_index()
        fig = px.choropleth_mapbox(
            neighbourhood_avg_price, 
            geojson=geojson_data, 
            locations="neighbourhood_cleansed", 
            featureidkey="properties.neighbourhood", 
            color="price",
            color_continuous_scale="Viridis",
            mapbox_style="carto-positron",
            zoom=10, 
            center={"lat": listings["latitude"].mean(), "lon": listings["longitude"].mean()}
        )
        st.plotly_chart(fig)

# 📌 3️⃣ Price Prediction Page
elif page == "Price Prediction":
    st.header(f"📈 Predict Airbnb Prices in {selected_city.capitalize()}")

    # Train price model and get dynamic features
    model, performance_df, selected_features, feature_importance, scaler, numerical_features = train_price_model(listings)

    # 📌 Display Model Performance Comparison
    st.subheader("📊 Model Performance Comparison")
    st.dataframe(performance_df)

    # Highlight the best-performing model
    best_model_name = performance_df["R² Score"].idxmax()
    st.write(f"🏆 **Best Performing Model:** {best_model_name}")

    # 📌 Dynamically create user inputs
    user_inputs = {}
    for feature in numerical_features:
        user_inputs[feature] = st.number_input(f"Enter {feature.replace('_', ' ').title()}", min_value=0, max_value=100, value=2)

    # Convert user input into DataFrame
    input_data = pd.DataFrame([user_inputs])

    # Ensure input data matches training order
    input_data = input_data.reindex(selected_features, axis=1, fill_value=0)

    # Normalize numerical features
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # Predict price
    predicted_price = model.predict(input_data)[0]
    st.write(f"**Predicted Price:** ${predicted_price:.2f}")

# 📌 4️⃣ Sentiment Analysis Page
elif page == "Sentiment Analysis":
    st.header(f"💬 Guest Review Analysis for {selected_city.capitalize()}")

    # Show word cloud for Airbnb reviews
    st.subheader("🌥️ Word Cloud of Guest Reviews")
    generate_wordcloud(reviews)

    # Show sentiment analysis
    st.subheader("📊 Sentiment Distribution")
    analyze_sentiment(reviews)
