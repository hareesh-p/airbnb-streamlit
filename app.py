import streamlit as st
import pandas as pd
import plotly.express as px
from load_data import load_airbnb_data
from price_prediction import train_price_model
from sentiment_analysis import generate_wordcloud

# 📍 Available cities
cities = ["capetown", "geneva", "hague", "hongkong", "singapore"]

# Sidebar: Select a city
st.sidebar.title("City Selection")
selected_city = st.sidebar.selectbox("Choose a City", cities)

# Option to select detailed or summary dataset
data_type = st.sidebar.radio("Select Data Type", ["Detailed", "Summary"])

# Load appropriate data based on selection
listings, reviews, neighbourhoods = load_airbnb_data(selected_city, data_type)

# App Title
st.title(f"🏡 Airbnb Market Analysis for {selected_city.capitalize()}")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Price Analysis", "Price Prediction", "Sentiment Analysis"])

# 📌 1️⃣ Price Analysis Page
if page == "Price Analysis":
    st.header(f"📊 Airbnb Price Analysis in {selected_city.capitalize()}")

    # Display count of properties in each neighborhood
    st.subheader("🏘️ Properties Count by Neighborhood")
    neighborhood_counts = listings["neighbourhood_cleansed"].value_counts().reset_index()
    neighborhood_counts.columns = ["Neighborhood", "Property Count"]
    st.dataframe(neighborhood_counts)

    # Option to view all properties or filter by neighborhood
    show_all = st.checkbox("Show all properties")
    if show_all:
        st.write("Displaying all properties in the dataset:")
        st.dataframe(listings)
    else:
        # Select neighborhood
        neighborhood = st.selectbox("Select a Neighborhood", listings["neighbourhood_cleansed"].unique())

        # Display average price in the selected neighborhood
        avg_price = listings[listings["neighbourhood_cleansed"] == neighborhood]["price"].mean()
        st.write(f"**Average Price in {neighborhood}:** ${avg_price:.2f}")

        # Show price distribution
        fig = px.histogram(listings[listings["neighbourhood_cleansed"] == neighborhood], x="price", nbins=50, title="Price Distribution")
        st.plotly_chart(fig)

# 📌 2️⃣ Price Prediction Page
elif page == "Price Prediction":
    st.header(f"📈 Predict Airbnb Prices in {selected_city.capitalize()}")

    # Train price model
    model, r2 = train_price_model(listings)
    st.write(f"Model R² Score: {r2:.2f}")

    # User inputs
    accommodates = st.number_input("Number of Guests", min_value=1, max_value=10, value=2)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=5, value=1)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=3, value=1)
    reviews_count = st.number_input("Number of Reviews", min_value=0, max_value=500, value=10)

    # Predict price
    input_data = pd.DataFrame([[accommodates, bedrooms, bathrooms, reviews_count]], columns=["accommodates", "bedrooms", "bathrooms", "number_of_reviews"])
    predicted_price = model.predict(input_data)[0]
    st.write(f"**Predicted Price:** ${predicted_price:.2f}")

# 📌 3️⃣ Sentiment Analysis Page
elif page == "Sentiment Analysis":
    st.header(f"💬 Guest Review Analysis for {selected_city.capitalize()}")

    # Show word cloud for Airbnb reviews
    st.subheader("Word Cloud of Guest Reviews")
    generate_wordcloud(reviews)
