import nltk
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")
nltk.download("vader_lexicon")

# Load stopwords and sentiment analyzer once
STOPWORDS = set(stopwords.words("english"))
SIA = SentimentIntensityAnalyzer()

def clean_text(text):
    """ Tokenize and remove stopwords from text efficiently """
    if pd.isna(text) or not isinstance(text, str):  # Handle NaN and non-string values
        return ""
    words = [word for word in word_tokenize(text.lower()) if word.isalpha() and word not in STOPWORDS]
    return " ".join(words)

def generate_wordcloud(reviews):
    """ Generate a word cloud for Airbnb reviews """
    if "comments" not in reviews.columns:
        st.write("No review comments available for this city.")
        return

    # Bulk clean reviews
    cleaned_reviews = reviews["comments"].dropna().map(clean_text)

    if cleaned_reviews.empty:
        st.write("No valid reviews available for word cloud.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(cleaned_reviews))

    # Create a figure explicitly
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    # Pass the figure to st.pyplot() explicitly
    st.pyplot(fig)

def analyze_sentiment(reviews):
    """ Perform sentiment analysis using VADER for speed """
    if "comments" not in reviews.columns:
        st.write("No review comments available for sentiment analysis.")
        return

    # Convert comments to string (avoids NaN issues)
    reviews["sentiment_score"] = reviews["comments"].dropna().map(lambda x: SIA.polarity_scores(str(x))["compound"])

    if reviews["sentiment_score"].dropna().empty:
        st.write("No valid reviews available for sentiment analysis.")
        return

    avg_sentiment = reviews["sentiment_score"].mean()
    st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f} (Scale: -1 to 1)")

    # Display sentiment distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(reviews["sentiment_score"], bins=30, color="blue", edgecolor="black")
    ax.set_title("Distribution of Review Sentiment Scores")
    ax.set_xlabel("Sentiment Score (-1: Negative, 1: Positive)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
