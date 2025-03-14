import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from textblob import TextBlob

nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    """ Tokenize and remove stopwords from text """
    words = word_tokenize(str(text).lower())
    words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]
    return " ".join(words)

def generate_wordcloud(reviews):
    """ Generate a word cloud for Airbnb reviews """
    if "comments" in reviews.columns:  # Check if the comments column exists
        reviews["cleaned_text"] = reviews["comments"].dropna().apply(clean_text)
        
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(reviews["cleaned_text"]))

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot()
    else:
        st.write("No review comments available for this city.")

def analyze_sentiment(reviews):
    """ Perform sentiment analysis using TextBlob """
    if "comments" in reviews.columns:
        reviews["sentiment_score"] = reviews["comments"].dropna().apply(lambda x: TextBlob(x).sentiment.polarity)
        
        avg_sentiment = reviews["sentiment_score"].mean()
        st.write(f"**Average Sentiment Score:** {avg_sentiment:.2f} (Scale: -1 to 1)")

        # Display sentiment distribution
        fig = plt.figure(figsize=(8, 4))
        plt.hist(reviews["sentiment_score"].dropna(), bins=30, color="blue", edgecolor="black")
        plt.title("Distribution of Review Sentiment Scores")
        plt.xlabel("Sentiment Score (-1: Negative, 1: Positive)")
        plt.ylabel("Count")
        st.pyplot(fig)
    else:
        st.write("No review comments available for sentiment analysis.")
