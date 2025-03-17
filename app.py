import logging  # Import logging module
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import io
import base64
from transformers import pipeline
from collections import Counter
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import torch

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Sentiment analysis pipeline using distilBERT
logging.info("Loading sentiment analysis model...")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
logging.info("Model loaded successfully.")


def clean_and_prepare_data(data):
    try:
        processed_data = []

        for response in data:
            rating = None
            text_feedback = []

            for answer in response['answers']:
                if answer['questionType'] == 'rating':
                    rating = answer['answer']
                elif answer['questionType'] == 'text':
                    text_feedback.append(str(answer['answer']))

            processed_data.append({
                'eNPS Score': rating,
                'Feedback': ' '.join(text_feedback),
                'Department': response.get('department', 'Unknown'),
                'Date': pd.to_datetime(response.get('createdAt', datetime.now()))
            })

        df = pd.DataFrame(processed_data)
        df['eNPS Score'] = pd.to_numeric(df['eNPS Score'], errors='coerce')
        return df

    except Exception as e:
        logging.error(f"Error during data preparation: {str(e)}")
        raise


def generate_word_cloud(texts):
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100
        ).generate(' '.join(texts))

        img = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()

        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()
    except Exception as e:
        logging.error(f"Word cloud generation error: {str(e)}")
        return None


def perform_sentiment_analysis(df):
    def analyze_sentiment(text):
        if isinstance(text, str) and text.strip():
            try:
                result = sentiment_analyzer(text[:512])[0]
                score = float(result['score'])
                return score if result['label'] == 'POSITIVE' else -score
            except Exception as e:
                logging.error(f"Sentiment analysis error: {str(e)}")
                return 0.0
        return 0.0

    try:
        df['Sentiment_Score'] = df['Feedback'].apply(analyze_sentiment)
        df['Sentiment_Score'] = pd.to_numeric(df['Sentiment_Score'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {str(e)}")
        df['Sentiment_Score'] = 0.0
        return df


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=80)
