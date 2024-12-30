import pandas as pd
import re
from nltk.sentiment import SentimentIntensityAnalyzer

def count_hashtags(message):
    return len(re.findall(r'#\w+', message)) if isinstance(message, str) else 0

def count_emojis(message):
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|"
        "[\U0001F680-\U0001F6FF]|[\U0001F700-\U0001F77F]|"
        "[\U0001F780-\U0001F7FF]|[\U0001F800-\U0001F8FF]|"
        "[\U0001F900-\U0001F9FF]|[\U0001FA00-\U0001FA6F]|"
        "[\U0001FA70-\U0001FAFF]|[\U00002702-\U000027B0]|"
        "[\U000024C2-\U0001F251]"
    )
    return len(emoji_pattern.findall(message)) if isinstance(message, str) else 0

def calculate_sentiment(message):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(message) if isinstance(message, str) else {'compound': 0}
    return sentiment['compound']

def calculate_text_length(message):
    return len(message.split()) if isinstance(message, str) else 0

def engineer_features(df):
    """
    Takes a DataFrame with columns: ['message', 'created_time', ...]
    and adds new columns: [hashtags, emojis, sentiment, text_length, hour, day_of_week].
    """
    df['hashtags'] = df['message'].apply(count_hashtags)
    df['emojis'] = df['message'].apply(count_emojis)
    df['sentiment'] = df['message'].apply(calculate_sentiment)
    df['text_length'] = df['message'].apply(calculate_text_length)

    df['created_time'] = pd.to_datetime(df['created_time'])
    df['hour'] = df['created_time'].dt.hour
    df['day_of_week'] = df['created_time'].dt.day_name()
    return df
