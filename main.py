import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob

# --- Step 1: Load Data ---
# For demo: Load a sample CSV (or use Twitter API to fetch live tweets)
data = pd.read_csv("sample_tweets.csv")  # Contains a 'text' column

# --- Step 2: Clean and Analyze Sentiment ---
def get_sentiment(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity

data['polarity'] = data['text'].apply(get_sentiment)
data['sentiment'] = data['polarity'].apply(
    lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')
)

# --- Step 3: Visualize Sentiment Distribution ---
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=data, palette='viridis')
plt.title('Sentiment Count')
plt.show()

# --- Step 4: WordCloud for Positive Sentiment ---
positive_text = ' '.join(data[data['sentiment'] == 'positive']['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Positive Sentiment Word Cloud")
plt.show()

# --- Step 5: Average Sentiment Score ---
print("Average Polarity Score:", data['polarity'].mean())

