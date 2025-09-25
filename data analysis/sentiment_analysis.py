from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

class SentimentAnalyzer:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text: str) -> str:
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words]
        
        return " ".join(tokens)
    
    def analyze_sentiment(self, texts: List[str]) -> Dict:
        sentiments = []
        processed_texts = []
        
        for text in texts:
            processed = self.preprocess_text(text)
            processed_texts.append(processed)
            blob = TextBlob(processed)
            sentiments.append(blob.sentiment.polarity)
        
        # Create word cloud
        all_text = " ".join(processed_texts)
        wordcloud = WordCloud(width=800, height=400).generate(all_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig("visualizations/wordcloud.png")
        plt.close()
        
        return {
            "sentiment_scores": sentiments,
            "average_sentiment": np.mean(sentiments),
            "sentiment_distribution": {
                "positive": sum(1 for s in sentiments if s > 0),
                "neutral": sum(1 for s in sentiments if s == 0),
                "negative": sum(1 for s in sentiments if s < 0)
            }
        }