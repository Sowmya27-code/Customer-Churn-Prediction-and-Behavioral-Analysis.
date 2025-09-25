import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from wordcloud import WordCloud
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

## 🧹 Data Preprocessing Function
def preprocess_dataset(data_frame):
    """
    Cleans and preprocesses the DataFrame for analysis.
    This includes handling missing values, removing duplicates,
    and encoding categorical features.
    """
    # Impute missing numerical data with the mean
    for col in data_frame.select_dtypes(include=np.number):
        data_frame[col].fillna(data_frame[col].mean(), inplace=True)
    
    # Drop any duplicate rows
    data_frame.drop_duplicates(inplace=True)
    
    # Use LabelEncoder for non-numerical columns, excluding 'date'
    for column in data_frame.select_dtypes(include=['object']):
        if column != 'date':
            label_encoder = LabelEncoder()
            data_frame[column] = label_encoder.fit_transform(data_frame[column].astype(str))
    
    return data_frame

## 📊 Exploratory Data Analysis (EDA)
def conduct_eda(df_input):
    """
    Performs exploratory data analysis on the DataFrame
    by generating summary statistics and creating key visualizations.
    """
    # Generate and print descriptive statistics
    descriptive_stats = df_input.describe()
    
    # Plot distributions of the first three numerical columns
    numerical_cols = df_input.select_dtypes(include=[np.number]).columns[:3]
    fig, axes = plt.subplots(1, len(numerical_cols), figsize=(15, 5))
    for i, col in enumerate(numerical_cols):
        sns.histplot(df_input[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('distribution_plots.png')
    plt.close()
    
    # Create a heatmap to visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = df_input.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    return descriptive_stats

## 🖼️ Visualization Generation
def generate_visualizations(df_to_viz):
    """
    Creates various plots to represent the data visually.
    """
    # Bar plot of column means
    plt.figure(figsize=(10, 6))
    df_to_viz.mean(numeric_only=True).plot(kind='bar')
    plt.title('Average Values for Numeric Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('mean_values_bar_plot.png')
    plt.close()
    
    # Check for 'date' column for time series plot
    if 'date' in df_to_viz.columns:
        plt.figure(figsize=(12, 6))
        df_to_viz.groupby('date').mean(numeric_only=True).plot(ax=plt.gca())
        plt.title('Time Series Trends')
        plt.savefig('time_series_line_plot.png')
        plt.close()
    
    # Scatter plot of two numerical features
    numeric_features = df_to_viz.select_dtypes(include=[np.number]).columns[:2]
    if len(numeric_features) >= 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(df_to_viz[numeric_features[0]], df_to_viz[numeric_features[1]])
        plt.xlabel(numeric_features[0])
        plt.ylabel(numeric_features[1])
        plt.title(f'Scatter Plot of {numeric_features[0]} vs {numeric_features[1]}')
        plt.savefig('feature_scatter_plot.png')
        plt.close()

## 🧠 Machine Learning Analysis
def run_ml_analysis(dataframe):
    """
    Prepares the data, trains machine learning models (Logistic Regression
    and Random Forest), and evaluates their performance.
    """
    numeric_subset = dataframe.select_dtypes(include=[np.number])
    features = numeric_subset.iloc[:, :-1]
    target = numeric_subset.iloc[:, -1]
    
    # Partition the data into training and testing sets
    X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(
        features, target, test_size=0.25, random_state=42
    )
    
    # Initialize and train the models
    ml_models = {
        'Logistic Regression': LogisticRegression(max_iter=250),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=120)
    }
    
    performance_metrics = {}
    for model_name, model_instance in ml_models.items():
        model_instance.fit(X_train_set, y_train_set)
        predictions = model_instance.predict(X_test_set)
        
        # Calculate key performance metrics
        performance_metrics[model_name] = {
            'Accuracy': accuracy_score(y_test_set, predictions),
            'Precision': precision_score(y_test_set, predictions, average='weighted'),
            'Recall': recall_score(y_test_set, predictions, average='weighted'),
            'F1-Score': f1_score(y_test_set, predictions, average='weighted')
        }
    
    return performance_metrics

## 💬 Natural Language Processing (NLP) - Sentiment Analysis
def perform_sentiment_analysis(text_corpus):
    """
    Analyzes the sentiment of a collection of text data and generates
    a word cloud visualization.
    """
    processed_texts = []
    # Tokenize, remove stopwords, and clean text
    for text_item in text_corpus:
        tokens = word_tokenize(str(text_item).lower())
        stop_words = set(stopwords.words('english'))
        clean_tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        processed_texts.append(' '.join(clean_tokens))
    
    # Calculate sentiment polarity using TextBlob
    sentiment_scores = [TextBlob(text).sentiment.polarity for text in processed_texts]
    
    # Generate a word cloud from the processed text
    full_text = ' '.join(processed_texts)
    wordcloud_image = WordCloud(width=900, height=450, background_color='white').generate(full_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_image, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('text_wordcloud.png')
    plt.close()
    
    return pd.Series(sentiment_scores).describe()

## 🚀 Main Execution Block
def main_pipeline():
    """
    Main function to execute the full data analysis pipeline.
    """
    try:
        print("Starting data analysis pipeline...")
        
        # 1. Load the dataset
        churn_data = pd.read_csv('churn-bigml-80.csv')
        
        # 2. Preprocess the data
        print("Step 1: Preprocessing the raw data...")
        cleaned_churn_data = preprocess_dataset(churn_data.copy())
        
        # 3. Conduct Exploratory Data Analysis
        print("Step 2: Conducting exploratory data analysis...")
        summary_stats = conduct_eda(cleaned_churn_data)
        
        # 4. Generate Visualizations
        print("Step 3: Generating various data visualizations...")
        generate_visualizations(cleaned_churn_data)
        
        # 5. Run Machine Learning Models
        print("Step 4: Training and evaluating machine learning models...")
        model_results = run_ml_analysis(cleaned_churn_data)
        
        # 6. Perform Sentiment Analysis (if applicable)
        final_results = model_results.copy()
        if 'text' in cleaned_churn_data.columns:
            print("Step 5: Performing sentiment analysis on text data...")
            sentiment_summary = perform_sentiment_analysis(cleaned_churn_data['text'])
            final_results['Sentiment Analysis Summary'] = sentiment_summary
        
        # 7. Save results
        with open('analysis_report.txt', 'w') as f:
            f.write("=== Data Analysis Report ===\n\n")
            f.write("1. Descriptive Statistics:\n")
            f.write(str(summary_stats))
            f.write("\n\n2. Machine Learning Model Performance:\n")
            f.write(str(model_results))
            f.write("\n\n3. NLP Results (if available):\n")
            f.write(str(final_results.get('Sentiment Analysis Summary', 'N/A')))
        
        print("\nAnalysis pipeline completed successfully! Results saved to 'analysis_report.txt'.")
        
    except FileNotFoundError:
        print("Error: The file 'churn-bigml-80.csv' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {str(e)}")

if __name__ == "__main__":
    main_pipeline()