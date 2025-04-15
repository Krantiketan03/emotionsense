import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import os
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from model_training import train_emotion_model
from utils import preprocess_text

class EmotionAnalyzer:
    def __init__(self):
        # Download resources if they don't exist
        self._download_nltk_resources()
        
        # Initialize model and vectorizer
        self.emotions = ["happiness", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 2), 
            min_df=2
        )
        
        # Try to load the model or train a new one
        self.model, self.vectorizer = self._load_or_train_model()
        
        # Initialize lemmatizer and stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def _download_nltk_resources(self):
        """Download necessary NLTK resources."""
        resources = ['punkt', 'stopwords', 'wordnet']
        for resource in resources:
            nltk.download(resource, quiet=True)
    
    def _load_or_train_model(self):
        """Load pre-trained model or train a new one."""
        try:
            # Try to load the model and vectorizer
            model_file = os.path.join(os.path.dirname(__file__), 'emotion_model.pkl')
            vectorizer_file = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
            
            if os.path.exists(model_file) and os.path.exists(vectorizer_file):
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                with open(vectorizer_file, 'rb') as f:
                    vectorizer = pickle.load(f)
                return model, vectorizer
            else:
                raise FileNotFoundError("Pre-trained model not found. Training new model.")
        except (FileNotFoundError, EOFError):
            # If loading fails, train a new model
            print("Training a new emotion detection model...")
            return train_emotion_model()
    
    def analyze_emotion(self, text):
        """
        Analyze the emotional content of a text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            tuple: (emotion, confidence, influential_words)
                - emotion: predicted emotion as string
                - confidence: confidence score (0-100)
                - influential_words: dictionary of influential words and their weights
        """
        # Preprocess the text
        preprocessed_text = preprocess_text(text)
        
        # Vectorize the text
        features = self.vectorizer.transform([preprocessed_text])
        
        # Get prediction and probabilities
        emotion_idx = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Get the predicted emotion and confidence
        emotion = self.emotions[emotion_idx]
        confidence = probabilities[emotion_idx] * 100
        
        # Identify influential words
        influential_words = self._get_influential_words(text, emotion_idx)
        
        return emotion, confidence, influential_words
    
    def get_emotion_probabilities(self, text):
        """
        Get probabilities for all emotions.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Mapping from emotion names to probability scores
        """
        # Preprocess the text
        preprocessed_text = preprocess_text(text)
        
        # Vectorize the text
        features = self.vectorizer.transform([preprocessed_text])
        
        # Get probabilities
        probabilities = self.model.predict_proba(features)[0]
        
        # Create dictionary mapping emotions to probabilities
        emotion_probs = {emotion: prob for emotion, prob in zip(self.emotions, probabilities)}
        
        return emotion_probs
        
    def _get_influential_words(self, text, emotion_idx):
        """
        Identify words that influenced the emotion prediction.
        
        Args:
            text (str): Original input text
            emotion_idx (int): Index of the predicted emotion
            
        Returns:
            dict: Mapping from influential words to their weights
        """
        # Get words from the text
        words = re.findall(r'\b\w+\b', text.lower())
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Dictionary to store word influences
        word_influences = {}
        
        # For each word, calculate its influence on the prediction
        for word in set(words):
            # Get the feature index for this word if it exists in the vectorizer
            try:
                word_idx = self.vectorizer.vocabulary_.get(word)
                if word_idx is not None:
                    # Get the coefficient for this word for the predicted emotion
                    coefficient = self.model.coef_[emotion_idx, word_idx]
                    word_influences[word] = abs(coefficient)
            except:
                continue
        
        # If we couldn't find any influential words, use word frequency
        if not word_influences:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            for word, freq in word_freq.items():
                word_influences[word] = freq / len(words)
        
        # Normalize the influences to a 0-1 scale
        if word_influences:
            max_influence = max(word_influences.values())
            word_influences = {word: score/max_influence for word, score in word_influences.items()}
        
        return word_influences
