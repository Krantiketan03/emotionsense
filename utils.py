import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    """
    Preprocess text for emotion analysis.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Simple word tokenization with regular expressions
    tokens = re.findall(r'\b\w+\b', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def highlight_influential_words(text, influential_words):
    """
    Highlight influential words in the original text.
    
    Args:
        text (str): Original input text
        influential_words (dict): Dictionary mapping words to their influence scores
        
    Returns:
        str: HTML-formatted text with highlighted words
    """
    if not influential_words:
        return text
    
    # Normalize influence scores to a 0-1 scale
    max_influence = max(influential_words.values())
    normalized_scores = {word: score/max_influence for word, score in influential_words.items()}
    
    # Sort words by length (descending) to handle cases where words might be substrings of others
    sorted_words = sorted(normalized_scores.keys(), key=len, reverse=True)
    
    # Create a copy of the text for highlighting
    highlighted_text = text
    
    # Add highlighting to each influential word
    for word in sorted_words:
        # Skip very short words
        if len(word) < 3:
            continue
        
        # Generate highlighting color based on influence score
        score = normalized_scores[word]
        intensity = int(255 * (1 - score))  # Higher score = more intense color
        color = f"rgb(255, {intensity}, {intensity})"
        
        # Create pattern to match the word with word boundaries
        pattern = r'\b' + re.escape(word) + r'\b'
        
        # Create replacement with highlighting
        replacement = f'<span style="background-color: {color}; padding: 0px 2px; border-radius: 3px; font-weight: bold;">{word}</span>'
        
        # Replace in case-insensitive manner
        highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)
    
    return highlighted_text
