import os
import nltk
import random
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import preprocess_text

def load_emotion_data():
    """
    Load emotion training data or create synthetic data if file doesn't exist.
    
    Returns:
        list: List of (text, emotion) tuples
    """
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'emotions.txt')
    
    try:
        # Try to load the data from file
        with open(data_path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                if line.strip():  # Skip empty lines
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        text, emotion = parts
                        data.append((text, emotion))
        
        if not data:
            raise FileNotFoundError("Data file exists but is empty")
        
        return data
    
    except (FileNotFoundError, OSError):
        # If file doesn't exist or can't be read, create synthetic data
        print("Emotion data file not found. Creating synthetic training data...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Create synthetic training data
        emotions = ["happiness", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
        data = []
        
        # Sample statements for each emotion
        happiness_texts = [
            "I am so happy today! Everything is going great.",
            "The celebration was wonderful and filled with joy.",
            "Spending time with my family makes me incredibly happy.",
            "I got promoted at work and I'm thrilled!",
            "This is the best day of my life, I'm so excited.",
            "We had a fantastic vacation full of fun and laughter.",
            "I'm delighted to announce that we're expecting a baby!",
            "The birthday party was a complete success and everyone had fun.",
            "I can't stop smiling after receiving such wonderful news.",
            "After years of hard work, I finally achieved my dream."
        ]
        
        sadness_texts = [
            "I feel so sad and empty inside after the loss.",
            "The movie's ending made me cry uncontrollably.",
            "I miss my old friends and the times we shared together.",
            "The news about the accident left me heartbroken.",
            "I've been feeling down and blue for several days now.",
            "Saying goodbye was one of the hardest things I've ever done.",
            "The story of the abandoned dog made me incredibly sad.",
            "I couldn't hold back tears when I heard about their situation.",
            "Sometimes life seems so unfair and it makes me melancholic.",
            "The funeral was a somber reminder of how precious life is."
        ]
        
        anger_texts = [
            "I'm furious about how they treated me at the restaurant!",
            "The customer service was terrible and I demand a refund!",
            "I can't believe they lied to me, I'm absolutely livid.",
            "Stop interrupting me when I'm trying to explain something!",
            "The constant noise from the neighbors is driving me mad.",
            "I'm fed up with people who don't respect others' time.",
            "The unfair treatment at work makes my blood boil.",
            "I slammed the door after our heated argument.",
            "That driver cut me off and nearly caused an accident! Unbelievable!",
            "I've had enough of your excuses and broken promises!"
        ]
        
        fear_texts = [
            "I'm terrified of what might happen during the surgery.",
            "The dark alley made me feel very unsafe and scared.",
            "The thought of public speaking fills me with dread.",
            "I'm afraid of losing my job in the upcoming layoffs.",
            "The strange noise in the middle of the night frightened me.",
            "I have a deep fear of heights that I can't overcome.",
            "The possibility of failure keeps me awake at night.",
            "I'm scared about the uncertain future ahead of us.",
            "The horror movie was so terrifying I couldn't watch alone.",
            "The diagnosis has left me fearful about what comes next."
        ]
        
        surprise_texts = [
            "I couldn't believe it when they jumped out to surprise me!",
            "The unexpected plot twist caught me completely off guard.",
            "Wow! I never expected to win the competition!",
            "The surprise birthday party left me speechless.",
            "I was astonished to discover the truth about my ancestry.",
            "My jaw dropped when I saw the incredible view for the first time.",
            "The sudden announcement of their engagement shocked everyone.",
            "I was taken aback by how much the town had changed.",
            "The magic trick was so amazing that I was completely baffled.",
            "No way! I never thought they would actually do it!"
        ]
        
        disgust_texts = [
            "The smell of rotten food made me feel sick to my stomach.",
            "I was disgusted by the unsanitary conditions of the restaurant.",
            "The sight of the moldy bread repulsed me completely.",
            "Their cruel behavior towards animals is absolutely revolting.",
            "I couldn't stand the disgusting mess left in the bathroom.",
            "The way they chewed with their mouth open was nauseating.",
            "I was grossed out by the slimy texture of the food.",
            "The graphic violence in the movie was too much to bear.",
            "Finding bugs in my food made me lose my appetite instantly.",
            "The filthy state of the hotel room made my skin crawl."
        ]
        
        neutral_texts = [
            "The meeting is scheduled for 3 pm in the conference room.",
            "The book contains 320 pages and was published last year.",
            "They announced that the store will be open until 9 pm.",
            "The weather forecast predicts rain for the weekend.",
            "The article discusses the economic implications of the policy.",
            "The train arrives at the station every hour on the hour.",
            "The museum is located in the center of the city.",
            "The document needs to be signed and returned by Friday.",
            "The software update includes several new features and bug fixes.",
            "The recipe calls for two cups of flour and one teaspoon of salt."
        ]
        
        # Add data for each emotion
        for text in happiness_texts:
            data.append((text, "happiness"))
        
        for text in sadness_texts:
            data.append((text, "sadness"))
        
        for text in anger_texts:
            data.append((text, "anger"))
        
        for text in fear_texts:
            data.append((text, "fear"))
        
        for text in surprise_texts:
            data.append((text, "surprise"))
        
        for text in disgust_texts:
            data.append((text, "disgust"))
        
        for text in neutral_texts:
            data.append((text, "neutral"))
        
        # Save the data to file for future use
        with open(data_path, 'w', encoding='utf-8') as f:
            for text, emotion in data:
                f.write(f"{text}\t{emotion}\n")
        
        return data

def train_emotion_model():
    """
    Train an emotion detection model.
    
    Returns:
        tuple: (model, vectorizer) - trained model and vectorizer
    """
    # Load emotion data
    data = load_emotion_data()
    
    # Extract texts and labels
    texts, emotions = zip(*data)
    
    # Preprocess texts
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # Convert emotion labels to indices
    emotion_labels = ["happiness", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
    y = np.array([emotion_to_idx[emotion] for emotion in emotions])
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000, 
        ngram_range=(1, 2), 
        min_df=1
    )
    X = vectorizer.fit_transform(preprocessed_texts)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Logistic Regression model
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Save the model and vectorizer
    model_file = os.path.join(os.path.dirname(__file__), 'emotion_model.pkl')
    vectorizer_file = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
    
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    return model, vectorizer

if __name__ == "__main__":
    # This will train and save the model if run directly
    train_emotion_model()
