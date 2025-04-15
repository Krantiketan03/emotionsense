# Text-based Emotion Analyzer

This application analyzes the emotional content of text using Natural Language Processing (NLP) and machine learning. It identifies the primary emotion in a given text and displays the influential keywords that contributed to the detection.

## Features

- Text input interface for paragraphs or sentences
- Minimum word limit validation (at least 5 words)
- NLP-based emotion analysis
- Emotion classification with confidence levels
- Visual display of influential keywords
- Interactive web UI with charts and visualizations

## Emotions Detected

The application can detect the following emotions:
- Happiness
- Sadness
- Anger
- Fear
- Surprise
- Disgust
- Neutral

## Setup Instructions for VS Code

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- VS Code with Python extension (recommended)

### Installation

1. **Clone or download the project files**:
   - Download the project files and extract them to a local folder

2. **Create and activate a virtual environment (recommended)**:
   ```
   python -m venv venv
   ```

   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

3. **Install the required dependencies**:
   ```
   pip install streamlit pandas numpy matplotlib nltk scikit-learn plotly
   ```

4. **Download NLTK resources**:
   Run the following in a Python console:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

5. **Run the application**:
   ```
   streamlit run app.py
   ```
   This will start the Streamlit server and open the application in your default web browser.

## How to Use

1. Enter a paragraph or sentence (minimum 5 words) in the text area
2. Click the "Analyze Emotion" button
3. View the detected emotion, confidence score, and highlighted influential words
4. Explore the emotion probability chart and keyword influence visualizations

## Project Structure

- `app.py`: Main Streamlit application and UI components
- `emotion_analyzer.py`: Core emotion analysis class and methods
- `model_training.py`: Machine learning model training and data management
- `utils.py`: Utility functions for text preprocessing and visualization
- `data/emotions.txt`: Training data for the emotion classification model
- `.streamlit/config.toml`: Streamlit configuration

## How It Works

1. **Text Processing**: Input text is tokenized, cleaned, and preprocessed
2. **Emotion Analysis**: The processed text is analyzed using NLP and machine learning
3. **Keyword Identification**: The system identifies words that influenced the emotion detection
4. **Visualization**: Results are displayed with confidence levels and keyword importance

## Technical Implementation

The application uses:
- Streamlit for the web interface
- NLTK for natural language processing
- scikit-learn for machine learning (LogisticRegression)
- pandas for data management
- plotly for interactive visualizations
- Regular expressions for text processing

The emotion detection model is trained on a dataset of labeled text examples, with each text assigned to one of seven emotion categories.