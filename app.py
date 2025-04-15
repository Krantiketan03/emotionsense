import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
import re
from emotion_analyzer import EmotionAnalyzer
from utils import preprocess_text, highlight_influential_words
import time

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize emotion analyzer
analyzer = EmotionAnalyzer()

# Page configuration
st.set_page_config(
    page_title="Text-based Emotion Analyzer",
    page_icon="😊",
    layout="wide"
)

# App title and description
st.title("Text-based Emotion Analyzer")
st.markdown("""
This application analyzes the emotional content of your text using Natural Language Processing (NLP).
Enter a paragraph or sentence below, and the app will identify the emotion and highlight influential words.
""")

# Set minimum word limit
MIN_WORDS = 5
st.info(f"Please enter at least {MIN_WORDS} words for accurate analysis.")

# Text input area
text_input = st.text_area("Enter your text here:", height=150, placeholder="Type your paragraph or sentence here (minimum 5 words)...")

# Analyze button with loading state
if st.button("Analyze Emotion"):
    # Check if text is provided
    if not text_input.strip():
        st.error("Please enter some text to analyze.")
    else:
        # Preprocessing and word count validation
        preprocessed_text = preprocess_text(text_input)
        tokens = re.findall(r'\b\w+\b', preprocessed_text)
        word_count = len([token for token in tokens if token.isalpha()])
        
        if word_count < MIN_WORDS:
            st.warning(f"Please enter at least {MIN_WORDS} words. Current count: {word_count}")
        else:
            # Analysis with loading spinner
            with st.spinner('Analyzing the emotional content...'):
                # Simulate processing time to give a more natural feeling
                time.sleep(1)
                
                # Perform analysis
                emotion, confidence, influential_words = analyzer.analyze_emotion(text_input)
                
                # Create two columns for display
                col1, col2 = st.columns([2, 1])
                
                # Display results in left column
                with col1:
                    st.subheader("Analysis Results")
                    st.markdown(f"**Detected Emotion:** {emotion}")
                    st.markdown(f"**Confidence:** {confidence:.2f}%")
                    
                    # Display highlighted text
                    st.subheader("Influential Words")
                    highlighted_text = highlight_influential_words(text_input, influential_words)
                    st.markdown(highlighted_text, unsafe_allow_html=True)
                
                # Display visualizations in right column
                with col2:
                    st.subheader("Emotion Confidence")
                    
                    # Get all emotion probabilities
                    emotion_probs = analyzer.get_emotion_probabilities(text_input)
                    
                    # Create a DataFrame for visualization
                    df = pd.DataFrame({
                        'Emotion': emotion_probs.keys(),
                        'Confidence (%)': [round(v * 100, 2) for v in emotion_probs.values()]
                    })
                    
                    # Sort by confidence
                    df = df.sort_values('Confidence (%)', ascending=False)
                    
                    # Plot using Plotly
                    fig = px.bar(
                        df, 
                        x='Confidence (%)', 
                        y='Emotion',
                        color='Emotion',
                        orientation='h'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display influential words and their weights
                st.subheader("Keyword Influence Analysis")
                
                influential_df = pd.DataFrame({
                    'Word': influential_words.keys(),
                    'Influence Score': influential_words.values()
                })
                influential_df = influential_df.sort_values('Influence Score', ascending=False).head(10)
                
                # Plot using Plotly
                fig = px.bar(
                    influential_df, 
                    x='Word', 
                    y='Influence Score',
                    color='Influence Score',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

# Add information about the app
st.markdown("---")
st.markdown("""
### How It Works
1. **Text Processing**: Your input text is tokenized, cleaned, and preprocessed.
2. **Emotion Analysis**: The processed text is analyzed using NLP and machine learning.
3. **Keyword Identification**: The system identifies words that influenced the emotion detection.
4. **Visualization**: Results are displayed with confidence levels and keyword importance.

### Emotions Detected
- Happiness
- Sadness
- Anger
- Fear
- Surprise
- Disgust
- Neutral
""")
