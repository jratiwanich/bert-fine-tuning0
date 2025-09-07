# main.py
# This script demonstrates using a pretrained Hugging Face pipeline for instant, meaningful NLP output.

from transformers import pipeline

# 1. Load a sentiment analysis pipeline (pretrained model)
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

# 2. Example sentence to analyze
sentence = "Hi, I'm having trouble using my email account."

# 3. Run the sentiment analysis pipeline on the sentence
sentiment_result = sentiment_analyzer(sentence)

# 4. Print the sentiment analysis output (label and score)
print("Sentiment analysis SCORE:", sentiment_result)

# 5. Load a text classification pipeline with the Falconsai/intent_classification model for intent classification
classifier = pipeline(
    "text-classification",
    model="Falconsai/intent_classification",
)

# 6. Run the classifier on the same sentence
classification_result = classifier(sentence)

# 7. Print the intent classification result
print("Intent classification output:", classification_result)
