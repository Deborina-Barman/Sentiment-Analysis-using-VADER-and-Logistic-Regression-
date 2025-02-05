🌟 Sentiment Analysis using VADER and Logistic Regression 🌟

🚀 Overview

This project implements an advanced sentiment analysis model using a combination of:

VADER (Valence Aware Dictionary and sEntiment Reasoner) for rule-based sentiment analysis.

Logistic Regression trained on Reddit comments for machine learning-based predictions.

The model classifies text into Positive, Negative, or Neutral sentiment.

🎯 Features

✅ Text Preprocessing: Cleans text by removing URLs, special characters, stopwords, and applies lemmatization.✅ VADER Sentiment Analysis: Uses NLTK's VADER for quick sentiment scoring.✅ Machine Learning Model: Implements TF-IDF vectorization with Logistic Regression.✅ Model Training & Evaluation: Uses scikit-learn for training and performance assessment.✅ Model Persistence: Saves the trained model and vectorizer using joblib for future use.✅ Gradio Chatbot Interface: Provides an interactive and user-friendly way to analyze sentiment.

📂 Dataset

The dataset used consists of Reddit comments, stored in a CSV file, with pre-cleaned text and labeled sentiment categories.

⚙️ Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Download NLTK Resources

import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

4️⃣ Train the Model

python train_model.py

5️⃣ Launch the Chatbot Interface

python app.py

📌 Usage

🎯 Train the model using train_model.py.
🎯 Use app.py to launch the Gradio interface for sentiment analysis.
🎯 Enter text in the chatbot to get sentiment predictions in real-time.

📁 Project Structure

📂 sentiment-analysis/
│   ├── train_model.py      # Model training script
│   ├── app.py              # Gradio chatbot interface
│   ├── requirements.txt    # Required dependencies
│   ├── tfidf_vectorizer.pkl # Saved vectorizer
│   ├── sentiment_model.pkl  # Saved sentiment model
│   ├── dataset/            # Folder containing dataset
│   ├── README.md           # Project documentation

📊 Model Evaluation

✔ Performance Metrics: The model is evaluated using accuracy and a classification report.
✔ Comparison of VADER and ML predictions to ensure optimal accuracy.

🤝 Contributing

🚀 Feel free to fork the repository, submit pull requests, and enhance the project!

📜 License

📝 This project is licensed under the MIT License.

🌟 If you like this project, don't forget to give it a star ⭐ on GitHub!

