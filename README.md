# Next Word Prediction using NLP and LSTM

This project uses **Natural Language Processing (NLP)** and **Deep Learning (LSTM)** to predict the next word in a given sentence.

Live demo link deployed on Streamlit - https://nextline.streamlit.app/

Dataset - https://www.kaggle.com/datasets/ronikdedhia/next-word-prediction

---

##  Project Overview
This project focuses on building a Next Word Prediction system — a fundamental Natural Language Processing (NLP) task used in predictive text, chatbots, and typing assistants.

The goal is to train a deep learning model using Long Short-Term Memory (LSTM) networks to predict the most probable next word in a given sentence, based on the context of previous words.

A text corpus (The Adventures of Sherlock Holmes by Arthur Conan Doyle) is used as training data. The text is cleaned, tokenized, and converted into numerical sequences that the model can learn from. The model is trained to understand language patterns and dependencies between words.

The system is deployed as an interactive Streamlit web app, where users can enter any phrase and the model predicts the next few words intelligently.

🧩 Key Features

### 🧩 Technologies Used
- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Joblib

---

## ⚙️ How It Works
1. The dataset is tokenized and converted into sequences.
2. An **LSTM** model learns context from previous words.
3. The Streamlit app lets users type a phrase and generates the next words.

