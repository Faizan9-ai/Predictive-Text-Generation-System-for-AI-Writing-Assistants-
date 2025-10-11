import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# ----------------------------------------------------
# Load model and tokenizer
# ----------------------------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("next_word_model.h5")
    tokenizer = joblib.load("tokenizer.pkl")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------
st.title("🧠 Next Word Prediction App")
st.write("Type a few words and the model will predict the next words.")

seed_text = st.text_input("Enter a starting phrase:", "to sherlock holmes she is")
next_words = st.slider("How many words to predict?", 1, 10, 5)

if st.button("🔮 Predict Next Words"):
    if seed_text.strip() == "":
        st.warning("Please enter a valid phrase.")
    else:
        # Prediction function
        def predict_next_word(seed_text, next_words):
            for _ in range(next_words):
                token_list = tokenizer.texts_to_sequences([seed_text])[0]
                token_list = pad_sequences([token_list], maxlen=10, padding='pre')
                predicted = np.argmax(model.predict(token_list, verbose=0))
                output_word = ""
                for word, index in tokenizer.word_index.items():
                    if index == predicted:
                        output_word = word
                        break
                seed_text += " " + output_word
            return seed_text

        result = predict_next_word(seed_text, next_words)
        st.success(result)
