import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ======================================
# Load Saved Files
# ======================================

model = load_model("emotion_lstm_model.keras")

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)


# ======================================
# Streamlit Page Configuration
# ======================================

st.set_page_config(
    page_title="Emotion Detection using LSTM",
    page_icon="😊",
    layout="centered"
)


# ======================================
# Custom CSS
# ======================================

st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

h1{
    color:#1f4e79;
    text-align:center;
}

.stButton>button{
    background-color:#1f77b4;
    color:white;
    font-size:18px;
    border-radius:10px;
    width:100%;
    height:50px;
}

textarea{
    font-size:18px !important;
}

.result{
    font-size:24px;
    color:green;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)


# ======================================
# Title
# ======================================

st.title("😊 Emotion Detection using LSTM")

st.write(
    "Enter any sentence below and click **Analyze Emotion**."
)

# ======================================
# User Input
# ======================================

user_text = st.text_area(
    "Enter Text",
    height=180
)

# ======================================
# Prediction
# ======================================

if st.button("Analyze Emotion"):

    if user_text.strip() == "":

        st.warning("Please enter some text.")

    else:

        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([user_text])

        # IMPORTANT
        # Replace 79 with your max_length if different
        padded = pad_sequences(
            sequence,
            maxlen=79,
            padding="post"
        )

        # Prediction
        prediction = model.predict(padded)

        predicted_class = np.argmax(prediction)

        emotion = label_encoder.inverse_transform(
            [predicted_class]
        )[0]

        confidence = np.max(prediction) * 100

        # Emoji Dictionary
        emoji_dict = {

            "joy":"😄",
            "happiness":"😁",
            "love":"❤️",
            "sadness":"😢",
            "anger":"😠",
            "fear":"😨",
            "surprise":"😲",
            "hate":"💢",
            "fun":"😂",
            "relief":"😌",
            "empty":"😶",
            "boredom":"🥱",
            "enthusiasm":"🤩",
            "worry":"😟"

        }

        emoji = emoji_dict.get(emotion, "😊")

        st.success("Prediction Completed!")

        st.markdown(
            f"<h2 style='color:green;'>Emotion : {emoji} {emotion.upper()}</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<h3 style='color:blue;'>Confidence : {confidence:.2f}%</h3>",
            unsafe_allow_html=True
        )