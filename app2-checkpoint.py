import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    st.error(f"NLTK resource download failed: {e}")

ps = PorterStemmer()


def transform_text(text):
    try:
        text = text.lower()
        text = nltk.word_tokenize(text)

        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)
    except Exception as e:
        st.error(f"Error occurred during text transformation: {e}")
        return ""


# Load pre-trained models
try:
    with open('vectorizer.pkl', 'rb') as file:
        tfidf = pickle.load(file)

    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Failed to load pre-trained models: {e}")

st.set_page_config(page_title="SMS Spam Classifier", page_icon=":samsung:")

# Header
st.title("SMS Spam Classifier")
st.markdown("---")

# Subheader
# st.subheader("Classify SMS messages as Spam or Not Spam")
# st.markdown("---")

messages = [
    "         ",
    "Hey John I hope this message finds you well. I wanted to follow up on our conversation from yesterday regarding the upcoming project. I've gone over the details you provided, and I believe we have a solid plan in place.I just wanted to confir",
    "Urgent! Please call 09061213237 from a landline. £5000 cash or a luxury holiday await you!",
    "Congratulations! You've won a guaranteed £1000 cash or a £2000 prize. Text WIN to 123456.",
    "Hello, how are you?",
    "Meeting at 3 pm tomorrow.",
    "Reminder: Your appointment is at 10 am.",
]

st.sidebar.subheader("Select a message:")
selected_message = st.sidebar.selectbox("", messages)

# Display thr message intextbox
input_sms = st.text_area("Enter the message", selected_message)

if st.button('Predict', key='predict_btn'):
    if not input_sms.strip():
        st.warning("Please enter an SMS message.")
    else:
        # Preprocesstext
        transformed_sms = transform_text(input_sms)
        if transformed_sms:
            try:
                # Vectorize
                vector_input = tfidf.transform([transformed_sms])
                # Predict
                result = model.predict(vector_input)[0]
                probabilities = model.predict_proba(vector_input)[0]
                # Display result
                if result == 1:
                    st.markdown("<div style='background-color: red; padding: 10px; border-radius: 5px;'>📩 This is a Spam message.</div>", unsafe_allow_html=True)
                else:
                    st.success("✅ This is not a Spam message.")
                st.write(f"Spam Probability: {probabilities[1]*100:.2f}%")
                st.write(f"Not Spam Probability: {probabilities[0]*100:.2f}%")
               
                st.write("### Explanation:")
                st.write("This prediction is based on a machine learning model trained on a dataset of SMS messages. It analyzes the text content of the message and predicts whether it is likely to be spam or not spam.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
st.markdown(" ")
st.markdown(" ")
st.markdown("  ")
st.markdown(" ")
st.markdown("---")
# st.markdown("Made by: Kanish")
st.markdown("Contact: 9876543210")

st.markdown(
    """
    <style>
    body {
        color: #333;
        background-color: #f8f9fa;
        font-family: Arial, sans-serif;
        line-height: 1.6;
    }
    .stButton>button {
    background-color: #007bff;
    color: #fff;
    border: none;
    border-radius: 25px; /* Adjust border-radius to make the button oval */
    padding: 10px 20px;
    font-size: 16px;
    transition: all 0.3s ease; /* Apply transition to all properties */
}

.stButton>button:hover {
    background-color: #0056b3;
    cursor: pointer;
    transform: scale(1.1); /* Apply scale transformation on hover for an amazing effect */
}
    </style>
    """,
    unsafe_allow_html=True
)
