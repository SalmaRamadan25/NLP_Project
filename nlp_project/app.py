import streamlit as st
import numpy as np
import random
import re
import joblib
from tensorflow.keras.models import load_model
import time


vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')
model = load_model('chat_model.h5')


data = {
  "data": [
    {
      "cat": "greeting",
      "pattern": [
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "yo", "sup", "what's up", "howdy", "heya", "hiya", "helloo", "hey there",
        "hi there", "hello, how are you?", "hello, I have a question", "hi, I need help"
      ],
      "response": ["Hello! Welcome to Hospital Assistant. How can I help you today?"]
    },
    {
      "cat": "hospital_hours",
      "pattern": [
        "what are your hours?", "when do you open?", "are you open now?",
        "hospital timing?", "can I visit now?", "what time do you close?",
        "when are you available?", "working hours?", "hospital schedule",
        "your opening hours", "hello, what are your hours?", "hi, what time do you open?"
      ],
      "response": [
        "The hospital is open 24/7 for emergencies.\nRegular OPD hours are:\nMonday-Friday: 9:00 AM - 5:00 PM\nSaturday: 9:00 AM - 1:00 PM\nSunday: Closed"
      ]
    },
    {
      "cat": "sunday_closed",
      "pattern": [
        "doctors on sunday", "available doctors on sunday", "which doctors are available on sunday?",
        "any doctor sunday?", "sunday doctor availability", "is hospital open on sunday?",
        "can I book appointment on sunday?", "can I visit on sunday?"
      ],
      "response": [
        "Please note: Sunday is a holiday. If it's urgent, please contact our emergency department which is open 24/7."
      ]
    },
    {
      "cat": "available_doctors",
      "pattern": [
        "who are the doctors?", "available doctors?", "do you have specialists?",
        "which doctors are in today?", "can I see a cardiologist?", "any pediatrician today?",
        "doctor schedule?", "which departments do you have?", "doctor availability?",
        "need an orthopedic doctor", "list of doctors", "what doctors do you have?"
      ],
      "response": [
        "Here are our available doctors:\nDr. Smith - General Medicine (Mon-Fri)\nDr. Johnson - Cardiology (Tue, Thu, Sat)\nDr. Williams - Pediatrics (Mon, Wed, Fri)\nDr. Brown - Orthopedics (Mon, Thu, Sat)\nFor appointments, please specify the doctor or department."
      ]
    },
    {
      "cat": "appointment",
      "pattern": [
        "I want to book an appointment", "can I make an appointment?", "appointment please",
        "need to see a doctor", "book slot with Dr. Smith", "how to schedule a visit?",
        "appointment with cardiologist", "reserve a time", "see doctor tomorrow",
        "how to get an appointment?", "can I schedule a visit?", "I want to see Dr. Williams"
      ],
      "response": [
        "To schedule an appointment, please call our reception at 555-0123 or provide the following details:\n- Preferred doctor\n- Preferred date and time\n- Your contact information"
      ]
    },
    {
      "cat": "emergency",
      "pattern": [
        "it's an emergency", "urgent help", "emergency case", "need help now", "life or death",
        "where is the emergency room?", "call emergency", "can I come to the ER?",
        "911 case", "emergency situation", "I'm in pain now", "need doctor immediately"
      ],
      "response": [
        "If this is a medical emergency, please call 911 immediately or visit our emergency department which is open 24/7."
      ]
    },
    {
      "cat": "goodbye",
      "pattern": [
        "thank you", "thanks", "bye", "see you", "take care", "have a good day",
        "goodbye", "cya", "later", "appreciate it", "thanks a lot", "thank you very much"
      ],
      "response": [
        "You're welcome! Take care and stay healthy.",
        "Goodbye! Feel free to return if you have more questions."
      ]
    },
    {
      "cat": "fallback",
      "pattern": [".*"],
      "response": [
        "I apologize, I didn't understand that. Could you please rephrase your question about hospital timings or available doctors?"
      ]
    }
  ]
}

def chat_response(user_input):
    user_input = user_input.lower()
    split_inputs = re.split(r'[.,;!?]| and ', user_input)
    responses = []

    for part in split_inputs:
        part = part.strip()
        if not part:
            continue
        vec = vectorizer.transform([part]).toarray()
        pred = model.predict(vec)
        intent = label_encoder.inverse_transform([np.argmax(pred)])[0]

        for item in data['data']:
            if item['cat'] == intent:
                response = random.choice(item['response'])
                if response not in responses:
                    responses.append(response)
                break

    if not responses:
        for item in data['data']:
            if item['cat'] == "fallback":
                return random.choice(item['response'])

    return "\n".join(responses)

#html
st.set_page_config(page_title="Hospital Assistant",page_icon="🏥", layout="centered")
st.markdown("<h1 style='text-align: center; color: black;'>🏥Hospital Assistant Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Ask anything about hospital</h2>", unsafe_allow_html=True)
 
#css
st.markdown("""
    <style>
    .stTextInput input {
        padding: 0.75rem;
        border-radius: 10px;
        border: 1px solid #ccc;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        float: right;
    }
    </style>
            
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    response = chat_response(user_input)
    st.session_state.messages.append(("You", user_input))
    st.session_state.messages.append(("Bot", response))

# chat
for sender, msg in st.session_state.messages:
    if sender == "You":
        st.markdown(f"**🧑‍💬 {sender}:** {msg}")
    else:
        st.markdown(f"**🤖 {sender}:** {msg}")

