import streamlit as st
from googletrans import Translator
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import time

# Initialize the translator
translator = Translator()

# Load T5 model and tokenizer for summarization
summarization_model = T5ForConditionalGeneration.from_pretrained("t5-small")
summarization_tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Initialize session state with history and settings
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "summarized_text" not in st.session_state:
    st.session_state.summarized_text = ""
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KO"
if "translation_history" not in st.session_state:
    st.session_state.translation_history = []
if "summarized_history" not in st.session_state:
    st.session_state.summarized_history = []

def add_spaces_between_sentences(text):
    text = re.sub(r'([.!?])(?=\S)', r'\1 ', text)
    return text

def translate_text_google(input_text, src_lang, tgt_lang):
    try:
        translation = translator.translate(input_text, src=src_lang, dest=tgt_lang)
        return translation.text
    except Exception as e:
        st.error(f"Translation service unavailable: {e}")
        return ""

def summarize_with_t5(text, num_sentences=3):
    try:
        input_text = f"summarize: {text}"
        inputs = summarization_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = summarization_model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
        summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        st.error(f"Summarization failed due to error: {e}")
        return ""

# Streamlit UI Setup
st.title("EnKoreS")
st.sidebar.markdown("### Translation and Summarization App")

# Language Selection
lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KO", "KO to EN"])

# Handle long text inputs by breaking them into chunks
MAX_TEXT_LENGTH = 5000
def chunk_text(text, max_length=MAX_TEXT_LENGTH):
    # Split the text into chunks that do not exceed max_length
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    return chunks

# Clear session state if direction changes
if lang_direction != st.session_state.lang_direction:
    st.session_state.lang_direction = lang_direction
    st.session_state.input_text = ""
    st.session_state.translated_text = ""
    st.session_state.summarized_text = ""

# Input Text Area
st.session_state.input_text = st.text_area("Enter text to translate:", value=st.session_state.input_text)

# Translate Button with Progress Indicator
if st.button("Translate"):
    if st.session_state.input_text.strip():
        with st.spinner("Translating..."):
            src_lang = "en" if st.session_state.lang_direction == "EN to KO" else "ko"
            tgt_lang = "ko" if st.session_state.lang_direction == "EN to KO" else "en"
            st.session_state.translated_text = ""
            st.session_state.summarized_text = ""
            
            # Handle long text by breaking it into chunks
            chunks = chunk_text(st.session_state.input_text)
            for chunk in chunks:
                translated_chunk = translate_text_google(chunk, src_lang, tgt_lang)
                st.session_state.translated_text += translated_chunk + " "
            st.session_state.translated_text = add_spaces_between_sentences(st.session_state.translated_text)
            
            # Save translation history
            st.session_state.translation_history.append(st.session_state.translated_text)

# Display Translated Text
if st.session_state.translated_text:
    st.text_area("Translated Text:", value=st.session_state.translated_text, height=150, disabled=True)

    # Summarize Button with Progress Indicator
    if st.button("Summarize"):
        if st.session_state.translated_text.strip():
            with st.spinner("Summarizing..."):
                processed_text = add_spaces_between_sentences(st.session_state.translated_text)
                st.session_state.summarized_text = summarize_with_t5(processed_text, num_sentences=3)
                
                # Save summarized history
                st.session_state.summarized_history.append(st.session_state.summarized_text)

# Display Summarized Text
if st.session_state.summarized_text:
    st.text_area("Summarized Text:", value=st.session_state.summarized_text, height=150, disabled=True)
    
    # Option to Translate Summarized Text
    if st.button("Translate Summarized Text"):
        if st.session_state.summarized_text.strip():
            with st.spinner("Translating Summarized Text..."):
                src_lang = "en" if st.session_state.lang_direction == "KO to EN" else "ko"
                tgt_lang = "ko" if st.session_state.lang_direction == "KO to EN" else "en"
                st.session_state.summarized_text = translate_text_google(st.session_state.summarized_text, src_lang, tgt_lang)
                st.session_state.summarized_text = add_spaces_between_sentences(st.session_state.summarized_text)

# Show History of Translations and Summaries
st.sidebar.markdown("### Translation History")
if st.session_state.translation_history:
    for i, trans in enumerate(st.session_state.translation_history[-5:], 1):  # Show last 5 translations
        st.sidebar.write(f"**Translation {i}:** {trans[:100]}...")  # Display first 100 chars

st.sidebar.markdown("### Summarized History")
if st.session_state.summarized_history:
    for i, summ in enumerate(st.session_state.summarized_history[-5:], 1):  # Show last 5 summaries
        st.sidebar.write(f"**Summary {i}:** {summ[:100]}...")  # Display first 100 chars
