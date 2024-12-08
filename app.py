import streamlit as st
from googletrans import Translator
from pyAutoSummarizer.base import summarization
import re

translator = Translator()

# Initialize session state
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "summarized_text" not in st.session_state:
    st.session_state.summarized_text = ""
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KO"

def add_spaces_between_sentences(text):
    text = re.sub(r'([.!?])(?=\S)', r'\1 ', text)
    return text

def translate_text_google(input_text, src_lang, tgt_lang):
    try:
        translation = translator.translate(input_text, src=src_lang, dest=tgt_lang)
        return translation.text
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return ""

def summarize_with_pyAutoSummarizer_en(text, num_sentences=3, stop_words_lang='en'):
    try:
        parameters = {
            'stop_words': ['en'],
            'n_words': -1,
            'n_chars': -1,
            'lowercase': True,
            'rmv_accents': True,
            'rmv_special_chars': True,
            'rmv_numbers': False,
            'rmv_custom_words': [],
            'verbose': False
        }
        smr = summarization(text, **parameters)
        rank = smr.summ_ext_LSA(embeddings=False, model='all-MiniLM-L6-v2')
        summary = smr.show_summary(rank, n=num_sentences)
        return summary
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return ""

def summarize_with_pyAutoSummarizer_ko(text, num_sentences=3, stop_words_lang='ko'):
    try:
        parameters = {
            'stop_words': ['ko'],
            'n_words': -1,
            'n_chars': -1,
            'lowercase': True,
            'rmv_accents': False,
            'rmv_special_chars': False,
            'rmv_numbers': False,
            'rmv_custom_words': [],
            'verbose': False
        }
        smr = summarization(text, **parameters)
        rank = smr.summ_ext_LSA(embeddings=False, model='all-MiniLM-L6-v2')
        summary = smr.show_summary(rank, n=num_sentences)
        return summary
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return ""

# Streamlit UI Setup
st.title("EnKoreS")
st.sidebar.markdown("### Translation and Summarization App")

# Language Selection
lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KO", "KO to EN"])

# Clear session state if direction changes
if lang_direction != st.session_state.lang_direction:
    st.session_state.lang_direction = lang_direction
    st.session_state.input_text = ""
    st.session_state.translated_text = ""
    st.session_state.summarized_text = ""

# Input Text Area
st.session_state.input_text = st.text_area("Enter text to translate:", value=st.session_state.input_text)

# Translate Button
if st.button("Translate"):
    if st.session_state.input_text.strip():
        src_lang = "en" if st.session_state.lang_direction == "EN to KO" else "ko"
        tgt_lang = "ko" if st.session_state.lang_direction == "EN to KO" else "en"
        st.session_state.translated_text = translate_text_google(st.session_state.input_text, src_lang, tgt_lang)
        st.session_state.translated_text = add_spaces_between_sentences(st.session_state.translated_text)
        st.session_state.summarized_text = ""

# Display Translated Text
if st.session_state.translated_text:
    st.text_area("Translated Text:", value=st.session_state.translated_text, height=150, disabled=True)

    # Summarize Button
    if st.button("Summarize"):
        if st.session_state.translated_text.strip():
            processed_text = add_spaces_between_sentences(st.session_state.translated_text)

            if st.session_state.lang_direction == "EN to KO":
                st.session_state.summarized_text = summarize_with_pyAutoSummarizer_ko(processed_text, stop_words_lang="ko")
            else:
                st.session_state.summarized_text = summarize_with_pyAutoSummarizer_en(processed_text, stop_words_lang="en")

# Display Summarized Text
if st.session_state.summarized_text:
    st.text_area("Summarized Text:", value=st.session_state.summarized_text, height=150, disabled=True)
    
    # Option to Translate Summarized Text
    if st.button("Translate Summarized Text"):
        if st.session_state.summarized_text.strip():
            src_lang = "en" if st.session_state.lang_direction == "KO to EN" else "ko"
            tgt_lang = "ko" if st.session_state.lang_direction == "KO to EN" else "en"
            st.session_state.summarized_text = translate_text_google(st.session_state.summarized_text, src_lang, tgt_lang)
            st.session_state.summarized_text = add_spaces_between_sentences(st.session_state.summarized_text)
