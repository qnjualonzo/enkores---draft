import streamlit as st
from googletrans import Translator
from pyAutoSummarizer.base import summarization
import re

translator = Translator()

if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "summarized_text" not in st.session_state:
    st.session_state.summarized_text = ""

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

def translate_and_summarize(input_text, lang_direction="EN to KO"):
    """
    Translates and summarizes the input text based on the selected language direction.
    """
    # Determine source and target languages
    src_lang = "en" if lang_direction == "EN to KO" else "ko"
    tgt_lang = "ko" if lang_direction == "EN to KO" else "en"
    
    # Translate text
    translated_text = translate_text_google(input_text, src_lang, tgt_lang)
    translated_text = add_spaces_between_sentences(translated_text)
    
    # Summarize text
    if lang_direction == "EN to KO":
        summarized_text = summarize_with_pyAutoSummarizer_ko(translated_text, stop_words_lang="ko")
    else:
        summarized_text = summarize_with_pyAutoSummarizer_en(translated_text, stop_words_lang="en")
    
    return translated_text, summarized_text

# Streamlit UI
st.title("EnKoreS")

if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KO"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KO", "KO to EN"])

if lang_direction != st.session_state.lang_direction:
    st.session_state.lang_direction = lang_direction
    st.session_state.input_text = ""
    st.session_state.translated_text = ""
    st.session_state.summarized_text = ""

st.session_state.input_text = st.text_area("Enter text to translate:", value=st.session_state.input_text)

if st.button("Translate and Summarize"):
    if st.session_state.input_text.strip():
        st.session_state.translated_text, st.session_state.summarized_text = translate_and_summarize(
            st.session_state.input_text, lang_direction
        )

if st.session_state.translated_text:
    st.text_area("Translated Text:", value=st.session_state.translated_text, height=150, disabled=True)

if st.session_state.summarized_text:
    st.text_area("Summarized Text:", value=st.session_state.summarized_text, height=150, disabled=True)
