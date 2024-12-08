import streamlit as st
from googletrans import Translator
from transformers import T5Tokenizer, T5ForConditionalGeneration, MarianMTModel, MarianTokenizer
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

# Initialize T5 model and tokenizer for summarization
summarization_model = T5ForConditionalGeneration.from_pretrained("t5-small")
summarization_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

# Initialize MarianMT for translation (more reliable for multilingual text)
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ko")
translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ko")

def add_spaces_between_sentences(text):
    text = re.sub(r'([.!?])(?=\S)', r'\1 ', text)
    return text

def translate_text(input_text, src_lang, tgt_lang):
    try:
        if src_lang == 'en' and tgt_lang == 'ko':  # English to Korean
            translated = translation_model.generate(**translation_tokenizer(input_text, return_tensors="pt", padding=True))
            translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
        elif src_lang == 'ko' and tgt_lang == 'en':  # Korean to English
            translated = translation_model.generate(**translation_tokenizer(input_text, return_tensors="pt", padding=True))
            translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return ""

def summarize_with_t5(text):
    try:
        # Encode the text using the T5 tokenizer
        inputs = summarization_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        # Generate the summary
        summary_ids = summarization_model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        # Decode the generated summary
        summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
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
        st.session_state.translated_text = translate_text(st.session_state.input_text, src_lang, tgt_lang)
        st.session_state.translated_text = add_spaces_between_sentences(st.session_state.translated_text)
        st.session_state.summarized_text = ""

# Display Translated Text
if st.session_state.translated_text:
    st.text_area("Translated Text:", value=st.session_state.translated_text, height=150, disabled=True)

    # Summarize Button with Progress Indicator
    with st.spinner("Summarizing..."):
        if st.button("Summarize"):
            if st.session_state.translated_text.strip():
                processed_text = add_spaces_between_sentences(st.session_state.translated_text)
                st.session_state.summarized_text = summarize_with_t5(processed_text)

# Display Summarized Text
if st.session_state.summarized_text:
    st.text_area("Summarized Text:", value=st.session_state.summarized_text, height=150, disabled=True)
    
    # Option to Translate Summarized Text
    if st.button("Translate Summarized Text"):
        if st.session_state.summarized_text.strip():
            src_lang = "en" if st.session_state.lang_direction == "KO to EN" else "ko"
            tgt_lang = "ko" if st.session_state.lang_direction == "KO to EN" else "en"
            st.session_state.summarized_text = translate_text(st.session_state.summarized_text, src_lang, tgt_lang)
            st.session_state.summarized_text = add_spaces_between_sentences(st.session_state.summarized_text)

# Session State Initialization Improvements
if lang_direction != st.session_state.lang_direction:
    st.session_state.lang_direction = lang_direction
    st.session_state.input_text = st.session_state.translated_text or ""
    st.session_state.translated_text = st.session_state.translated_text or ""
    st.session_state.summarized_text = st.session_state.summarized_text or ""
