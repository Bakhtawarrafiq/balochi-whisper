import streamlit as st
import torch
import numpy as np
import librosa
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

# Page configuration
st.set_page_config(
    page_title="Balochi Speech Recognition",
    page_icon="🎙️",
    layout="centered"
)

MODEL_ID = "bakhtawar123/balochi-whisper-small-v3"

# Load the fine-tuned Whisper model
@st.cache_resource(show_spinner="Loading Balochi Whisper model...")
def load_model():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        low_cpu_mem_usage=True
    )
    model.eval()
    return processor, model


st.title("🎙️ Balochi Speech Recognition")

st.write(
    "Convert Balochi speech into text using "
    "a fine-tuned Whisper Small model."
)

st.divider()

# Audio input
uploaded_file = st.file_uploader(
    "Upload a Balochi audio file",
    type=["wav", "mp3", "m4a", "flac"]
)

recorded_audio = st.audio_input("Or record your voice")

audio_file = recorded_audio if recorded_audio else uploaded_file

if audio_file is not None:
    st.audio(audio_file)

    if st.button("Transcribe Audio", type="primary"):
        try:
            with st.spinner("Transcribing Balochi speech..."):
                audio_bytes = audio_file.getvalue()

                # Decode and resample audio to 16 kHz
                import io

                audio, sr = librosa.load(
                    io.BytesIO(audio_bytes),
                    sr=16000,
                    mono=True
                )

                if len(audio) == 0:
                    st.error("The audio file is empty.")
                    st.stop()

                processor, model = load_model()

                inputs = processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt"
                )

                # Match the model's original Urdu decoder setting
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language="ur",
                    task="transcribe"
                )

                with torch.inference_mode():
                    predicted_ids = model.generate(
                        inputs.input_features,
                        forced_decoder_ids=forced_decoder_ids
                    )

                transcription = processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]

                st.session_state["transcription"] = transcription

        except Exception as e:
            st.error(f"Transcription failed: {e}")

if "transcription" in st.session_state:
    st.subheader("Balochi Transcription")

    st.text_area(
        "Recognized Text",
        value=st.session_state["transcription"],
        height=160
    )

    st.download_button(
        "Download Transcription",
        data=st.session_state["transcription"],
        file_name="balochi_transcription.txt",
        mime="text/plain"
    )

st.divider()
st.caption("Powered by Hugging Face Whisper Small")
