"""Transcribe browser recordings with the locally installed Whisper model."""

import wave
from io import BytesIO
from pathlib import Path

import streamlit as st

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "whisper-tiny.en"
SAMPLE_RATE = 16_000


@st.cache_resource(show_spinner=False)
def _recognizer():
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Voice input needs the local Whisper model at {MODEL_PATH}.")
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_PATH, local_files_only=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=-1,
    )


def transcribe(recording: bytes) -> str:
    import numpy as np

    with wave.open(BytesIO(recording)) as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError("Expected a 16 kHz mono WAV recording.")
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    if not samples.size:
        return ""
    result = _recognizer()({"raw": samples.astype(np.float32) / 32768, "sampling_rate": SAMPLE_RATE})
    return result["text"].strip()
