import streamlit as st
import scipy.io.wavfile
from transformers import AutoProcessor,MusicgenForConditionalGeneration
import torch
import scipy.io.wavfile as wavfile
st.title("🎵 AI Music Generator")
st.write("Create original music using AI from your own description")
st.divider()
style = st.selectbox(
    "Choose music style:",
    ["K-pop", "pop", "Cinematic", "Calm", "Electronic"]
)
prompt = st.text_area(
    "Describe your music:",
    placeholder="Example: energetic K-pop instrumental with deep bass and powerful drums"
)
length = st.selectbox("choose music length:",
                      ["short", "medium", "long"])
if "model" not in st.session_state:
    st.session_state["model"] = None
if st.button("🎶 Generate Music"):
    if prompt:
        with st.spinner("Generating your music🎵"):
            processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
            model = MusicgenForConditionalGeneration.from_pretrained(
                "facebook/musicgen-small"
            )
            full_prompt = style + "music," + prompt
            inputs = processor(
                text =[full_prompt],
                padding=True,
                return_tensors="pt"
            )
            max_new_tokens = {
                "short": 128,
                "medium": 256,
                "long": 384
            }[length]
            audio_values = model.generate(
                **inputs,
                max_new_tokens= max_new_tokens
            )
            scipy.io.wavfile.write("generated_music.wav",
                                   rate=model.config.audio_encoder.sampling_rate,
                                   data=audio_values[0,0].numpy()
                                   )
            st.success("Music generated successfully! 🎉")
            with open("generated_music.wav", "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/wav")
                st.download_button(
                    "⬇️ Download Music",
                    audio_bytes,
                    "generated_music.wav",
                    "audio/wav"
                )
                st.divider()
                st.write("🎵 Created with Python, Streamlit and MusicGen")
    else:
        st.warning("please describe the music you want.")