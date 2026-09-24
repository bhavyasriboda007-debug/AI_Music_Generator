from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
print("Loading AI music model...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
prompt = input("Describe the music you want: ")
inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt"
)
audio_values = model.generate(**inputs,max_new_tokens=256)
scipy.io.wavfile.write(
    "generated_music.wav",
rate=model.config.audio_encoder.sampling_rate, data=audio_values[0,0].numpy()    
)
print("Music generated successfully!")
print("Saved as generated_music.wav") 