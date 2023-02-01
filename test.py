from streaming_translation.audio2text import async_whisper
import time
import whisper
from streaming_translation.audio2text.byte2np import load_audio
from streaming_translation.text2text import youdao

translator = youdao.translator("en")
with async_whisper.WhisperStream("English") as stream:
    while True:
        segments = stream.convert()
        for segment in segments:
            print(segment)
            print(translator.translate(segment))


whisper.DecodingOptions()
whisper.decode()

# model = whisper.load_model("medium")


# with open("test.wav", "rb") as f:
# 	audio = f.read()

# audio = load_audio(audio)

# # load audio and pad/trim it to fit 30 seconds
# # audio = whisper.load_audio("loopback_record.wav")
# print(type(audio), audio.shape)
# # make log-Mel spectrogram and move to the same device as the model
# # mel = whisper.log_mel_spectrogram(audio).to(model.device)

# # # detect the spoken language
# # _, probs = model.detect_language(mel)
# # print(f"Detected language: {max(probs, key=probs.get)}")

# # # decode the audio
# # options = whisper.DecodingOptions()
# # result = whisper.decode(model, mel, options)

# step = whisper.audio.N_SAMPLES
# i = step

# # while True:
# 	# audio = whisper.pad_or_trim(audio, i)
# 	# print(type(audio), audio.shape)
# 	# i += step
# result = model.transcribe(audio, language="Chinese", no_speech_threshold=0.2, fp16=True)

# # print the recognized text
# print(result)
