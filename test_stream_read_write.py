from satranscriber.stream import speaker
import time
import wave
import pyaudiowpatch as pyaudio

buffer = bytes()

with speaker.Stream() as stream:
    # speakers = stream.speakers
    channels = stream.channels
    rate = stream.rate
    frames_per_buffer = stream.frames_per_buffer

    time.sleep(10)
    buffer += stream.buffer

print("ok")

# with pyaudio.PyAudio() as p:
#     def callback(in_data, frame_count, time_info, status_flag):
#         global buffer
#         l = frame_count * channels * frames_per_buffer
#         l = min(l, len(buffer))
#         # print(l)
#         res = buffer[:l]
#         buffer = buffer[l:]
#         return (res, pyaudio.paContinue)

#     with p.open(format=pyaudio.paInt16,
#                 channels=channels,
#                 rate=rate,
#                 frames_per_buffer=frames_per_buffer,
#                 output=True,
#                 stream_callback=callback
#                 ) as ss:
#         time.sleep(100)
