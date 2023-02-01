import pyaudiowpatch as pyaudio
import time
import wave
from queue import Queue
from io import BytesIO


def get_speaker():
    with pyaudio.PyAudio() as p:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"])
        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                """
                Try to find loopback device with same name(and [Loopback suffix]).
                Unfortunately, this is the most adequate way at the moment.
                """
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                print("Default loopback output device not found.\n\nRun `python -m pyaudiowpatch` to check available devices.\nExiting...\n")
                exit()
    return default_speakers


class Stream:
    def __enter__(self):
        p = pyaudio.PyAudio()
        speaker = get_speaker()

        self.p = p
        self.speaker = speaker
        self.channels = speaker["maxInputChannels"]
        self.rate = int(speaker["defaultSampleRate"])
        self.frames_per_buffer = 1024
        self.input_device_index = speaker["index"]

        self.buffer = bytes()
        self.first_time = None
        self.timestamps = []
        self.buffer_index_offset = 0

        self.stream = p.open(format=pyaudio.paInt16,
                             channels=self.channels,
                             rate=self.rate,
                             frames_per_buffer=self.frames_per_buffer,
                             input=True,
                             input_device_index=self.input_device_index,
                             stream_callback=self.callback)
        return self

    # def clear_buffer(self):
    #     self.buffer = bytes()
    
    def get_buffer(self, time_offset):
        if len(self.timestamps) == 0:
            return bytes()

        while self.timestamps[0]["timestamp"] < time_offset:
            self.timestamps = self.timestamps[1:]
        buffer_index = self.timestamps[0]["buffer_index"] - self.buffer_index_offset

        self.buffer = self.buffer[buffer_index:]
        self.buffer_index_offset = self.timestamps[0]["buffer_index"]

        container = BytesIO()
        wav = wave.open(container, "wb")
        wav.setnchannels(self.channels)
        wav.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wav.setframerate(self.rate)
        wav.writeframes(self.buffer)
        container.seek(0)

        buffer = container.read()
        with open("test.wav", "wb") as f:
            f.write(buffer)

        return buffer

    def callback(self, in_data, frame_count, time_info, status):
        
        adc_time = time_info["input_buffer_adc_time"]

        if self.first_time is None:
            self.first_time = time_info["input_buffer_adc_time"]

        self.timestamps.append({
            "timestamp": adc_time - self.first_time,
            "buffer_index": len(self.buffer)
        })

        self.buffer += in_data

        return (None, pyaudio.paContinue)

    def __exit__(self, type, value, traceback):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
