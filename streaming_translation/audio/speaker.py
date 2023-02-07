import pyaudiowpatch as pyaudio
import numpy as np
import threading

from . import stream


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


class Stream(stream.Stream):
    def __enter__(self):
        self.p = pyaudio.PyAudio()

        speaker = get_speaker()

        self.speaker_sr = int(speaker["defaultSampleRate"])

        self.buffer = np.ndarray(0, dtype=np.float32)
        self.lock = threading.Lock()

        self.stream = self.p.open(
            format=                 pyaudio.paInt16,
            channels=               2,
            rate=                   self.speaker_sr,
            frames_per_buffer=      1024,
            input=                  True,
            input_device_index=     speaker["index"],
            stream_callback=        self.callback
        )

        return self

    def __exit__(self, type, value, traceback):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def read(self) -> np.ndarray:
        with self.lock:
            length = len(self.buffer)
            k = self.speaker_sr / Stream.SAMPLE_RATE
            result = list()
            end = 0
            for group in range(0,  int(length / k)):
                start = round(group * k)
                end = round((group + 1) * k)
                if end == start:
                    end += 1
                if end > length:
                    break
                result.append(np.average(self.buffer[start:end]))
            self.buffer = self.buffer[end:]

        return np.fromiter(result, dtype=np.float32)

    def callback(self, in_data, frame_count, time_info, status):
        with self.lock:
            array = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            one_channel = (array[0::2] + array[1::2]) / 2
            self.buffer = np.concatenate([self.buffer, one_channel])
        return (None, pyaudio.paContinue)


if __name__ == "__main__":
    import time
    with Stream() as stream:
        for i in range(1000000):
            time.sleep(1)
            print(stream.read())
